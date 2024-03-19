import pdb
import glob
import json
import logging
import os
import subprocess
import sys
from typing import Dict, Optional, Sequence

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.data import partition_dataset
from monai.handlers import CheckpointLoader

from monailabel.config import settings
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.class_utils import unload_module
from monailabel.utils.others.generic import device_list, name_to_device

from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask

logger = logging.getLogger(__name__)

class MedSamBundleTrainTask(BundleTrainTask):
    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        enable_tracking=False,
        model_dict_key="model",
        load_strict=False,
    ):
        super().__init__(
            path=path,
            conf=conf,
            const=const,
            enable_tracking=enable_tracking,
            model_dict_key=model_dict_key,
            load_strict=load_strict,
        )

    def _partition_datalist(self, datalist, request, shuffle=False):
        val_split = request.get("val_split", 0.2)
        val_split = 0 # Different than BundleTrainTask._partition_datalist
        logger.info(f"Total Records in Dataset: {len(datalist)}; Validation Split: {val_split}")

        if val_split > 0.0:
            train_datalist, val_datalist = partition_dataset(
                datalist, ratios=[(1 - val_split), val_split], shuffle=shuffle
            )
        else:
            train_datalist = datalist
            val_datalist = None if val_split < 0 else []

        logger.info(f"Total Records for Training: {len(train_datalist)}")
        logger.info(f"Total Records for Validation: {len(val_datalist) if val_datalist else ''}")
        return train_datalist, val_datalist
    

    def __call__(self, request, datastore: Datastore):
        logger.info(f"Train Request: {request}")
        ds = self._fetch_datalist(request, datastore)
        # [{'image': '/sddata/projects/segmentationMonaiLabel/datastore/visivite_GA.png', 'label': '/sddata/projects/segmentationMonaiLabel/datastore/labels/final/visivite_GA.png'}, {'image': '/sddata/projects/segmentationMonaiLabel/datastore/76_year_old_GA.png', 'label': '/sddata/projects/segmentationMonaiLabel/datastore/labels/final/76_year_old_GA.png'}, {'image': '/sddata/projects/segmentationMonaiLabel/datastore/Fundus_photograph_of_Geographic_atrophy.png', 'label': '/sddata/projects/segmentationMonaiLabel/datastore/labels/final/Fundus_photograph_of_Geographic_atrophy.png'}]
        train_ds, val_ds = self._partition_datalist(ds, request)

        max_epochs = request.get("max_epochs", 50)
        pretrained = request.get("pretrained", True)
        multi_gpu = request.get("multi_gpu", True)
        force_multi_gpu = request.get("force_multi_gpu", False)
        run_id = request.get("run_id", "run")
        # Sample Request Body
        # {"val_split": 0.3, "run_id": "0001", "gpus": "0"}

        multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

        gpus = request.get("gpus", "0") #  # Different than BundleTrainTask: gpus = request.get("gpus", "all")
        gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
        multi_gpu = True if force_multi_gpu or multi_gpu and len(gpus) > 1 else False
        logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        # device = name_to_device(request.get("device", "cpu"))
        device = name_to_device(request.get("device", "cuda"))
        logger.info(f"Using device: {device}; Type: {type(device)}")

        tracking = request.get(
            "tracking", "mlflow" if self.enable_tracking and settings.MONAI_LABEL_TRACKING_ENABLED else ""
        )
        tracking = tracking[0] if isinstance(tracking, list) else tracking
        tracking_uri = request.get("tracking_uri")
        tracking_uri = tracking_uri if tracking_uri else settings.MONAI_LABEL_TRACKING_URI
        tracking_experiment_name = request.get("tracking_experiment_name")
        tracking_experiment_name = tracking_experiment_name if tracking_experiment_name else request.get("model")
        tracking_run_name = request.get("tracking_run_name")
        logger.info(f"(Experiment Management) Tracking: {tracking}")
        logger.info(f"(Experiment Management) Tracking URI: {tracking_uri}")
        logger.info(f"(Experiment Management) Experiment Name: {tracking_experiment_name}")
        logger.info(f"(Experiment Management) Run Name: {tracking_run_name}")

        train_handlers = self.bundle_config.get(self.const.key_train_handlers(), [])

        model_filename = request.get("model_filename", "model.pt")
        model_filename = model_filename if isinstance(model_filename, str) else model_filename[0]
        model_pytorch = os.path.join(self.bundle_path, "models", model_filename)

        self._load_checkpoint(model_pytorch, pretrained, train_handlers)

        overrides = {
            self.const.key_bundle_root(): self.bundle_path,
            self.const.key_train_trainer_max_epochs(): max_epochs,
            self.const.key_train_dataset_data(): train_ds,
            self.const.key_device(): device,
            self.const.key_train_handlers(): train_handlers,
        }

        # update config options from user
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                displayable_configs = self.bundle_config.get_parsed_content(k, instantiate=True)
                overrides[k] = {c: request[c] for c in displayable_configs.keys()}

        if tracking and tracking.lower() != "none":
            overrides[self.const.key_tracking()] = tracking
            if tracking_uri:
                overrides[self.const.key_tracking_uri()] = tracking_uri
            if tracking_experiment_name:
                overrides[self.const.key_experiment_name()] = tracking_experiment_name
            if tracking_run_name:
                overrides[self.const.key_run_name()] = tracking_run_name

        # external validation datalist supported through bundle itself (pass -1 in the request to use the same)
        if val_ds is not None:
            overrides[self.const.key_validate_dataset_data()] = val_ds

        # allow derived class to update further overrides
        self._update_overrides(overrides)

        if multi_gpu:
            config_paths = [
                c
                for c in self.const.multi_gpu_configs()
                if os.path.exists(os.path.join(self.bundle_path, "configs", c))
            ]
            if not config_paths:
                logger.warning(
                    f"Ignore Multi-GPU Training; No multi-gpu train config {self.const.multi_gpu_configs()} exists"
                )
                return

            train_path = os.path.join(self.bundle_path, "configs", "monailabel_train.json")
            multi_gpu_train_path = os.path.join(self.bundle_path, "configs", config_paths[0])
            logging_file = os.path.join(self.bundle_path, "configs", "logging.conf")
            for k, v in overrides.items():
                self.bundle_config.set(v, k)
            ConfigParser.export_config_file(self.bundle_config.config, train_path, indent=2)  # type: ignore

            sys.path.insert(0, self.bundle_path)
            unload_module("scripts")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
            logger.info(f"Using CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
            cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={len(gpus)}",
                "-m",
                "monai.bundle",
                "run",
                run_id,  # run_id, user can pass the arg
                "--meta_file",
                self.bundle_metadata_path,
                "--config_file",
                f"['{train_path}','{multi_gpu_train_path}']",
                "--logging_file",
                logging_file,
            ]

            if tracking:
                cmd.extend(["--tracking", tracking])
                if tracking_uri:
                    cmd.extend(["--tracking_uri", tracking_uri])

            self.run_multi_gpu(request, cmd, env)
        else:
            sys.path.insert(0, self.bundle_path)
            unload_module("scripts")
            print("\n\n\nRUN SINGLE GPU\n\n\n")
            pdb.set_trace()
            self.run_single_gpu(request, overrides)

        sys.path.remove(self.bundle_path)

        logger.info("Training Finished....")
        return {}
