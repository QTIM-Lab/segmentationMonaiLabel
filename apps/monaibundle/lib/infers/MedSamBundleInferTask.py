import pdb
import copy
import logging
import os
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch, numpy as np
from monai.data import decollate_batch
from monai.inferers import Inferer, SimpleInferer, SlidingWindowInferer
from monai.utils import deprecated

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.utils.transform import dump_data, run_transforms
from monailabel.transform.cache import CacheTransformDatad
from monailabel.transform.writer import ClassificationWriter, DetectionWriter, Writer
from monailabel.utils.others.generic import device_list, device_map, name_to_device

from monailabel.tasks.infer.bundle import BundleInferTask, BundleConstants
from monai.transforms import SaveImaged
from monailabel.tasks.infer.basic_infer import CallBackTypes

from lib.writers.MedSamWriter import MedSamWriter
from monailabel.utils.others.class_utils import unload_module
import sys

from monai.transforms import Compose, LoadImaged

from monailabel.transform.pre import LoadImageTensord


## BB
import torchvision



from PIL import Image

logger = logging.getLogger(__name__)


class MedSamBundleConstants(BundleConstants):
    def key_dataset(self) -> str:
        return ["dataset"]


class MedSamBundleInferTask(BundleInferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        type: Union[str, InferType] = "",
        pre_filter: Optional[Sequence] = None,
        post_filter: Optional[Sequence] = None,
        extend_load_image: bool = True,
        add_post_restore: bool = False,
        dropout: float = 0.0,
        load_strict=False,
        **kwargs,
    ):
        if const is None:
            const = MedSamBundleConstants()
        
        super().__init__(
            path=path,
            conf=conf,
            const=const,
            type=type,
            pre_filter=pre_filter,
            post_filter=post_filter,
            extend_load_image=extend_load_image,
            add_post_restore=add_post_restore,
            dropout=dropout,
            load_strict=load_strict,
            **kwargs,
        )

    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any, Any]:
        """
        You can provide your own writer.  However, this writer saves the prediction/label mask to file
        and fetches result json

        :param data: typically it is post processed data
        :param extension: output label extension
        :param dtype: output label dtype
        :return: tuple of output_file and result_json
        """
        logger.info("Writing Result...")

        writer_obj = MedSamWriter(label='pred')#BB
        return writer_obj(data)


    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        inferer = self.inferer(data) # grabs the inferer and returns it - BB
        # pdb.set_trace()
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")

        network = self._get_network(device, data)
        if network:
            inputs = data[self.input_key]
            # inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
            # inputs = inputs[None] if convert_to_batch else inputs
            # inputs = inputs.to(torch.device(device))

            with torch.no_grad():
                # network(data['image'])
                # network(network._model, data['device'], data['image'])
                br = self.bundle_config.get(self.const.key_bundle_root())
                # network = os.path.join(br, 'models', self.const.model_pytorch())
                network = os.path.join(br, 'models', 'model_best.pt')
                # pdb.set_trace()
                outputs = inferer(inputs, network=network, device=data['device'])
            # pdb.set_trace()

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            if convert_to_batch:
                if isinstance(outputs, dict):
                    outputs_d = decollate_batch(outputs)
                    outputs = outputs_d[0]
                else:
                    outputs = outputs[0]

            data[self.output_label_key] = outputs
        else:
            # consider them as callable transforms
            data = run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
        return data
    
    # from BundleInferTask
    def pre_transforms(self, data=None) -> Sequence[Callable]:
        # Update bundle parameters based on user's option
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                self.bundle_config[k].update({c: data[c] for c in self.displayable_configs.keys()})
                self.bundle_config.parse()


        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        pre = []
        for k in self.const.key_preprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                if isinstance(c, Compose):
                    pre = list(c.transforms)
                elif isinstance(c, torchvision.transforms.Compose):
                    pre = c.transforms
                elif isinstance(c(data['image_path']), torch.utils.data.Dataset):
                    # pdb.set_trace()
                    pre = c(data['image_path'], mode="infer")
                else:
                    pre = c


        # pre = self._filter_transforms(pre, self.pre_filter)
        # for t in pre:
        #     if isinstance(t, LoadImaged):
        #         t._loader.image_only = False
        
        # if pre and self.extend_load_image:
        #     res = []
        #     for t in pre:
        #         if isinstance(t, LoadImaged):
        #             res.append(LoadImageTensord(keys=t.keys, load_image_d=t))
        #         else:
        #             res.append(t)
        #     pre = res
        
        sys.path.remove(self.bundle_path)

        # pdb.set_trace()

        return pre
    

    def __call__(
        self, request, callbacks: Union[Dict[CallBackTypes, Any], None] = None
    ) -> Union[Dict, Tuple[str, Dict[str, Any]]]:

        """
        It provides basic implementation to run the following in order
            - Run Pre Transforms
            - Run Inferer
            - Run Invert Transforms
            - Run Post Transforms
            - Run Writer to save the label mask and result params

        You can provide callbacks which can be useful while writing pipelines to consume intermediate outputs
        Callback function should consume data and return data (modified/updated) e.g. `def my_cb(data): return data`

        Returns: Label (File Path) and Result Params (JSON)
        """
        begin = time.time()
        req = copy.deepcopy(self._config)
        req.update(request)

        # device
        device = name_to_device(req.get("device", "cuda"))

        # device = name_to_device('cpu')
        print(f"\n\n\n\n DEVICE: {device} \n\n\n\n")
        req["device"] = device

        logger.setLevel(req.get("logging", "INFO").upper())
        if req.get("image") is not None and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
            # /tmp/tmpnhmsopkn.png
            # [file for file in os.listdir('/tmp') if file.find('.png') != -1]
        else:
            dump_data(req, logger.level)
            data = req
        # callbacks useful in case of pipeliens to consume intermediate output from each of the following stages
        # callback function should consume data and returns data (modified/updated)
        callbacks = callbacks if callbacks else {}
        callback_run_pre_transforms = callbacks.get(CallBackTypes.PRE_TRANSFORMS)
        callback_run_inferer = callbacks.get(CallBackTypes.INFERER)
        callback_run_invert_transforms = callbacks.get(CallBackTypes.INVERT_TRANSFORMS)
        callback_run_post_transforms = callbacks.get(CallBackTypes.POST_TRANSFORMS)
        callback_writer = callbacks.get(CallBackTypes.WRITER)
        
        start = time.time()
        pre_transforms = self.pre_transforms(data)
        # !next(iter(pre_transforms))
        # pre_transforms.__len__()
        # pre_transforms
        # pdb.set_trace()
        #######
        # data = self.run_pre_transforms(data, pre_transforms)
        # if callback_run_pre_transforms:
        #     data = callback_run_pre_transforms(data)
        latency_pre = time.time() - start
        ###############
        data['image'] =  pre_transforms

        
        start = time.time()
        if self.type == InferType.DETECTION:
            data = self.run_detector(data, device=device)
        else:
            # pdb.set_trace()
            data = self.run_inferer(data, device=device)    
            # data = self.run_inferer(pre_transforms, device=device)    

        if callback_run_inferer:
            data = callback_run_inferer(data)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms(data))
        if callback_run_invert_transforms:
            data = callback_run_invert_transforms(data)
        latency_invert = time.time() - start

        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        if callback_run_post_transforms:
            data = callback_run_post_transforms(data)
        latency_post = time.time() - start

        if self.skip_writer:
            return dict(data)

        start = time.time()

        print("THIS IS NEW!")

        result_file_name, result_json = self.writer(data)
        
        if callback_writer:
            data = callback_writer(data)
        latency_write = time.time() - start

        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Inferer: {:.4f}; Invert: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_inferer,
                latency_invert,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "infer": round(latency_inferer, 2),
            "invert": round(latency_invert, 2),
            "post": round(latency_post, 2),
            "write": round(latency_write, 2),
            "total": round(latency_total, 2),
            "transform": data.get("latencies"),
        }

        # Add Centroids to the result json to consume in OHIF v3
        centroids = data.get("centroids", None)
        if centroids is not None:
            centroids_dict = dict()
            for c in centroids:
                all_items = list(c.items())
                centroids_dict[all_items[0][0]] = [str(i) for i in all_items[0][1]]  # making it json compatible
            result_json["centroids"] = centroids_dict
        else:
            result_json["centroids"] = dict()

        if result_file_name is not None and isinstance(result_file_name, str):
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        # pdb.set_trace()
        
        return result_file_name, result_json
