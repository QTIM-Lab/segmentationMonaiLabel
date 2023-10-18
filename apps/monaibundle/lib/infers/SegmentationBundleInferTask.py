import copy
import logging
import os
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
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

logger = logging.getLogger(__name__)











import glob
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Union

from monai.bundle import ConfigItem, ConfigParser
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Compose, LoadImaged, SaveImaged

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from monailabel.transform.pre import LoadImageTensord
from monailabel.utils.others.class_utils import unload_module
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)



class BundleConstants:

    def configs(self) -> Sequence[str]:
        return ["inference.json", "inference.yaml"]



    def metadata_json(self) -> str:
        return "metadata.json"



    def model_pytorch(self) -> str:
        return "model.pt"



    def model_torchscript(self) -> str:
        return "model.ts"



    def key_device(self) -> str:
        return "device"



    def key_bundle_root(self) -> str:
        return "bundle_root"



    def key_network_def(self) -> str:
        return "network_def"



    def key_preprocessing(self) -> Sequence[str]:
        return ["preprocessing", "pre_transforms"]



    def key_postprocessing(self) -> Sequence[str]:
        return ["postprocessing", "post_transforms"]



    def key_inferer(self) -> Sequence[str]:
        return ["inferer"]



    def key_detector(self) -> Sequence[str]:
        return ["detector"]



    def key_detector_ops(self) -> Sequence[str]:
        return ["detector_ops"]



    def key_displayable_configs(self) -> Sequence[str]:
        return ["displayable_configs"]




class BundleInferTask(BasicInferTask):
    """
    This provides Inference Engine for Monai Bundle.
    """

    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        type: Union[str, InferType] = "",
        pre_filter: Optional[Sequence] = None,
        post_filter: Optional[Sequence] = [SaveImaged],
        extend_load_image: bool = True,
        add_post_restore: bool = True,
        dropout: float = 0.0,
        load_strict=False,
        **kwargs,
    ):
        self.valid: bool = False
        self.const = const if const else BundleConstants()

        self.pre_filter = pre_filter
        self.post_filter = post_filter
        self.extend_load_image = extend_load_image
        self.dropout = dropout

        config_paths = [c for c in self.const.configs() if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no infer config {self.const.configs()} exists")
            return

        sys.path.insert(0, path)
        unload_module("scripts")

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])
        self.bundle_config = self._load_bundle_config(self.bundle_path, self.bundle_config_path)
        # For deepedit inferer - allow the use of clicks
        self.bundle_config.config["use_click"] = True if type.lower() == "deepedit" else False

        if self.dropout > 0:
            self.bundle_config["network_def"]["dropout"] = self.dropout

        network = None
        model_path = os.path.join(path, "models", self.const.model_pytorch())
        print("iiii model path: ", model_path)
        if os.path.exists(model_path):
            print("parsed content: ", )
            network = self.bundle_config.get_parsed_content(self.const.key_network_def(), instantiate=True)[0]
            print("OOOOOO : ", network)
        else:
            model_path = os.path.join(path, "models", self.const.model_torchscript())
            print("XXXXXXX: ", model_path)
            if not os.path.exists(model_path):
                logger.warning(
                    f"Ignore {path} as neither {self.const.model_pytorch()} nor {self.const.model_torchscript()} exists"
                )
                sys.path.remove(self.bundle_path)
                return

        # https://docs.monai.io/en/latest/mb_specification.html#metadata-json-file
        with open(os.path.join(path, "configs", self.const.metadata_json())) as fp:
            metadata = json.load(fp)

        self.key_image, image = next(iter(metadata["network_data_format"]["inputs"].items()))
        self.key_pred, pred = next(iter(metadata["network_data_format"]["outputs"].items()))

        # labels = ({v.lower(): int(k) for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"})
        labels = {}
        for k, v in pred.get("channel_def", {}).items():
            if (not type.lower() == "deepedit") and (v.lower() != "background"):
                labels[v.lower()] = int(k)
            else:
                labels[v.lower()] = int(k)
        description = metadata.get("description")
        spatial_shape = image.get("spatial_shape")
        dimension = len(spatial_shape) if spatial_shape else 3
        type = self._get_type(os.path.basename(path), type)

        # if detection task, set post restore to False by default.
        self.add_post_restore = False if type == "detection" else add_post_restore

        super().__init__(
            path=model_path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            preload=strtobool(conf.get("preload", "false")),
            load_strict=load_strict,
            **kwargs,
        )

        # Add models options if more than one model is provided by bundle.
        pytorch_models = [os.path.basename(p) for p in glob.glob(os.path.join(path, "models", "*.pt"))]
        pytorch_models.sort(key=len)
        self._config.update({"model_filename": pytorch_models})
        # Add bundle's loadable params to MONAI Label config, load exposed keys and params to options panel
        for k in self.const.key_displayable_configs():
            if self.bundle_config.get(k):
                self.displayable_configs = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                self._config.update(self.displayable_configs)

        self.valid = True
        self.version = metadata.get("version")
        sys.path.remove(self.bundle_path)


    def is_valid(self) -> bool:
        return self.valid



    def info(self) -> Dict[str, Any]:
        i = super().info()
        i["version"] = self.version
        return i



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
                pre = list(c.transforms) if isinstance(c, Compose) else c

        pre = self._filter_transforms(pre, self.pre_filter)

        for t in pre:
            if isinstance(t, LoadImaged):
                t._loader.image_only = False

        if pre and self.extend_load_image:
            res = []
            for t in pre:
                if isinstance(t, LoadImaged):
                    res.append(LoadImageTensord(keys=t.keys, load_image_d=t))
                else:
                    res.append(t)
            pre = res

        sys.path.remove(self.bundle_path)
        return pre



    def inferer(self, data=None) -> Inferer:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        i = None
        for k in self.const.key_inferer():
            if self.bundle_config.get(k):
                i = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                break

        sys.path.remove(self.bundle_path)
        return i if i is not None else SimpleInferer()



    def detector(self, data=None) -> Optional[Callable]:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        d = None
        for k in self.const.key_detector():
            if self.bundle_config.get(k):
                detector = self.bundle_config.get_parsed_content(k, instantiate=True)  # type: ignore
                for k in self.const.key_detector_ops():
                    self.bundle_config.get_parsed_content(k, instantiate=True)

                if detector is None or callable(detector):
                    d = detector  # type: ignore
                    break
                raise ValueError("Invalid Detector type;  It's not callable")

        sys.path.remove(self.bundle_path)
        return d



    def post_transforms(self, data=None) -> Sequence[Callable]:
        sys.path.insert(0, self.bundle_path)
        unload_module("scripts")
        self._update_device(data)

        post = []
        for k in self.const.key_postprocessing():
            if self.bundle_config.get(k):
                c = self.bundle_config.get_parsed_content(k, instantiate=True)
                post = list(c.transforms) if isinstance(c, Compose) else c

        post = self._filter_transforms(post, self.post_filter)

        if self.add_post_restore:
            post.append(Restored(keys=self.key_pred, ref_image=self.key_image))

        sys.path.remove(self.bundle_path)
        return post


    def _get_type(self, name, type):
        name = name.lower() if name else ""
        return (
            (
                InferType.DEEPEDIT
                if "deepedit" in name
                else InferType.DEEPGROW
                if "deepgrow" in name
                else InferType.DETECTION
                if "detection" in name
                else InferType.SEGMENTATION
                if "segmentation" in name
                else InferType.CLASSIFICATION
                if "classification" in name
                else InferType.SEGMENTATION
            )
            if not type
            else type
        )

    def _filter_transforms(self, transforms, filters):
        if not filters or not transforms:
            return transforms

        res = []
        for t in transforms:
            if not [f for f in filters if isinstance(t, f)]:
                res.append(t)
        return res

    def _update_device(self, data):
        k_device = self.const.key_device()
        device = data.get(k_device) if data else None
        if device:
            self.bundle_config.config.update({k_device: device})  # type: ignore
            if self.bundle_config.ref_resolver.items.get(k_device):
                self.bundle_config.ref_resolver.items[k_device] = ConfigItem(config=device, id=k_device)

    def _load_bundle_config(self, path, config):
        bundle_config = ConfigParser()
        bundle_config.read_config(config)
        bundle_config.config.update({self.const.key_bundle_root(): path})  # type: ignore
        return bundle_config


























class SegmentationBundleInferTask(BundleInferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        const: Optional[BundleConstants] = None,
        type: Union[str, InferType] = InferType.CLASSIFICATION,
        pre_filter: Optional[Sequence] = None,
        post_filter: Optional[Sequence] = None,
        extend_load_image: bool = True,
        add_post_restore: bool = False,
        dropout: float = 0.0,
        load_strict=False,
        **kwargs,
    ):
        
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

    def __call__(
        self, request, callbacks: Union[Dict[CallBackTypes, Any], None] = None
    ) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        print("entering call for seg bundle infer task !!!!!!!!!!!!!")
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
        req["device"] = device

        logger.setLevel(req.get("logging", "INFO").upper())
        if req.get("image") is not None and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
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

        print("before transforms")
        start = time.time()
        pre_transforms = self.pre_transforms(data)
        data = self.run_pre_transforms(data, pre_transforms)
        if callback_run_pre_transforms:
            data = callback_run_pre_transforms(data)
        latency_pre = time.time() - start
        print("after transforms")

        start = time.time()
        print("before run inferer: ")
        if self.type == InferType.DETECTION:
            data = self.run_detector(data, device=device)
        else:
            data = self.run_inferer(data, device=device)

        print("after run inferrer")
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
        return result_file_name, result_json
    



    def _get_network(self, device, data):
        print("Enter get network")
        path = self.get_path()
        logger.info(f"Infer model path: {path}")

        if data and self._config.get("model_filename"):
            model_filename = data.get("model_filename")
            model_filename = model_filename if isinstance(model_filename, str) else model_filename[0]
            user_path = os.path.join(os.path.dirname(self.path[0]), model_filename)
            if user_path and os.path.exists(user_path):
                path = user_path
                logger.info(f"Using <User> provided model_file: {user_path}")
            else:
                logger.info(f"Ignoring <User> provided model_file (not valid): {user_path}")

        if not path and not self.network:
            if self.type == InferType.SCRIBBLES:
                return None

            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                f"Model Path ({self.path}) does not exist/valid",
            )

        cached = self._networks.get(device)
        statbuf = os.stat(path) if path else None
        network = None
        if cached:
            if statbuf and statbuf.st_mtime == cached[1]:
                network = cached[0]
            elif statbuf:
                logger.warning(f"Reload model from cache.  Prev ts: {cached[1]}; Current ts: {statbuf.st_mtime}")

        if network is None:
            if self.network:
                network = copy.deepcopy(self.network)
                print("network before to: ", network)
                network.to(torch.device(device))

                if path:
                    checkpoint = torch.load(path, map_location=torch.device(device))
                    model_state_dict = checkpoint.get(self.model_state_dict, checkpoint)

                    if set(self.network.state_dict().keys()) != set(checkpoint.keys()):
                        logger.warning(
                            f"Checkpoint keys don't match network.state_dict()! Items that exist in only one dict"
                            f" but not in the other: {set(self.network.state_dict().keys()) ^ set(checkpoint.keys())}"
                        )
                        logger.warning(
                            "The run will now continue unless load_strict is set to True. "
                            "If loading fails or the network behaves abnormally, please check the loaded weights"
                        )
                    network.load_state_dict(model_state_dict, strict=self.load_strict)
            else:
                network = torch.jit.load(path, map_location=torch.device(device))

            if self.train_mode:
                network.train()
            else:
                network.eval()
            self._networks[device] = (network, statbuf.st_mtime if statbuf else 0)

        print('network before return: ', network)
        return network