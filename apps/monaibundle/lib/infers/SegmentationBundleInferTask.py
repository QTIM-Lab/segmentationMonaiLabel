import pdb
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

from lib.writers.segmentation_writer import SegmentationWriter
from monailabel.utils.others.class_utils import unload_module
import sys

from monai.transforms import Compose, LoadImaged

from monailabel.transform.pre import LoadImageTensord
import torchvision.transforms as td_transforms

from PIL import Image

logger = logging.getLogger(__name__)


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
        type: Union[str, InferType] = "",
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
        writer_obj = SegmentationWriter(label=self.output_label_key)
        return writer_obj(data)

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
        # import pdb; pdb.set_trace()
        # device = name_to_device(req.get("device", "cuda"))
        device = name_to_device('cpu')
        print(f"\n\n\n\n DEVICE: {device} \n\n\n\n")
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

        start = time.time()
        print("self pre transforms: ", self.pre_transforms)
        pre_transforms = self.pre_transforms(data)
        print("pre transforms: ", pre_transforms)
        print("data before transforms: ", data)
        # data = self.run_pre_transforms(data, pre_transforms)
        # print("data after transforms: ", data)


        test_transform = td_transforms.Compose([
            # monai.transforms.LoadImage(image_only=True),
            td_transforms.ToTensor(),
            td_transforms.Resize((512,512)),
            td_transforms.Normalize(mean=[0.522, 0.300, 0.167], std=[0.240, 0.189, 0.147])
            # td_transforms.Normalize(mean=[0.524, 0.301, 0.169], std=[0.240, 0.190, 0.148]) # Train statistics (no val!)
        ])
        data_image = Image.open(data['image']).convert('RGB')
        data['image'] = test_transform(data_image)

        print("data after transforms: ", data)

        if callback_run_pre_transforms:
            data = callback_run_pre_transforms(data)
        latency_pre = time.time() - start

        start = time.time()
        if self.type == InferType.DETECTION:
            data = self.run_detector(data, device=device)
        else:
            data = self.run_inferer(data, device=device)
            print(f"run_inferer data returned: \n\n{data}\n\n")

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
