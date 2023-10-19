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
