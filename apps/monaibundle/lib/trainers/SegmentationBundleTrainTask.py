
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

class SegmentationBundleTrainTask(BundleTrainTask):
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
        val_split = 0
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