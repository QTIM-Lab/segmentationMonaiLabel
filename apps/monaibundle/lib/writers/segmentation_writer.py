from typing import Any, Dict, Iterable, List, Optional, Tuple
import tempfile
import numpy as np
import torch
from monai.data import MetaTensor
from monailabel.utils.others.generic import file_ext
import cv2

import logging

logger = logging.getLogger(__name__)

# SEE: https://docs.monai.io/projects/label/en/latest/_modules/monailabel/transform/writer.html

def write_cv2(image_np, output_file, dtype):
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()
    
    # Ensure that the data shape is [classes, height, width]
    if len(image_np.shape) == 3:
        image_np = image_np.transpose(1, 2, 0)

    if dtype:
        image_np = image_np.astype(dtype)

    # Create a 8-bit image (adjust the data type as needed)
    image_np = np.uint8(image_np * 255)

    # Save the image using OpenCV
    cv2.imwrite(output_file, image_np)


class SegmentationWriter:
    def __init__(
        self,
        label="pred",
        json=None,
        ref_image=None,
        key_extension="result_extension",
        key_dtype="result_dtype",
        key_compress="result_compress",
        key_write_to_file="result_write_to_file",
        meta_key_postfix="meta_dict",
        nibabel=False,
    ):
        self.label = label
        self.json = json
        self.ref_image = ref_image if ref_image else label

        # User can specify through params
        self.key_extension = key_extension
        self.key_dtype = key_dtype
        self.key_compress = key_compress
        self.key_write_to_file = key_write_to_file
        self.meta_key_postfix = meta_key_postfix
        self.nibabel = nibabel

    def __call__(self, data) -> Tuple[Any, Any]:
        logger.setLevel(data.get("logging", "INFO").upper())

        path = data.get("image_path")
        ext = file_ext(path) if path else None
        dtype = data.get(self.key_dtype, None)
        # compress = data.get(self.key_compress, False)
        write_to_file = data.get(self.key_write_to_file, True)

        ext = data.get(self.key_extension) if data.get(self.key_extension) else ext
        write_to_file = write_to_file if ext else False
        logger.info(f"Result ext: {ext}; write_to_file: {write_to_file}; dtype: {dtype}")

        if isinstance(data[self.label], MetaTensor):
            image_np = data[self.label].array
        else:
            image_np = data[self.label]

        # Always using Restored as the last transform before writing
        meta_dict = data.get(f"{self.ref_image}_{self.meta_key_postfix}")
        affine = meta_dict.get("affine") if meta_dict else None
        if affine is None and isinstance(data[self.ref_image], MetaTensor):
            affine = data[self.ref_image].affine

        logger.debug(f"Image: {image_np.shape}; Data Image: {data[self.label].shape}")

        output_file = None
        output_json = data.get(self.json, {})
        if write_to_file:
            output_file = tempfile.NamedTemporaryFile(suffix=ext).name
            logger.debug(f"Saving Image to: {output_file}")

            write_cv2(image_np, output_file, dtype)

            # if self.is_multichannel_image(image_np):
            #     if ext != ".seg.nrrd":
            #         logger.warning(
            #             f"Using extension '{ext}' with multi-channel 4D label will probably fail"
            #             + "Consider to use extension '.seg.nrrd'"
            #         )
            #     labels = data.get("labels")
            #     color_map = data.get("color_map")
            #     logger.debug("Using write_seg_nrrd...")
            #     write_seg_nrrd(image_np, output_file, dtype, affine, labels, color_map)
            # # Issue with slicer:: https://discourse.itk.org/t/saving-non-orthogonal-volume-in-nifti-format/2760/22
            # elif self.nibabel and ext and ext.lower() in [".nii", ".nii.gz"]:
            #     logger.debug("Using MONAI write_nifti...")
            #     write_nifti(image_np, output_file, affine=affine, output_dtype=dtype)
            # else:
            #     write_itk(image_np, output_file, affine if len(image_np.shape) > 2 else None, dtype, compress)
        else:
            output_file = image_np

        return output_file, output_json
