# Custom BundleInferTasks

A custom Python class that inherits from a BundleInferTask

## SegmentationBundleInferTask

This custom BundleInferTask exists for 2 reasons:

1. Needed a custom writer to writer.

The output from our model because the original one just outputted .nii.gz's and maybe didn't hook up properly, but I just also wanted more control over how we were giving it back to our client website and how it was working with the output from the model, and to output as .png.

[More details on the README for writers here](../writers)

2. A "hack" to get my PyTorch transforms be applied to the input image data, instead of having to do it from the YAML

If you look at the call function, I print out the transforms (called "pre_transforms" because applied to input data before inference (prediction), as opposed to post transforms which are applied after inference) as they were from the YAML file, and didn't like them because, while they were VERY similar, I noticed it would output slightly different values and needed the original PyTorch transforms, not the seemingly identical MONAI transforms. I then just "hack" into it by manually setting them and applying them to the image data and saving it, which the original transforms did (but applied to the "image" key as seen in the hashed out code in the [inference.yaml](../../model/SegformerBundle/configs/inference.yaml) where it says @image which is just 'image' and applied to keys of an anticipated dict, versus PyTorch which doesn't have that concept and hence we have to apply to a tensor and grab it from data\['image'\] then save it back there)
