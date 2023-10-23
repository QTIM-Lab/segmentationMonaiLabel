# Custom BundleTrainTasks

A custom Python class that inherits from a BundleTrainTask

## SegmentationBundleTrainTask

This custom BundleTrainTask exists for 1 reason:

1. Because I wanted to not have a validation split for training (data science concept), and didn't know how to more properly configure it

BundleTrainTask docs: [https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask](https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask)

When looking at the docs, there is a def config function, where val_split is set to 0.2. Could try changing that to get rid of val_datasets. Could try changing it to -1 and changing in the YAML probably something like val_split as a variable to 0

In normal/hypothetical future setup, almost certainly would want to retain a val_dataset when training on the labeled images in the datastore, because that is just good data science practice. We don't do it here because we want to show it improves on these examples, and wouldn't want one to be in the val set and have us thinking we trained on our label on it (for demo purposes in this case). But in the future we also may want the val_split to be 0.15 or something (kind of depends on how much data you have and the distribution etc) so good to investigate but overriding the _partition_datalist function works just as well...
