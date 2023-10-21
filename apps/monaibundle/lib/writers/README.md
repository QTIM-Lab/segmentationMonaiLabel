# Custom (output) Writers

A custom Python class that resembles (does not inherit) a Writer/its similar cousins here:

[https://docs.monai.io/projects/label/en/latest/_modules/monailabel/transform/writer.html](https://docs.monai.io/projects/label/en/latest/_modules/monailabel/transform/writer.html)

## Overview

Writers are needed for BundleInferTasks (or more broadly, InferTasks) to take the output from the model, i.e.:

```python
# also referred to as predictions
output = model(input)
```

And turn that into something that the API server can send over a curl command/http

## SegmentationWriter

Because the other Writer classes like ClassificationWriter did not inherit from Writer, this class does not either. Instead, it simply defnes a call function which returns the path to the saved image (or image_np, the image as a np array) and json, and is returned to client via API :8000/infer

Not entirely sure how/why this works fully, would have to follow the stack trace further. It was derived from the original "Writer" class, instead changing the call function to write with cv2 (opencv-python) the prediction output (which happens to be RGB, but would probably work fine with more/less channels too???). Otherwise, was basically the same.
