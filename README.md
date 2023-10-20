# MONAI Label for Segmentation

This repo shows an example of a MONAI Label (API) Server backend with a React app frontend.

MONAI Label Docs: [https://monai.io/label.html](https://monai.io/label.html)

React: [https://react.dev/](https://react.dev/)

## Starting From Scratch

MONAI Label API Server:

```bash
git clone https://github.com/QTIM-Lab/segmentationMonaiLabel.git
cd segmentationMonaiLabel

# Make a new "datastore" (simply a folder that MONAI Label will populate when you specify it in monailabel start_server ... cmd)
mkdir datastore

# Activate a python environment with MONAI Label required imports
# https://docs.monai.io/projects/label/en/latest/installation.html
source venv/bin/activate

# Install python requirements if not already
pip install -r apps/requirements.txt

# Check CUDA is working...
python -c "import torch; print(torch.cuda.is_available())"

# Launch the server with the IntegrationBundle and the SegformerBundle
# (erase IntegrationBundle if don't want, only need Segformer for segmentation)
monailabel start_server --app apps/monaibundle --studies datastore --conf bundles IntegrationBundle,SegformerBundle --conf zoo_source ngc
```

React Frontend

```bash
cd segmentationMonaiLabel/frontend
npm i
npm start
```

## Purpose

MONAI Label apps (apps that talk to the MONAI Label API server) allow clinicians to more easily annotate labels for datasets by having a ML model provide an initial prediction for the user to either tweak -- or completely throw away if not useful. This means that, at worst, the MONAI Label doesn't impact the time it takes to annotate at all. But, in practice often saves significant time by providing a good starting point for an annotator to refine upon when labeling.

Applications like these serve to both improve the quality and quantity of labeled medical datasets.

## Overview

The term "MONAI Label app" is defined as a client-server architecture utilizing the MONAI Label API Server. For this app, the client is a React frontend, and the server is the MONAI Label API server. Oftentimes, the client is different such as 3DSlicer or OHIF, but the MONAI Label API server does not care what client is consuming requests to it. We choose to use a React front end to provide a web interface for this amazing tool (MONAI Label), improving accessibility and leveraging modern web tools.

The MONAI Label API server is written in Python and exposes routes such as :8000/train or :8000/info, which communicates over HTTP(S?) but over the atypical port 8000 to avoid conflicts. Any client can consume the request by querying the RESTful API.

While other MONAI Label apps might have different tasks (classification, regression, instance segmentation, etc.), this repo focuses on cropped fundus (semantic) segmentation of cup and disc regions.

## Communication Between Client (React) and Server (MONAI Label)

Todo..

## Inference

Todo...

## Training

Todo....
