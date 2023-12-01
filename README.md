# MONAI Label for Segmentation

This repo shows an example of a MONAI Label (API) Server backend with a React app frontend.

MONAI Label Docs: [https://monai.io/label.html](https://monai.io/label.html)

React: [https://react.dev/](https://react.dev/)

## Table of Contents
- [Purpose](#purpose)
- [Overview](#overview)
- [MONAI Label API Server Docs](#monai-label-api-server-docs)
- [React Front End Docs](#react-frontend-docs)
- [Local Development](#local-development)
- [Web Deployment](#web-deployment)


## Purpose

MONAI Label apps (apps that talk to the MONAI Label API server) allow clinicians to more easily annotate labels for datasets by having a ML model provide an initial prediction for the user to either tweak -- or completely throw away if not useful. This means that, at worst, the MONAI Label doesn't impact the time it takes to annotate at all. But, in practice often saves significant time by providing a good starting point for an annotator to refine upon when labeling.

Applications like these serve to both improve the quality and quantity of labeled medical datasets.

## Overview

The term "MONAI Label app" is defined as a client-server architecture utilizing the MONAI Label API server. For this app, the client is a React frontend, and the server is the MONAI Label API server. Oftentimes, the client is different such as 3DSlicer or OHIF, but the MONAI Label API server does not care what client is consuming requests to it. We choose to use a React front end to provide a web interface for this amazing tool (MONAI Label), improving accessibility and leveraging modern web tools.

The MONAI Label API server is written in Python and exposes routes such as :8000/train or :8000/info, which communicates over HTTP(S?) but over the atypical port 8000 to avoid conflicts. Any client can consume the request by querying the RESTful API.

While other MONAI Label apps might have different tasks (classification, regression, instance segmentation, etc.), this repo focuses on cropped fundus (semantic) segmentation of cup and disc regions.

## MONAI Label API Server Docs

[MONAI Label API Server Docs](apps/monaibundle/)

## React Frontend Docs

[React Frontend Docs](frontend/)

## Local Development

The following describes how to start the application locally, for development purposes.

MONAI Label API Server:

```bash
git clone https://github.com/QTIM-Lab/segmentationMonaiLabel.git
cd segmentationMonaiLabel

# Make a new "datastore" (simply a folder that MONAI Label will populate when you specify it in monailabel start_server ... cmd)
mkdir datastore

# Activate a python environment with MONAI Label required imports (docs say python 3.8 or 3.9, not sure if crucial, I use 3.9.13-16)
# https://docs.monai.io/projects/label/en/latest/installation.html
source venv/bin/activate
# or
pyenv shell 3.9.13
# etc...
# and maybe do the following line because docs also say, but I've seen it's just not necessary usually
# sudo apt install python3-dev

# Update wheel and pip (good practice and specified by MONAI Label installation docs)
pip install --upgrade pip setuptools wheel
# Install python requirements if not already
pip install -r apps/monaibundle/requirements.txt

# Check CUDA is working...
python -c "import torch; print(torch.cuda.is_available())"

# Launch the server with the IntegrationBundle and the SegformerBundle
# (erase IntegrationBundle if don't want, only need Segformer for segmentation)
monailabel start_server --app apps/monaibundle --studies datastore --conf bundles IntegrationBundle,SegformerBundle --conf zoo_source ngc
```

React Frontend:

```bash
# if haven't already...
git clone https://github.com/QTIM-Lab/segmentationMonaiLabel.git
cd segmentationMonaiLabel/frontend
# Use nvm and you don't need sudo -> https://github.com/nvm-sh/nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
nvm install 20.8.0 # Install npm version 20.8.0
## Also the first time you run you won't have pre-trained model "nvidia/mit-b5".
## Make sure you set local_files_only=True in apps/monaibundle/model/SegformerBundle/scripts/net.py
## on your first run and then reset to False afterwards. You will need to download initially.
npm i
npm start
```

## Web Deployment

The following describes how to deploy the (unsecured, development) application to web. It is not secure because it does not use HTTPS, or even any sort of user authentication. For this reason, only public, non-PHI data should be used for the server until you secure your application under a different procedure.

MONAI Label API Server:

1. Start up/create a server/cloud virtual machine (VM) with a GPU (Tesla V100 will work, T4 might, A100 certainly will and better for training, etc)
2. Configure VM to open up port 8000 for Ingress
3. Follow same bash script as local deployment for MONAI Label API Server

React Frontend:

1. Start up/create a server/cloud VM for web-server purposes
2. Configure VM to open up port 80 only for HTTP connection (not HTTPS ports)
3. Follow the bash script below
4. Change the IP address for the React "fetch" calls to the web-server's IP

```bash
# React Frontend bash commands with node and nginx setup

# good practice, don't forget to do on VMs...
sudo apt update

# Get node 18, maybe not best practice, but repeatable for ephemeral dev apps...
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

git clone https://github.com/QTIM-Lab/segmentationMonaiLabel.git

cd segmentationMonaiLabel/frontend
# You will see it has some vulnerabilities associated with nth-check, another reason
# why this is an insecure dev cloud deployment only
npm i
npm run build
# creates build folder in frontend

# get nginx
sudo apt-get install -y nginx
# Copy the build folder over to 
cp ~/segmentationMonaiLabel/frontend/build /var/www/html
# Change the nginx config to point to your build folder
vim /etc/nginx/sites-available/default
# ... change the line where it says "/var/www/html/" to /var/www/html/build/" then :wq

# check for errors and restart
sudo nginx -t
sudo systemctl reload nginx

# app should now be served on http at the domain name, i.e. 34.205.25.54
```
