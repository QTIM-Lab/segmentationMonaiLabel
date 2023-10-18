# Start

```bash
cd ~/Documents/monai_label

source venv/bin/activate

cd ~/Documents/segmentationMonaiLabel

monailabel start_server --app apps/monaibundle --studies datastore --conf bundles IntegrationBundle,SegformerBundle --conf zoo_source ngc
```

new terminal

```bash
cd ~/Documents/segmentationMonaiLabel/frontend
npm start
```