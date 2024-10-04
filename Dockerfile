FROM projectmonai/monailabel:latest

COPY apps/monaibundle/requirements.txt /

RUN pip install -r /requirements.txt

# CMD ["monailabel" "start_server" "--app" "apps/monaibundle" "--studies" "datastore" "--conf" "bundles" "IntegrationBundle,SegformerBundle,MedSamBundle" "--conf" "zoo_source" "ngc"]
CMD ["echo", "hooked up"]