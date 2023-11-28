# Post Deploy
* Changed 34.91.126.112 to localhost in js files
* Uploaded an image and it dropped into datastore/
* Put model_orig.pt (and duplicate called model.pt) in apps/monaibundle/model/SegformerBundle/models
* Need to restart server if you added models
  - Also needed: local_files_only=False from apps/monaibundle/model/SegformerBundle/scripts/net.py
  - Mask prints 254's instead of 255's : apps/monaibundle/model/SegformerBundle/scripts/dataloaders.py
    * Check git diff for changes
* Can access api at /. Enpoints tried successfully:
  *	/info:
    - can see all 4 GPUs
    - can see trainers: int\seg bundle
	- can see int\seg scoring "train_stats"
	- can see datastore and total was 0 and is now 1 after upload image
