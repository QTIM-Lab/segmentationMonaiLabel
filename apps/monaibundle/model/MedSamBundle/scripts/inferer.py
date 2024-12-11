import pdb
import torch, numpy as np
from PIL import Image
from monai.inferers import SimpleInferer
from transformers import SamModel, SamProcessor

class MedSamInferer(SimpleInferer):
    """
    MedSamInferer is a subclass of SimpleInferer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Load the MedSAM model
        # pdb.set_trace()
        self._model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base", local_files_only=False)
        self._processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
        model_ckpt_path = "/monailabel/apps/monaibundle/model/MedSamBundle/models/advaith_epoch=30-step=961.ckpt"
        # pdb.set_trace()
        raw_torch_load = torch.load(model_ckpt_path, map_location=torch.device('cuda'))
        # Create a new state dictionary with updated keys
        new_state_dict = {key.replace("model.", ""): value for key, value in raw_torch_load['state_dict'].items()}
        self._model.load_state_dict(state_dict = new_state_dict, strict=True)
        self._model.to('cuda') 



    def forward(self, inputs, device, trained=True):
        # if trained:
        #     print("""

        #     TRAINED

        #     """)
        #     # model_ckpt_path = "/monailabel/apps/monaibundle/model/MedSamBundle/models/epoch=20-step=210.ckpt"
        #     model_ckpt_path = "/monailabel/apps/monaibundle/model/MedSamBundle/models/advaith_epoch=30-step=961.ckpt"
        #     # pdb.set_trace()
        #     raw_torch_load = torch.load(model_ckpt_path, map_location=torch.device('cuda'))
        #      # Create a new state dictionary with updated keys
        #     new_state_dict = {key.replace("model.", ""): value for key, value in raw_torch_load['state_dict'].items()}
        #     self._model.load_state_dict(state_dict = new_state_dict, strict=True)
        # else:
        #     print("""

        #     NOT TRAINED

        #     """)
        #     self._model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
        # pdb.set_trace()
        input_instance = inputs[0]
        pv = input_instance["pixel_values"]
        ib = input_instance["input_boxes"]
        # ib = [0, 0, list(pv.shape)[2], list(pv.shape)[3]]
        # pv.shape
        # pdb.set_trace()
        # pv = input_instance["pixel_values"].reshape((1, 3, 1024, 1024))
        # ib = input_instance["input_boxes"].reshape(1, 1, 4)
        outputs = self._model(pixel_values=pv.to(device), input_boxes=ib.to(device), multimask_output=False)
        ### AV
        logits = outputs.pred_masks
        o_s = input_instance["original_sizes"].cpu()
        r_i_s = input_instance["reshaped_input_sizes"].cpu()
        logits_resized = self._processor.image_processor.post_process_masks(logits, o_s, r_i_s, binarize=False)
        # pdb.set_trace()
        probs = [l.sigmoid().cpu() for l in logits_resized]
        # logits_resized.squeeze().max()
        ### AV
        ### BB
        # logits = outputs.pred_masks
        # l_sig = logits.sigmoid().cpu()
        # logits.squeeze().max()
        # o_s = input_instance["original_sizes"].cpu()
        # r_i_s = input_instance["reshaped_input_sizes"].cpu()
        # # pdb.set_trace()
        # probs = self._processor.image_processor.post_process_masks(l_sig, o_s, r_i_s, binarize=False)
        ### BB
        binary_mask = (probs[0] > 0.50).int() * 255
        tmp = Image.fromarray(np.uint8(np.array(binary_mask.squeeze()))).convert('RGB')
        # tmp.save("/scratch90/bb-10_25_2024/segmentationMonaiLabel/tmp.png")
        # max(probs[0])
        # np.unique(binary_mask)
        # np.unique(probs)
        # tmp = Image.fromarray(np.uint8(np.array(binary_mask.squeeze()))).convert('RGB')
        # tmp.save("/sddata/projects/segmentationMonaiLabel/tmp_inferer.png")

        return binary_mask



    def __call__(self, inputs, network=None, *args, **kwargs):
        """Unified callable function API of Inferers.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        # if network is not None:
        #     pdb.set_trace()
        #     print("LOADING model_best.pt")
        #     self._model.load_state_dict(torch.load(network))
        device = kwargs.get('device', None) if kwargs else 'cpu'
        print("Device:", device)
        # pdb.set_trace()
        return self.forward(inputs, device, trained=kwargs.get('trained', True))
        # Below will be run from SimpleInferer and we are overriding
        # return super().__call__(inputs, network, *args, **kwargs)