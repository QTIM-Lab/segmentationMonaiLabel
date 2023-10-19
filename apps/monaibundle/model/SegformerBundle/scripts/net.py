import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCustomNet(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()

        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", local_files_only=True
        )
        # Get the model's configuration
        config = pretrained_model.config
        # Modify the configuration to add dropout
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        config.classifier_dropout_prob = dropout_prob
        config.drop_path_rate = dropout_prob

        # Initialize a new model with the modified configuration
        self._model = SegformerForSemanticSegmentation(config=config)

        # Load the pretrained weights on the class itself
        self._model.load_state_dict(pretrained_model.state_dict())

        # Replace the decoder head classifier with a new one
        self._model.decode_head.classifier = nn.Conv2d(768, 3, kernel_size=1)

    def forward(self, input):
        pred = self._model(input).logits
        return nn.functional.interpolate(pred, size=input.shape[-2:], mode="bilinear", align_corners=False)

    def state_dict(self, *args, **kwargs):
        """Removes the _model. prefix from the keys of the state dict."""
        state_dict = super().state_dict(*args, **kwargs)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len("_model."):]
            new_state_dict[new_key] = value
        return new_state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Adds the _model. prefix to the keys of the state dict."""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = "_model." + key
            new_state_dict[new_key] = value
        return super().load_state_dict(new_state_dict, *args, **kwargs)
