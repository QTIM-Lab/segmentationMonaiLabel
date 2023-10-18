import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


# class CustomSegformerModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class CustomSegformerModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(CustomSegformerModel, self).__init__()

        self.pretrained_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
        self.config = self.pretrained_model.config

        # Modify the configuration to add dropout
        self.config.hidden_dropout_prob = dropout_prob
        self.config.attention_probs_dropout_prob = dropout_prob
        self.config.classifier_dropout_prob = dropout_prob
        self.config.drop_path_rate = dropout_prob

        self.model = SegformerForSemanticSegmentation(config=self.config)
        # self.model.load_state_dict(self.pretrained_model.state_dict())
        self.model.decode_head.classifier = nn.Conv2d(768, 3, kernel_size=1)

    def forward(self, input):
        return self.model(input)
