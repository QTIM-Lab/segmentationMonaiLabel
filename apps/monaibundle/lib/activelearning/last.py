# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

import random

logger = logging.getLogger(__name__)


class Last(Strategy):
    """
    Consider implementing a first strategy for active learning
    """

    def __init__(self):
        super().__init__("Get Last Sample")

    def __call__(self, request, datastore: Datastore):
        print("LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL ---OMFR!!!!!!!!?S?A??F?FE?EW HEY!")
        unlabeled_images = datastore.get_unlabeled_images()
        labeled_images = datastore.get_labeled_images()
        
        all_images = unlabeled_images + labeled_images

        if not len(all_images):
            return None

        random_choice = random.choice(all_images)

        label = None
        if random_choice in labeled_images:
            # get label
            print('hi')
        
        return {
            "id": random_choice,
            "label": label
        }
