import torch
from PIL import Image


def inference(net, transforms, filenames):
    for fn in filenames:
        with Image.open(fn) as im:
            tim=transforms(im)
            outputs=net(tim[None])
            _, predictions = torch.max(outputs, 1)
            print(fn, predictions[0].item())
