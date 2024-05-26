# Uses a pretrained FCN_ResNET50_Weights and Faster R-CNN 
import torch 
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.utils import make_grid
from torchvision.io import read_image 

from PIL import Image 
import matplotlib.pyplot as plt 
from pathlib import Path

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold', 'axes.grid': False})

road1_int = read_image(str(Path('CalPolyRoadDataset/road') / '0.png'))
road2_int = read_image(str(Path('CalPolyRoadDataset/road') / '1.jpg'))
road_int = [road1_int, road2_int]

score_theshold = .8
weights = FCN_ResNet50_Weights.DEFAULT
transforms = weights.transforms(resize_size=None)

rcnn_model = fcn_resnet50(weights=weights)
rcnn_model = rcnn_model.eval()

batch = torch.stack([transforms(d) for d in road_int])
output = rcnn_model(batch)['out']

print(weights.meta["categories"])
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
normalized_masks = torch.nn.functional.softmax(output, dim=1)

road_masks = [
    normalized_masks[img_idx, sem_class_to_idx[cls]]
    for img_idx in range(len(road_int))
    for cls in ('dog', 'boat')
]

show(road_masks)