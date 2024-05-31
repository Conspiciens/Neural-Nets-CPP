import os 
import torch 
import cv2

from torchvision.io import read_image 
from torchvision.ops.boxes import masks_to_boxes 
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class SidewalkDataset():
    ''' 
        Manages the CalPolyRoadDataset and prepares images to be processed
    ''' 
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "sidewalk"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "sidewalkMask"))))


    def __getitem__(self, index): 
        
        img_path = os.path.join(self.root, "sidewalk", self.imgs[index])
        mask_path = os.path.join(self.root, "sidewalkMask", self.masks[index])

        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transforms is not None: 
            img = self.transforms(img)
            mask = self.transforms(mask)
    
        return img, mask

    def __len__(self): 
        return len(self.imgs)


if __name__ == "__main__":
    dataset = SidewalkDataset("CalPolyRoadDataset/")
    img, target = dataset.__getitem__(0)
    print(target)
    print(img)