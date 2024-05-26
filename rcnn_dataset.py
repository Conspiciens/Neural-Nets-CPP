import os 
import torch 

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

        img = read_image(img_path)
        mask = read_image(mask_path)

        obj_ids = torch.unique(mask)

        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        mask = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(mask)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_ids = index 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        img = tv_tensors.Image(img)

        target = {}
        target["masks"] = tv_tensors.Mask(mask)
        target["labels"] = labels
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["image_id"] = image_ids
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None: 
            img, target = self.transforms(img, target)
    
        return img, target

    def __len__(self): 
        return len(self.imgs)


if __name__ == "__main__":
    dataset = SidewalkDataset("CalPolyRoadDataset/")
    img, target = dataset.__getitem__(0)
    print(target)
    print(img)