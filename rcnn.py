import torch 
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import rcnn_dataset
import numpy as np

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.utils import make_grid
from torchvision.io import read_image 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor 
from torchvision.transforms import v2 as T

from pytorch_helpers import utils
from pytorch_helpers import engine 
from PIL import Image 
import matplotlib.pyplot as plt 
from pathlib import Path

# https://debuggercafe.com/road-pothole-detection-with-pytorch-faster-rcnn-resnet50/
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Implement road mask detection with a Mask R-CNN pretrained 

dataset_directory = "CalPolyRoadDataset/"

# setup model 
def get_model_instance_segmentation(num_classes): 
   model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT") 

   in_features = model.roi_heads.box_predictor.cls_score.in_features

   print(in_features)

   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

   in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
   hidden_layer = 256

   model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 
        hidden_layer, 
        num_classes
   )

   return model


def get_transform(train): 
    ''' 
        Transform the Image with the current selected commands
    ''' 
    transforms = [] 
    if train: 
        # 50% probablity of the image being flipped
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    # torch convert tv_tensor to float and are expected to have values [0, 1]?
    transforms.append(T.ToDtype(torch.float, scale=True))
    # torch convert tensors to Pure Tensors remving any metadeta
    transforms.append(T.ToPureTensor())

    # Componses all the transforms command above
    return T.Compose(transforms)

def training_rcnn(): 
    ''' 
        Fine-Tunning RCNN Model with CalPolyRoadDataset to determine whether it's a sidewalk
    ''' 
    # Check if cuda devices exists otherwise use cpu 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    dataset = rcnn_dataset.SidewalkDataset('CalPolyRoadDataset/', get_transform(train=True)) 
    dataset_test = rcnn_dataset.SidewalkDataset('CalPolyRoadDataset/', get_transform(train=False))

    print(len(dataset))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-40])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 4,
        shuffle = True, 
        num_workers = 0,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size = 2, 
        shuffle = False, 
        num_workers = 0, 
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.1,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)

    print("That's it!")
    torch.save(model.state_dict(), "cifar10model.pth")
    test_rcnn(model, device)


def test_rcnn(model, device): 
    ''' 
        Evaluate RCNN Model with current image selected
    ''' 
    image = read_image("CalPolyroadDataset/sidewalk/10.png")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"Road: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image_1 = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(6, 6))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()

def test_saved_rcnn(): 
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load("cifar10model.pth"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_rcnn(model, device)
    

training_rcnn()