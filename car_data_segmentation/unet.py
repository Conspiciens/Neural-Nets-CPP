import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import unet_dataset
import matplotlib.pyplot as plt
import cv2

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

# Credit for this model architecture goes to https://arxiv.org/abs/1505.04597
# https://github.com/karma218/autonomousvehiclelab/blob/main/workspaces/isaac_ros-dev/src/road_segmentation/road_segmentation/road_segmentation_node.py
# semantic segmentation

num_classes = 2

class EncoderBlock(nn.Module):        
    # Consists of Conv -> ReLU -> MaxPool
    def __init__(self, in_chans, out_chans, layers=2, sampling_factor=2, padding="same"):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(in_chans, out_chans, 3, 1, padding=padding))
        self.encoder.append(nn.ReLU())
        for _ in range(layers-1):
            self.encoder.append(nn.Conv2d(out_chans, out_chans, 3, 1, padding=padding))
            self.encoder.append(nn.ReLU())
        self.mp = nn.MaxPool2d(sampling_factor)
    def forward(self, x):
        #print("Encoder forward", x.shape)
        for enc in self.encoder:
            x = enc(x)
        mp_out = self.mp(x)
        return mp_out, x

class DecoderBlock(nn.Module):
    # Consists of 2x2 transposed convolution -> Conv -> relu
    def __init__(self, in_chans, out_chans, layers=2, skip_connection=True, sampling_factor=2, padding="same"):
        super().__init__()
        skip_factor = 1 if skip_connection else 2
        self.decoder = nn.ModuleList()
        self.tconv = nn.ConvTranspose2d(in_chans, in_chans // 2, sampling_factor, sampling_factor)

        self.decoder.append(nn.Conv2d(in_chans // skip_factor, out_chans, 3, 1, padding=padding))
        self.decoder.append(nn.ReLU())

        for _ in range(layers-1):
            self.decoder.append(nn.Conv2d(out_chans, out_chans, 3, 1, padding=padding))
            self.decoder.append(nn.ReLU())

        self.skip_connection = skip_connection
        self.padding = padding
    def forward(self, x, enc_features=None):
        x = self.tconv(x)
        if self.skip_connection:
            if self.padding != "same":
                # Crop the enc_features to the same size as input
                w = x.size(-1)
                c = (enc_features.size(-1) - w) // 2
                enc_features = enc_features[:,:,c:c+w,c:c+w]
            x = torch.cat((enc_features, x), dim=1)
        for dec in self.decoder:
            x = dec(x)
        return x

class UNet(nn.Module):
    def __init__(self, nclass=1, in_chans=1, depth=5, layers=2, sampling_factor=2, skip_connection=True, padding="same"):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        out_chans = 64
        for _ in range(depth):
            self.encoder.append(EncoderBlock(in_chans, out_chans, layers, sampling_factor, padding))
            in_chans, out_chans = out_chans, out_chans*2

        out_chans = in_chans // 2
        for _ in range(depth-1):
            self.decoder.append(DecoderBlock(in_chans, out_chans, layers, skip_connection, sampling_factor, padding))
            in_chans, out_chans = out_chans, out_chans//2
        # Add a 1x1 convolution to produce final classes
        self.logits = nn.Conv2d(in_chans, nclass, 1, 1)

    def forward(self, x):
        #print("Forward shape ", x.shape)
        encoded = []
        for enc in self.encoder:
            x, enc_output = enc(x)
            encoded.append(enc_output)
        x = encoded.pop()
        for dec in self.decoder:
            enc_output = encoded.pop()
            x = dec(x, enc_output)

        # Return the logits
        #print("Logits shape ", self.logits(x).shape)
        return self.logits(x)


def train_u_net(): 
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor() 
    ])

    train_dataset = unet_dataset.SidewalkDataset("../CalPolyRoadDataset/", transform)
    val_dataset = unet_dataset.SidewalkDataset("../CalPolyRoadDataset/", transform)

    indices = torch.randperm(len(train_dataset)).tolist()
    dataset = torch.utils.data.Subset(train_dataset, indices[:-70])
    dataset_test = torch.utils.data.Subset(val_dataset, indices[-10:])

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 4,
        shuffle = True, 
        num_workers = 0,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size = 2, 
        shuffle = False, 
        num_workers = 0, 
    )

    H = {"train_loss": [], "test_loss": []}

    unet = UNet().to('cpu')
    lossFunc = BCEWithLogitsLoss() 
    opt = Adam(unet.parameters(), lr=0.001)

    train_steps = len(train_dataset)
    test_steps = len(val_dataset) 

    for i in range(train_steps):
        unet.train()

        totalTrainLoss = 0 
        totalTestLoss = 0 

        for (i, (x, y)) in enumerate(data_loader): 

            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad() 
            loss.backward() 
            opt.step() 

            totalTrainLoss += loss
        
        with torch.no_grad(): 
            unet.eval()
            for (x, y) in data_loader_test:
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)


        avgTrainLoss = totalTrainLoss / train_steps
        avgTestLoss = totalTestLoss / test_steps

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.show()

    torch.save(unet.state_dict(), "cifar10model.pth")
    make_prediction(unet, "../CalPolyRoadDataset/sidewalk/6.png")

# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
def make_prediction(model, imgPath): 
    model.eval() 

    with torch.no_grad(): 

        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        image = cv2.resize(image, (224, 224))
        orig = image.copy()

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to('cpu')

        print(image.size())

        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor() 
        ])

        image = transform(image.squeeze()).unsqueeze(1)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        print(predMask)
        predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
        print(predMask)
        predMask = (predMask > 0.50) * 255
        predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
        prepare_plot(orig, predMask)


def prepare_plot(origImage, predMask):
    # initialize our figure
    ogMask = cv2.imread("../CalPolyRoadDataset/sidewalkMask/6.png", 0)
    ogMask = cv2.resize(ogMask, ((224, 224)))

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(ogMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    plt.show()

if __name__ == '__main__': 
    train_u_net()