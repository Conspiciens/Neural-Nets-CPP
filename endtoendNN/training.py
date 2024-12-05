import torch 
import torchvision 

import preprocessing 

from torchvision import transforms 
from nn import End_to_End_NN
from torch import nn


EPOCH = 100
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor()
]) 

def train():
    data = preprocessing.data_processing()

	# Load the Model
    model = End_to_End_NN()
    front, left, right, steering = data[5]
    loss_fn = nn.MSELoss() 

    total_loss = 0
    for epoch in range(EPOCH): 
        for (i, data) in enumerate(data): 
            front_img, left_img, right_img, steering = data

            steering = int(steering)
            steering = torch.tensor([[steering]], dtype=torch.float32) 

            front_img = transform(front_img) 
            left_img = transform(left_img) 
            right_img = transform(right_img)
             
            output = model(front_img) 
            loss = loss_fn(output, steering) 
            loss.backward() 
            total_loss += loss.item() 
           

            output = model(left_img) 
            loss = loss_fn(output, steering) 
            loss.backward() 
            total_loss += loss.item() 

            output = model(right_img) 
            loss = loss_fn(output, steering) 
            loss.backward() 
            total_loss += loss.item() 
    
    


def main(): 
    train()

if __name__ == '__main__': 
    main()
