import torch 
import torchvision 

import preprocessing 

from torchvision import transforms 
from nn import End_to_End_NN


transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor()
]) 

def train():
    preprocess = preprocessing.data_processing()

	# Load the Model
    model = End_to_End_NN()

    front, left, right, steeering = preprocess[5]
    front = transform(front)
    print(front.size)
    model.forward(front)

	
    


def main(): 
    train()

if __name__ == '__main__': 
    main()
