import torch
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image 
import matplotlib.pyplot as plt 
import json

# From Pytorch scratch without any models type.

batch_size = 32
dataset_dir = "CalPolyRoadDataset/" 

classes = ['road', 'curbs', 'tree', 'sidewalk', 'vehicle']

class NeuralNetwork(nn.Module): 
    '''
        Convulational Neural Network 
    ''' 
    
    def __init__(self): 
        super(NeuralNetwork, self).__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=(3,3), padding = 3), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)), 
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(2, 2), padding = 3), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)), 
            torch.nn.Flatten(), 
            torch.nn.Linear(147456, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2) 
        )

    def forward(self, data): 
        out = self.main(data)
        return F.softmax(out, dim=1) 


def read_json(name): 
    json_file = open(dataset_dir + "/" + name + '.json') 

    data = json.load(json_file)

    print(data['shapes'][0]['points'])

    json_file.close() 

def test_cnn(): 
    transform = transforms.Compose([
        transforms.Resize((180, 180)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.491, 0.468], 
            std = [0.197, 0.195, 0.196]
        )
    ])

    dataset = torchvision.datasets.ImageFolder(
        root = dataset_dir,
        transform = transform
    )

    print(dataset.classes)
def train_cnn(): 
    '''
    '''

    # Implement Normalization in the Transforms compose function 
    # Reszie the image 180x180 
    # Transfrom the image from [0, 255] to [0, 1]
    # Normalize by output[channel] = (input[channel] - mean[channel]) / sd[channel]

    # Mean [0.500492513179779, 0.49119654297828674, 0.4684107005596161]
    # Standard Deviation [0.19734451174736023, 0.19522404670715332, 0.19612807035446167]
    transform = transforms.Compose([
        transforms.Resize((180, 180)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.491, 0.468], 
            std = [0.197, 0.195, 0.196]
        )
    ])

    dataset = torchvision.datasets.ImageFolder(
        root = dataset_dir,
        transform = transform
    )

    print(dataset.classes)

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size]
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset, 32, 
        shuffle = True, 
    )

    data_loader_validation = torch.utils.data.DataLoader(
        dataset, 32, 
        shuffle = True,
    )

    network = NeuralNetwork() 
    # summary(network, (3, 180, 180))

    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.1)

    acc = 0 
    count = 0

    # Optimize the model for 9 epochs
    n_epochs = 100
    total_loss = 0
    loss_hist = [0] * n_epochs
    accuracy_hist = [0] * n_epochs


    for epoch in range(n_epochs): 
        for img, labels in data_loader_train: 
            pred = network.forward(img)
            loss = nn.CrossEntropyLoss()(pred, labels)

            # back prop
            loss.backward() 
            # updates the weights 
            optimizer.step() 
            optimizer.zero_grad() 

            loss_hist[epoch] += loss.item() * labels.size(0)
            is_correct = (torch.argmax(pred, dim=1) == labels).float() 
            accuracy_hist[epoch] += is_correct.sum() 
            
        loss_hist[epoch] /= len(data_loader_train.dataset)
        accuracy_hist[epoch] /= len(data_loader_train.dataset)
        with torch.no_grad(): 
            for inputs, labels in data_loader_validation: 
                y_pred = network.forward(inputs)
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
        
        print(f"End of {epoch}, accuracy {(acc / count) * 100}")
        

    acc /= count 
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

    transform = transforms.Compose([
        transforms.Resize((180, 180)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.491, 0.468], 
            std = [0.197, 0.195, 0.196]
        )
    ])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_hist, lw = 3)
    ax.set_title("Training Loss", size = 15)
    ax.set_xlabel("Epoch", size = 15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_hist, lw=3)
    ax.set_title("Training Accuracy", size=15)
    ax.set_xlabel("Epoch", size=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    plt.show()


    img = Image.open(eval_dir + "image1034.jpg")
    inp = transform(img)
    inp = inp.unsqueeze(0)
    network.eval()
    output = network(inp)
    print(torch.argmax(y_pred, 1))
    print(output)


    torch.save(network.state_dict(), "cifar10model.pth")


read_json('0')
