import cv2 
import einops 
import numpy as np 
import pickle 
import socket 
import threading 
import netifaces
import struct
from sys import platform 

import torch
import torchvision 
import torchvision.transforms as transforms 
import torch.optim as optim 
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image 
from torchvision.transforms import v2 as T

model = None 
isModelEval = False
sock = None
device = None 
conn = None
addr = None

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
    def __init__(self, nclass = 1, in_chans = 3, depth=5, layers=2, sampling_factor=2, skip_connection=True, padding="same"):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        out_chans = 64
        for _ in range(depth):
            self.encoder.append(EncoderBlock(in_chans, out_chans, layers, sampling_factor, padding))
            in_chans, out_chans = out_chans, out_chans * 2

        out_chans = in_chans // 2
        for _ in range(depth-1):
            self.decoder.append(DecoderBlock(in_chans, out_chans, layers, skip_connection, sampling_factor, padding))
            in_chans, out_chans = out_chans, out_chans // 2

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

def init_model(): 

    global model 
    global device

    model = UNet()

    model.load_state_dict(torch.load("cifar10model.pth"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run_model(frame): 
    ''' 
        Evaluate U-Net Model with current image selected
    ''' 
    global model 
    global isModelEval
    global device

    orig = None 
    with torch.no_grad():
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        image = cv2.resize(image, (224, 224))
        orig = image.copy()

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to('cpu') 

        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
        predMask = (predMask > 0.10) * 255
        predMask = predMask.astype(np.uint8)


        predMask = torch.from_numpy(predMask).unsqueeze(0).repeat(3, 1, 1)
        predMask = torch.gt(predMask, 0)
        
        orig = orig.transpose(2, 0, 1)
        orig = torch.from_numpy(orig)

        orig = draw_segmentation_masks(orig, predMask, alpha=0.5, colors="blue")

    orig = orig.numpy()
    cv2_image = np.transpose(orig, (1, 2, 0)) 
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    return cv2_image

def check_for_folder() -> int: 
    '''
        Check if dataset folder is created otherwise create folder, returns current file count in folder 
        @params: None 
        @return: int 
    '''
    FOLDER = 'personal_dataset' 

    if os.path.exists(FOLDER) == True: 
        return len(os.listdir(FOLDER))
    
    os.mkdir(FOLDER)

    return len(os.listdir(FOLDER))
        

def get_private_ip(interface='wlan0') -> str: 
    '''
        Check if private ip address avaliable so it can wait for a client connection 
        @params: string
        @return: string
    '''
    address = None 
    if platform == "linux" or platform == "linux2": 
        address = netifaces.ifaddresses(interface)
    elif platform == "darwin": 
        interface = 'en0'
        address = netifaces.ifaddresses(interface)

    if netifaces.AF_INET in address: 
        print(address[netifaces.AF_INET][0]['addr'])
        return str(address[netifaces.AF_INET][0]['addr'])
    else: 
        return ""

def connect_to_client(private_ip: str) -> None: 
    '''
        Begin waiting for a connection on the given private ip 
        @params: string 
        @return: None 
    '''
    global sock
    global conn
    global addr
    PORT = 10050

    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(( private_ip, PORT ))
        print('Listening at {}'.format(sock.getsockname()))
        sock.listen(1)
        conn, addr = sock.accept() 

        # Removed sock as global for multi-threading i/o task
        # return sock

    except socket.error as e: 
        raise

def server_camera() -> None:
    '''
        Server Camera that waits for connection to send the client the frame 
    '''
    global file_count 
    global sock

    # file_count = check_for_folder() 
    private_ip = get_private_ip() 
    init_model()

    # 192.168.1.25 -> Raspberry pi 4 

    # Make a socket for connection at Host and port, wait for connection
    proceed = True
    conn = None 
    addr = None 
    clientthread = None
    try: 
        if private_ip:
            clientthread = threading.Thread(None, connect_to_client, args=(private_ip,))
            clientthread.start()
            print("Client thread is starting")

        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 224) 
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
        vid.set(cv2.CAP_PROP_FPS, 36) 

        if (clientthread is not None 
            and clientthread.is_alive() == False):
            clientthread.join()
        
        handle_client(vid, clientthread)
    except Exception as e: 
        print(f"Error: {e}") 
    finally: 
        if sock is not None: 
            sock.close()
    


def handle_client(vid, clientthread): 
    global sock 
    global conn 
    global addr

    while vid.isOpened(): 
        suc, img = vid.read() 
        if suc == False: 
            break 
        
        img = run_model(img)

        # if a connection exists 
        if conn: 
            a = pickle.dumps(img)
            message = struct.pack("Q", len(a)) + a
            try: 
                conn.sendall(message)
            except socket.error: 
                conn = None 
                clientthread = None 
                sock = None 
                addr = None 
                continue
            key = cv2.waitKey(13)
            if key == 13: 
                conn.close() 
                clientthread = None
                sock = None
                conn = None 
                addr = None
            
        # if connection exists then join to the main thread
        if (addr is not None 
            and clientthread is not None 
            and clientthread.is_alive() == False): 
            clientthread.join() 
    
        # if no thread exists then start checking if private IP exists 
        if (conn == None 
            and clientthread is None): 
            private_ip = get_private_ip()
            if not private_ip: 
                continue
            clientthread = threading.Thread(None, connect_to_client, args=(private_ip,))
            clientthread.start() 
            if sock is None: 
                continue


if __name__ == '__main__':
    server_camera()
    # build_model()