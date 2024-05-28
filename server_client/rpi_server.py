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

def init_model(): 

    global model 
    global device

    model = get_model_instance_segmentation(2) 

    model.load_state_dict(torch.load("cifar10model.pth"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes): 
   model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT") 

   in_features = model.roi_heads.box_predictor.cls_score.in_features

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

def run_model(frame): 
    ''' 
        Evaluate RCNN Model with current image selected
    ''' 
    global model 
    global isModelEval
    global device

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0 , 1)
    frame = frame.float() / 255.0

    eval_transform = get_transform(train=False)

    if isModelEval == False: 
        model.eval()

    with torch.no_grad():
        x = eval_transform(frame)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = frame

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"Road: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image_1 = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

    output_image = output_image.numpy()

    cv2_image = np.transpose(output_image, (1, 2, 0)) 
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