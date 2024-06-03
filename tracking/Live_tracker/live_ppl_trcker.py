from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import os
from matplotlib import pyplot as plt
import glob
from natsort import natsorted
import time

model = YOLO("yolov8n.pt")


sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}

def box_area(box):
    return (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])

conf = .5
p_idx = 1
rect1 = []
img_ct = 0
max_name = 0



# Open the video capture
cap = cv2.VideoCapture(6)  # Use 0 for the default camera

# Set the desired frame size
frame_width = 640
frame_height = 480


# Set the frame size for camera 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Unable to open the camera")
    exit()

while True:
    
    # Read a frame from the camera
    ret, frame2 = cap.read()
    
     # Check if the frame was successfully read
    if not ret:
        print("Failed to capture frame")
        break
    image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    rect2 = []  
    img_ct += 1
    #image2 = Image.open(image2)
    image2 = np.asarray(image2)
    image2 = cv2.resize(image2, (640, 348))

    Image2 = image2.copy()
    
    results = model.predict(image2, verbose=False)
    boxes = results[0].boxes.data
    black = np.zeros_like(image2, dtype=np.uint8)
 
  
    
    if rect1 == []:
      ctt = 0
      for box in boxes:
        if box[-2] > conf and int(box[5]) == 0:
          p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

          cv2.rectangle(black, p1, p2, (255,255,255), thickness=-1)
          dict = {
              'box': (p1, p2),
              'matched': False,
              'name': ctt,
              'color': [],
              'label': int(box[5]),
            }
          ctt += 1
          rect2.append(dict)
      rect2 = sorted(rect2, key=lambda x: box_area(x['box']), reverse=True) 
      max_name = ctt


    else:
        ctt = 0
        for box in boxes:
          if box[-2] > conf and int(box[5]) == 0:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            cv2.rectangle(black, p1, p2, (255,255,255), thickness=-1)
            dict = {
                'box': (p1, p2),
                'matched': False,
                'name': None,
                'color': [],
                'label': int(box[5]),
              }
            ctt += 1
            rect2.append(dict)

        rect2 = sorted(rect2, key=lambda x: box_area(x['box']), reverse=True) 
    
    if len(rect2) == 0:
      print("NO PEOPLE DETECTED")

      continue
    
    maskedIm = Image2 & black
    gray = cv2.cvtColor(maskedIm, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0.5)
    sift = cv2.SIFT_create()
    key2, desc2 = sift.detectAndCompute(gray,None)
    
    
    if not rect1:

      rect1 = rect2[:]
      rect2 = []
      key1 = key2
      key2 = ()
      desc1 = desc2
      desc2 = ()
      image = image2
      image2 = []
      Image1 = Image2
      Image2 = []

      continue

    else:

      matches = bf.match(desc1, desc2)
      matches = sorted(matches, key = lambda x:x.distance)


      xx = 0
      boxes_counter = 0
      pairs = []


      if len(rect1) <= len(rect2):
        
        for r, re1 in enumerate(rect1):  
          b2_counter = np.zeros((len(rect2),1))

          for i, re2 in enumerate(rect2):
            if re2['label'] != re1['label']:
              continue
            if re2['matched'] == True: 
              continue
            matches_counter = 0
            for mat in matches:
              a = mat.queryIdx
              b = mat.trainIdx
              (x1, y1) = map(int, key1[a].pt)
              (x2, y2) = map(int, key2[b].pt)
              if x1 > re1['box'][0][0] and x1 < re1['box'][1][0] and y1 > re1['box'][0][1] and y1 < re1['box'][1][1] and x2 > re2['box'][0][0] and x2 < re2['box'][1][0] and y2 > re2['box'][0][1] and y2 < re2['box'][1][1]:
                  ct = 0
                  matches_counter += 1 
            b2_counter[i] = matches_counter
          mxx = np.max(b2_counter)
          mxx_index = np.argmax(b2_counter)

          if mxx < 10:
            continue
          else:

            rect2[mxx_index]['name'] = re1['name']
            pairs.append([re1, rect2[mxx_index]])

            re1['matched'] = True
            rect2[mxx_index]['matched'] = True

    
      elif len(rect1) > len(rect2):
        
        for r, re2 in enumerate(rect2):  
        
          b2_counter = np.zeros((len(rect1),1))

          for i, re1 in enumerate(rect1):
            
            if re2['label'] != re1['label']:
              continue
            if re1['matched'] == True: 
              continue
            matches_counter = 0
            for mat in matches:
              a = mat.queryIdx
              b = mat.trainIdx
              (x1, y1) = map(int, key1[a].pt)
              (x2, y2) = map(int, key2[b].pt)
              if x1 > re1['box'][0][0] and x1 < re1['box'][1][0] and y1 > re1['box'][0][1] and y1 < re1['box'][1][1] and x2 > re2['box'][0][0] and x2 < re2['box'][1][0] and y2 > re2['box'][0][1] and y2 < re2['box'][1][1]:
                  ct = 0
                  matches_counter += 1 

            b2_counter[i] = matches_counter

          mxx = np.max(b2_counter)
          mxx_index = np.argmax(b2_counter)
          if mxx < 10:
            continue
          else:
            re2['name']  = rect1[mxx_index]['name']
            re2['matched'] = True
            rect1[mxx_index]['matched'] = True
            pairs.append([rect1[mxx_index], re2])

      for i in rect2:
          if i['name'] == None:
            i['name'] = max_name 
            max_name += 1



    #   for prs in pairs: 
    #     name1 = prs[0]['name']
    #     name2 = prs[1]['name']
    #     if name1 != name2:
    #       print("MISMATCH")
    #       print("name1 and name2: ", name1, name2)
    #       print("IMAGE NUMBER:", p_idx)
    #       print("--------")
    #       for pzz in pairs:
    #         zz1 = pzz[0]['name']
    #         zz2 = pzz[1]['name']


    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    for mtc in pairs:
      cv2.rectangle(image2, mtc[1]['box'][0], mtc[1]['box'][1], (255, 0, 0), thickness= 2)
      box2 = mtc[1]['box'][0]
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      font_thickness = 2
      cv2.putText(image2, str(mtc[1]['name']), box2, font, font_scale, (255, 255, 255), font_thickness)
    cv2.imshow("image", image2)


    for rt in rect2:
       rt['matched'] = False


    rect1 = rect2
    rect1 = sorted(rect1, key=lambda x: box_area(x['box']), reverse=True) 


    key1 = key2
    desc1 = desc2
    image = Image2.copy()

    # Check if 'q' key is pressed to quit the program
    if cv2.waitKey(1) == ord('q'):
        break
    
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()


