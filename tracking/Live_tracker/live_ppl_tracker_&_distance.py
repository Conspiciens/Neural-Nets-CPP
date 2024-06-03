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
import pyrealsense2 as rs
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from functools import partial


def box_area(box):
    return (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])

def coord_finder(p1, p2, ic_row, ic_col, c_mr):
  col1 = p1[0]
  row1 = p1[1]
  col2 = p2[0]
  row2 = p2[1]
  d_r = abs(row2-row1)
  d_c = abs(col2-col1)
  
  
  # print("dist r and c: ", d_r, d_c)
  c_r = int((row1+row2)*0.38)
  c_c = int((col1 + col2)*0.5)
  
  # print("center of the box: ", c_r, c_c)
  per_ru = int(0.05*d_r)
  per_rd = int(0.05*d_r)
  per_c = int(0.05*d_c)
  
  # print("per_c per_r: ", per_c, per_r)
  b1r = c_r - per_ru
  b1c = c_c - per_c
  
  b2r = c_r + per_rd
  b2c = c_c + per_c
  
  b1 = (b1c, b1r)
  b2 = (b2c, b2r)
  #print("b1, b2:", b1, b2)
  start = time.time()
  print(type(c_mr))
  print(c_mr.shape)
  row_slice = slice(b1r, b2r)
  col_slice = slice(b1c, b2c)
  cent_box = c_mr[row_slice, col_slice]
  print(cent_box)
  
  
  zc = np.mean(cent_box)
  
  ypx = c_r - ic_row
  
  xpx= c_c - ic_col
  
  ysmm = ypx*(F/fy)
  
  xsmm = xpx*(F/fx)
  
  yc = zc*(ysmm/F)
  
  xc = zc*(xsmm/F)
  return xc, yc, zc, b1, b2

model = YOLO("yolov8n.pt")


akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


conf = .5
p_idx = 1
rect1 = []

img_ct = 0
max_name = 0

# pixels size in mm for 640x480 
F = 1.93 # mm, focal length 
fx = 375.2639 # focal length along x axis in pixels 
fy = 375.7042 # focal lenght along y axis in pixels 

px = F/fx # mm, size of one pixel along x axis
py = F/fy # mm, siz of one pixel along y axis 

t = 1/60 # s, time between frames 60 is 60 fps

ic_row = 174
ic_col = 320

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        image2 = np.asanyarray(color_frame.get_data())
        c_mr = np.asanyarray(depth_frame.get_data())
        
        Image2 = image2.copy()
        
        results = model.predict(image2, verbose=False)
        
        boxes = results[0].boxes.data
        black = np.zeros_like(image2, dtype=np.uint8)
        
        max_name = 0
        rect2 = []
        img_ct =+ 1
        ctt = 0
        for box in boxes:
          if box[-2] > conf and int(box[5]) == 0:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            xc, yc, zc, b1, b2 = coord_finder(p1, p2, ic_row, ic_col, c_mr)

            cv2.rectangle(image2, b1, b2, (255,255,255), thickness=2)

            cv2.rectangle(black, p1, p2, (255,255,255), thickness=-1)
            dict = {
                'box': (p1, p2),
                'matched': False,
                'name': ctt,
                'cntr': 1,
                'coordinates1': 0,
                'coordinates2': (xc, yc, zc),
                'predicted': 0,
                'label': int(box[5]),
              }
            ctt += 1
            rect2.append(dict)
            
        rect2 = sorted(rect2, key=lambda x: box_area(x['box']), reverse=True) 
        max_name = ctt
            
        if len(rect2) == 0:
            print("NO PEOPLE DETECTED")

            continue
        
        maskedIm = Image2 & black
        gray = cv2.cvtColor(maskedIm, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0.5)
        key2, desc2 = akaze.detectAndCompute(gray,None)
        
        
        if desc2 is None:
            continue
    
        if rect1 == []:

          rect1 = rect2[:]
          rect2 = []
          key1 = key2
          key2 = ()
          desc1 = desc2
          desc2 = ()
          image = image2
          image2 = []
          #Image1 = Image2
          #Image2 = []

          continue
      
        else:
            
            start = time.time()
            
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

                    re1['matched'] = True
                    rect2[mxx_index]['matched'] = True
                    
                    rect2[mxx_index]['cntr'] = re1['cntr']
                    #print("COUNTER BEFORE: ", rect2[mxx_index]['cntr'])

                    if rect2[mxx_index]['cntr'] == 1:

                      rect2[mxx_index]['coordinates1'] = re1['coordinates2']

                      rect2[mxx_index]['cntr'] += 1

                      rect2[mxx_index]['predicted'] = re1['predicted']


                    elif rect2[mxx_index]['cntr'] == 5: 
                      
                      rect2[mxx_index]['coordinates1'] = re1['coordinates1']

                      rect2[mxx_index]['cntr'] = 1

                      x1, y1, z1 = rect2[mxx_index]['coordinates1']
                      x2, y2, z2 = rect2[mxx_index]['coordinates2']

                      xd = (x2 - x1)*2
                      yd = (y2 - y1)*2
                      zd = (z2 - z1)*2 

                      xp = x2 + xd
                      yp = y2 + yd
                      zp = z2 + zd
                      print('zp:', zp)
        
                      if z2 <= 1500:
                        print("too close, STOP")
                       
                      elif zp <= 1500:
                        print("Predicted Stop", zp, "z2", z2)
                        
                      else:
                        print(" ok ")

                    else:

                      rect2[mxx_index]['coordinates1'] = re1['coordinates1']
                      rect2[mxx_index]['predicted'] = re1['predicted']
                      rect2[mxx_index]['cntr'] = re1['cntr']
                      rect2[mxx_index]['cntr'] += 1

                    pairs.append([re1, rect2[mxx_index]])
                  
                  
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
                
            
            for mtc in pairs:
              cv2.rectangle(image2, mtc[1]['box'][0], mtc[1]['box'][1], (255, 0, 0), thickness= 2)
              box2 = mtc[1]['box'][0]
              font = cv2.FONT_HERSHEY_SIMPLEX
              font_scale = 0.5
              font_thickness = 2
              cv2.putText(image2, str(mtc[1]['name']), box2, font, font_scale, (255, 255, 255), font_thickness)
            
            #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            cv2.imshow("image", image2)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            for rt in rect2:
                rt['matched'] = False
    
  
            rect1 = rect2
            rect1 = sorted(rect1, key=lambda x: box_area(x['box']), reverse=True) 

            key1 = key2
            desc1 = desc2

            image = Image2.copy()
            end = time.time()
            
            tot = end - start 
            
            #print("time", tot)
                       
finally:

    # Stop streaming
    pipeline.stop()

