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

path = './New_York/Ped1/*.jpg'
res_fold = './results/'

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}

def box_area(box):
    return (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])

conf = .4
p_idx = 1
rect1 = []
img_ct = 0
for img in natsorted(glob.iglob(path)):  
  start = time.time()
  rect2 = []  
  #print("image counter: ", img_ct)
  img_ct += 1
  image2 = Image.open(img)
  image2 = np.asarray(image2)
  image2 = cv2.resize(image2, (640, 348))

  Image2 = image2.copy()
  #print(np.shape(Image2))
  
  results = model.predict(image2)
  boxes = results[0].boxes.data
  black = np.zeros_like(image2, dtype=np.uint8)
  
  for i, box in enumerate(boxes):
    #print(box[-2])
    if box[-2] > conf:
      p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(black, p1, p2, (255,255,255), thickness=-1)
      #print(int(box[5]))
      dict = {
          'box': (p1, p2),
          'matched': False,
          'name': i,
          'color': [],
          'label': int(box[5]),
      }
      rect2.append(dict)
  
  
  rect2 = sorted(rect2, key=lambda x: box_area(x['box']), reverse=True) 
 

  maskedIm = Image2 & black
  gray = cv2.cvtColor(maskedIm, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3,3), 0.5)
  sift = cv2.SIFT_create()
  key2, desc2 = sift.detectAndCompute(gray,None)
  
  #print("rect1 In the Benigging")

  # for bz in rect1:
  #   print(bz['color'], bz['name'])

  if not rect1:
    #print("len rect1: ", len(rect1))
    #print("len rect2: ", len(rect2))
    rect1 = rect2[:]
    rect2 = []
    #print("len key2: ", len(key2))
    key1 = key2
    key2 = ()
    desc1 = desc2
    desc2 = ()
    image = image2
    image2 = []
    Image1 = Image2
    Image2 = []
    #print("NO PREVE MATCHES")
    #input("Press Enter to continue to the next iteration...")
    continue
    
  else:
    # print(" ")  
    # print("BOTH IMAGES AVAILABLE")
    # print(" ")
    # print("len rect 1", len(rect1))
    # print("LENGTH KEYS 2", len(key2))
    # print("LENGTH KEYS 1", len(key1))
    matches = bf.match(desc1, desc2)
    #print(matches)
    matches = sorted(matches, key = lambda x:x.distance)
    # rint("LENGTH matches", len(matches))
    #img3 = cv2.drawMatches(image, key1, image2, key2, matches[:], image, flags=2)
    #plt.imshow(img3), plt.show()
    
    #print("")
    #print("rect1 after else")
    #for bz in rect1:
    #  print(bz['color'], bz['name'])
    
    rectA = []
    rectB = []
    xx = 0
    boxes_counter = 0
    pairs = []
    #print("New pairs", pairs)
    # print(" ")
    # print("len rect 1", len(rect1))
    # print("len rect 2", len((rect2)))
    # print(len("len key1", key1))
    # print(len("len key2", key2))
    #print(" ")
    if len(rect1) <= len(rect2):
        rectA = rect1
        rectB = rect2
    elif len(rect1) >= len(rect2):
        rectA = rect2
        rectB = rect1
    #print(len(matches))
    
    for r, re12 in enumerate(rectA):  
      #print(re12['matched'])
      b2_counter = np.zeros((len(rectB),1))
      for i, re34 in enumerate(rectB):
        #print(re34['matched'])
        if re34['label'] != re12['label']:
          continue
        if re34['matched'] == True: 
          #print("LREADY MATCHED")
          continue
        matches_counter = 0
        for mat in matches:
          a = mat.queryIdx
          b = mat.trainIdx
          (x1, y1) = map(int, key1[a].pt)
          (x2, y2) = map(int, key2[b].pt)
          if x1 > re12['box'][0][0] and x1 < re12['box'][1][0] and y1 > re12['box'][0][1] and y1 < re12['box'][1][1] and x2 > re34['box'][0][0] and x2 < re34['box'][1][0] and y2 > re34['box'][0][1] and y2 < re34['box'][1][1]:
              ct = 0
              matches_counter += 1 
        b2_counter[i] = matches_counter
        
      mxx = np.max(b2_counter)
      mxx_index = np.argmax(b2_counter)
      #print("b2 counter", b2_counter)
      #print("max index", mxx_index)
      if len(rect1) <= len(rect2):
        #print("name new rect bef", rectB[mxx_index]['name'])
        #print("name old rect bef", re12['name'])
        #print("--------")
        rect2[mxx_index]['name'] = re12['name']
        rectB[mxx_index]['name'] = re12['name']
        print("RECTB IS NEW IMAGE")
        print("rect1, rect2", re12['name'], rect2[mxx_index]['name'])
        
        pairs.append([re12, rectB[mxx_index]])
        re12['matched'] = True
        rectB[mxx_index]['matched'] = True
        
      elif len(rect1) > len(rect2):
        
        re12['name']  = rectB[mxx_index]['name']
        print("RECTB IS THE OLD IMAGE")
        print("name1, name2: ",rectB[mxx_index]['name'], re12['name'])
        pairs.append([rectB[mxx_index], re12])
        re12['matched'] = True
        rectB[mxx_index]['matched'] = True
        
        #print("CIIIII", pairs)
        #print(re12, rectB[mxx_index])
          
      #input("Press Enter to continue to the next iteration...")

      # print(pairs[0][0])
      # print(pairs[0][0][0])
      # print(pairs[0][0][1])
      # print(pairs[0][1][0])
      # print(pairs[0][1][1])
    #print(pairs)  
    #input("Press Enter to continue to the next iteration...")
   
    for prs in pairs: 
      name1 = prs[0]['name']
      name2 = prs[1]['name']
      if name1 != name2:
        print("MISMATCH")
        print("name1 and name2: ", name1, name2)
        print("IMAGE NUMBER:", p_idx)
        print("--------")
        for pzz in pairs:
          zz1 = pzz[0]['name']
          zz2 = pzz[1]['name']
          print(zz1,zz2)
        input("Press Enter to continue to the next iteration...")

    
    # print(" ")
    # print("rect1 before color matching")
    # for bz in rect1:
    #   print(bz['color'], bz['name'])
    #   print(" ")
    # print("rect2 before color matching")
    # for bz in rect2:
    #   print(bz['color'], bz['name'])
    
    for mtc in pairs:
     
      R = 0
      G = 255
      B = 255
      cv2.rectangle(image, mtc[0]['box'][0], mtc[0]['box'][1], (R, G, B), thickness= 2)
      cv2.rectangle(image2, mtc[1]['box'][0], mtc[1]['box'][1], (R, G, B), thickness= 2)
      
                   
      box1 = mtc[0]['box'][0]
      box2 = mtc[1]['box'][0]
      # st1 = (mtc[0]['box'][0][1], mtc[0]['box'][1][0] )
      #print("st1", st1)
      #print("box1", box1)
      #print(box1[0])
      # Add text above the boxes
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      font_thickness = 2
      #print(type(box1[0]))



      cv2.putText(image, str(mtc[0]['name']), box1, font, font_scale, (255, 255, 255), font_thickness)
      cv2.putText(image2, str(mtc[1]['name']), box2, font, font_scale, (255, 255, 255), font_thickness)
      
    # Create a figure and a subplot grid with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[1].imshow(image2)
    plt.tight_layout()
    #plt.show() # use this to show pairs 

    pair_filename = f"pair_{p_idx}.png"  # Change this filename format as needed
    plt.savefig(os.path.join(res_fold, pair_filename))
    plt.close()
    p_idx +=1
    
    
    for rt in rect2:
      #print(rt['name'])
      rt['matched'] = False
    
    # print("")
    # print("rect 2 colors after coloring")
    # for bz in rect2:
    #   print(bz['color'], bz['name'])
      
    
    rect1 = rect2[:]
    key1 = key2
    desc1 = desc2
    
    # print(" ")
    # print("rect 1 after renaming")
    # for bz in rect1:
    #   print(bz['color'], bz['name'])
    
    # print("negative birinni")
    # plt.imshow(image2)
    # plt.show()
    # print("birinni")
    image = Image2.copy()
    elapsed_time = time.time() - start
    print("elapsed time: ", elapsed_time)
    
    # plt.imshow(image)
    # plt.show()
    #print(rect1)
  #print("COMPARISON OVER, NEXT IMAGE")
  #input("Press Enter to continue to the next iteration...")
  #time.sleep(1)
    