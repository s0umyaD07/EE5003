
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations


def Euclidean_d(x, y):

    d = math.sqrt(x**2 + y**2)
    
    return d


def convert(x, y, w, h): 

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax




def cv2DrawBox(detections, img ):
 
    pass_counter = 0 ;
    if len(detections) > 0:  						
        centroid_d = dict() 						
        objectId = 0	
        x_ball= 0
        y_ball = 0
       
        for label, confidence, bbox in detections:				
            
            name_tag = label  
            
            
            if name_tag == 'sports ball':                
                x, y, w, h = (bbox[0],
             bbox[1],
             bbox[2],
             bbox[3])
                
                
                x_ball=int(x)
                y_ball=int(y)
            
            if name_tag == 'person':                
                x, y, w, h = (bbox[0],
             bbox[1],
             bbox[2],
             bbox[3])
                         
              
                xmin, ymin, xmax, ymax = convert(float(x), float(y), float(w), float(h))   
                centroid_d[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) 
                objectId += 1            	
        Ball_possesion_List =[]
        
        Ball_possesion = "No" 
        
        for idx,px in centroid_d.items(): 
            dx, dy = px[0] - x_ball, px[1] - y_ball  	
            distance = Euclidean_d(dx, dy) 			
            if distance <50.0:		
                if idx not in Ball_possesion_List:
                    Ball_possesion_List.append(idx)       
                   
                    Ball_possesion ="Yes"
                    pass_counter += 1 
                   
                   
        for idx, box in centroid_d.items():  
            if idx in Ball_possesion_List:   
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) 
                
                
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) 
                
	      
        text = "Ball is in possession: %s " % str(Ball_possesion)  	
  
        location = (10,25)												
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  
       

       

    return img , pass_counter
     

netMain = None
metaMain = None
altNames = None
network = None
class_names =None
class_colors =None

def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames ,network, class_names, class_colors
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    
    network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    
    cap = cv2.VideoCapture("project.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
  

    out = cv2.VideoWriter("./Demo/test5_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(new_width, new_height))
    
   
        

    
 
    darknet_image = darknet.make_image(new_width, new_height, 3)
    pass_main_counter = 0 
    while True:
        
        prev_time = time.time()
        ret, frame_read = cap.read()
       
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,(new_width, new_height),interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
        
        image , pc = cv2DrawBox(detections, frame_resized)
        
        pass_main_counter += pc ;
        
        
         
        res_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = 100
        
        ret, thresh_img = cv2.threshold(res_gray, thresh, 255, cv2.THRESH_BINARY)
        # find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        prev = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Detect players
            if (h >= (1.5) * w):
                if (w > 15 and h >= 15):
                    # green range
                    lower_green = np.array([40, 40, 40])
                    upper_green = np.array([70, 255, 255])
                    # yellow range
                    lower_yellow = np.array([25, 0,0])
                    upper_yellow = np.array([35, 255, 255])
                    
                    player_img = image[y:y + h, x:x + w]
                    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                    # If player has yellow jersy
                    mask1 = cv2.inRange(player_hsv, lower_yellow, upper_yellow)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)
                    # If player has green jersy
                    mask2 = cv2.inRange(player_hsv, lower_green, upper_green)
                    res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                    nzCountgreen = cv2.countNonZero(res2)

                    if (nzCount >= 15):
                        # Mark yellow players as yellow
                       cv2.putText(image, 'yellow', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        
                    else:
                       pass
                    if (nzCountgreen >= 15):
                        # Mark green players as green
                       cv2.putText(image, 'green', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        
                    else:
                       pass
            

      
   
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
