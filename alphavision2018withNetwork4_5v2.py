#!/usr/bin/env python3
import numpy as np
import os
import sys
from operator import itemgetter
import subprocess
##command = ['sudo rfkill block wifi']#,'sudo rfkill block bluetooth','sudo ./jetson_clocks.sh']

# issue commands to kill wifi and
#bluetooth and to clock the processor
try:

    os.system('sudo rfkill block wifi')

    print(c)
    time.sleep(.1)

    os.system('sudo rfkill block bluetooth')
    
except:
    #print ('Didnt stop wifi')
    pass
    
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("/home/nvidia/Desktop/models/research")
sys.path.append("/home/nvidia/Desktop/models/research/object_detection")
sys.path.append("/home/nvidia/Desktop/models/research/object_detection/utils")

import tensorflow as tf
import time
import copy

from tensorflow.core.framework import graph_pb2
from utils import label_map_util
from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
#from PIL import Image
import cv2

import socket
from networktables import NetworkTables as nt
import smbus2 as smbus
from operator import itemgetter
#from threading import Thread

################# Get info for taking pictures and if no input then time out:

answer, answer2, answer3, answer4 = None,None,None,None

answer = input('Would you like to save images (y or n)? [default = n]: ')
answer = answer or 'n'
answer4 = input('Would you like me to show images? (y/n) [default = n]: ')
answer4 = answer4 or 'n'
if answer.strip() == ('y') or answer4.strip() == 'y':
    print('Ok take pictures set to: ',answer)
    answer2 = input('I will save pictures every ?? frames [default 40]: ')
    answer2 = answer2 or '40'

#print(answer,answer2)

#tom = 0
#setup network table connection to robot
def net():
    #tr
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0)
    print('Network Tables is setup on Nvidia TX2')

    while 1:
        
        try:
           ip = socket.gethostbyname('roboRIO-329-FRC.local')
           print('Connected to robot')
           break
        except:
            print('Waiting for NWT and Roborio connection')
            time.sleep(.5)
            pass


    return(ip)

ip = net()
        
nt.initialize(server=ip)
sd = nt.getTable("SmartDashboard")
print ('Connected to SmartDashboard')      
        
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

print('starting graph!')
input_graph = tf.Graph()
with tf.Session(graph=input_graph):
    score = tf.placeholder(tf.float32, shape=(None, 1917, 1), name="Postprocessor/convert_scores")
    expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
    for node in input_graph.as_graph_def().node:
        if node.name == "Postprocessor/convert_scores":
            score_def = node
        if node.name == "Postprocessor/ExpandDims_1":
            expand_def = node


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/home/nvidia/Desktop/models/research/object_detection/powercube_inference_graph/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

    edges = {}
    name_to_node_map = {}
    node_seq = {}
    seq = 0
    for node in od_graph_def.node:
      n = _node_name(node.name)
      name_to_node_map[n] = node
      edges[n] = [_node_name(x) for x in node.input]
      node_seq[n] = seq
      seq += 1

    for d in dest_nodes:
      assert d in name_to_node_map, "%s is not in graph" % d

    nodes_to_keep = set()
    next_to_visit = dest_nodes[:]
    while next_to_visit:
      n = next_to_visit[0]
      del next_to_visit[0]
      if n in nodes_to_keep:
        continue
      nodes_to_keep.add(n)
      next_to_visit += edges[n]

    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

    nodes_to_remove = set()
    for n in node_seq:
      if n in nodes_to_keep_list: continue
      nodes_to_remove.add(n)
    nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

    keep = graph_pb2.GraphDef()
    for n in nodes_to_keep_list:
      keep.node.extend([copy.deepcopy(name_to_node_map[n])])

    remove = graph_pb2.GraphDef()
    remove.node.extend([score_def])
    remove.node.extend([expand_def])
    for n in nodes_to_remove_list:
      remove.node.extend([copy.deepcopy(name_to_node_map[n])])

    with tf.device('/gpu:0'):
      tf.import_graph_def(keep, name='')
    with tf.device('/cpu:0'):
      tf.import_graph_def(remove, name='')


NUM_CLASSES = 1
label_map = label_map_util.load_labelmap('data/object-detection.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


##def load_image_into_numpy_array(image):
##  (im_width, im_height) = image.size
##  return np.array(image.getdata()).reshape(
##      (im_height, im_width, 3)).astype(np.uint8)
cap = cv2.VideoCapture(2)
##########print('Trying to open stream')
##########time.sleep(2)
##########swap = 1
##########counttt = 0
##########
##########ret,image_np = cap.read()
##########
##########
##########while not ret:
##########      cap = cv2.VideoCapture(swap)
##########      time.sleep(2)
##########      counttt +=1
##########      print('looking for Camera!',swap)
##########      if counttt%3 == 0:
##########          swap = 1 - swap
##########          #print('looking for Camera!',swap)
##########      ret,image_np = cap.read()
##########      time.sleep(.2)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
    expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
    score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
    expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i = 0
    print('starting cube detection!')
    tcount, start, finish, dcaptot, dstarttot, dnettot, ale, cube, cubes = 0, 0, 0, 0, 0, 0, 0, [] ,[]
    sendCount = 0
    counttt = 0
    swap = 1
    while True:
          sendCount +=1
          sd.putNumber("Frame Counter", sendCount)
          start = time.time()
          tcapStart=time.time()
          ret,image_np = cap.read()
          
          
          while not ret:
              #counttt = 0

####              if not cap.isOpened():
####                print("Camera Stream not connected ",cap.isOpened())
####                while not cap.isOpened():
####                    cap = cv2.VideoCapture(int(swap))
####                    print('looking for Camera!',swap,cap.isOpened())
####                    counttt +=1
####                    if counttt%3 == 0:
####                        swap = int(1 - swap)
####                    time.sleep(3)
####              ret,image_np = cap.read()


              
              cap = cv2.VideoCapture(swap)
              time.sleep(2)
              counttt +=1
              print('looking for Camera!',swap)
              if counttt%3 == 0:
                  swap = 1 - swap
                  
              ret,image_np = cap.read()
              
              
          
          tcapFin = time.time()
          image_np_expanded = np.expand_dims(image_np, axis=0)
        
          tnetStart = time.time()
          (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
          (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={score_in:score, expand_in: expand})
          #print 'Iteration %d: %.3f sec' %(i, time.time()-start_time)
          #
          #if scores[0][0] != None:
          #print(type(scores))
          scored = scores[0].tolist()
          #print(scored)       
  
          tnetFin = time.time()

          #################################################
          ######          score = scores[0].tolist()
          box = boxes[0].tolist()
          cnt = 0
          b = [] #ymin, ymax, xmin, xmax
          b1 = []
          for b1 in box:
        
            b1.extend([scored[cnt]])
       
            cnt += 1
          cnt = 0

##          time.sleep(1)

          b = sorted(box,key = lambda x: x[4], reverse = True)
          #print (type(b), type(b[0][4]),b[0][4])
          if sendCount > 50000:
              sendCount = 0
          
          if b[0][4] < 0.5:
              try:
                  sd.putNumber("Angle offset", 180)
                  cube_detect = False
              except:
                  pass
          
          #print(b[0:2])
          #print('New')
          
          
           #b5 = max(b[0:5],key=itemgetter(2))
          
          b6old = [0,0,0,0,0]
          for b6 in b[0:5]:
            if b6[2] > b6old[2] and b6[4] >.5:
              b6old = b6
              b5 = b6
              
            
          for r in range(5):
            
            
            
            #print('b 0:5 is: ',b[0:5])
            #print('B5 is: ',b5)
            
            if b[r][4] >.5:
             
              #print('All',b5,'===',b[r],r)
              
                     
              if b5 != b[r]:
                #print('nope',b[r],r)
                zz=0
                #r+=1
              else:
               # print('Happy',b5,b[r],r)
               # print(b5[0],b[r][0],r)
               #b5[1] min and max x
                ans = round((b5[1] + b5[3])/2,2)
                ansOut = round(ans*62-31,2)#62 is max field of view and 31 is half that.
                #print('a',b5[0],'b',b5[1],'c',b5[2],'d',b5[3])
                #print('ansOut: ',ansOut)
                try:
                    sd.putNumber("Angle offset", ansOut)
                    cube_detect = True
                except:
                    pass
                shutdown = 0.0
                try:
                    shutdown = float(sd.getNumber('shutdown',0.0))
                    #print(shutdown)
                    if shutdown == 329.0:
                        #os.system("sudo tmux kill-session")
                        #time.sleep(1)
                        os.system("sudo shutdown now -P")
                except:
                    pass
                #time.sleep(.5)
                break
            r+=1
            #print(r)
          r=0
            
              
##
          time3 = time.time()
          try:
              if answer == 'y' or answer4 == 'y':
                if sendCount%int(answer2) ==0:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
                  if answer == 'y':
                    cv2.imwrite('/home/nvidia/Desktop/gameimages/img' + str(sendCount) + '.png',image_np)
                    print('/home/nvidia/Desktop/gameimages/img' + str(sendCount) + '.png')
                  if answer4 == 'y':
                    cv2.imshow('object_detection',image_np) #, cv2.resize(image_np, (800,600))
      
              
             
          except:
              pass
             
          finish = time.time()
          dcap = tcapFin - tcapStart
          dcaptot = dcaptot + dcap
          dstart = finish - start
          dstarttot = dstarttot + dstart
          dnet = tnetFin - tnetStart
          dnettot = dnettot + dnet
          tcount +=1
          
          

          if tcount == 100:
##              testShut = sd.isConnected()
##              print(testShut)
##              if testShut == False:
##                  print('Lost Robot Connection!')
##                  ip = net()          
##                  nt.initialize(server=ip)
##                  sd = nt.getTable("SmartDashboard")
              tcount = 0
              try:
                  print ('FPS: ',round(1/(dstarttot/100),2) , '  Capture: ' , dcaptot/100 , '  NeuralNet: ', dnettot/100)
                  print(ansOut)
                  sd.putNumber("FramesPerSecond", round(1/(dstarttot/100),2))
              except:
                  pass
              dcaptot, dstarttot, dnettot = 0, 0, 0
          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.distroyAllWindows()
            break
