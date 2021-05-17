#ReID in one server and detection and tracking in client
import sys
import subprocess
import cv2
import pickle
import socket
import struct
from pedestrian_detection_ssdlite import api
import time
from scipy.optimize import linear_sum_assignment as linear_assignment
from utils import box_iou2, assign_detections_to_trackers, draw_box_label
from collections import deque
from tracker import Tracker
import numpy as np
import threading
import os

#global variable
S_IP = '192.168.1.111'
S_port = 8089

identity_num = 0
max_age=5
min_hits=1
tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])


def open_cam_rtsp(uri, width, height, latency):
	gst_str = ('rtspsrc location={} latency={} ! '
			   'rtph264depay ! h264parse ! omxh264dec ! '
			   'nvvidconv ! '
			   'video/x-raw, width=(int){}, height=(int){}, '
			   'format=(string)BGRx ! '
			   'videoconvert ! appsink').format(uri, latency, width, height)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def read_video():

	uri = "rtsp://admin:edge1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
	cap = open_cam_rtsp(uri, 640, 480, 200)


	if not cap.isOpened():
		sys.exit('Failed to open camera!')
	return cap

vs = cv2.VideoCapture("xxxx.mp4")
files = os.listdir('set00-occ')
def read_local_video():
	for i in range(len(files)):

		path = './set00-occ/' + str (i+1) + '.jpg' 
		frame = cv2.imread(path)

	return frame


def get_frame(cap):

	counter=0
	while (cap.isOpened()):
		#t1 = cv2.getTickCount()
		counter+=1
		#if counter % 12 !=0:
		#	print(counter)
		#	continue
		
		if counter % 5 != 0:
			ret, frame = cap.read()
			continue
		break
	
	return frame

def detect_person(frame):

	detection_results = api.get_person_bbox(frame, thr=0.5)

	#persons = []
	#for bbox in detection_results:
	#	x1 = int(bbox[0][0])
	#	y1 = int(bbox[0][1])
	#	x2 = int(bbox[1][0])
	#	y2 = int(bbox[1][1])

		#person = frame[y1:y2, x1:x2, :]
		#persons.append(person)

	#return detection_results, persons
	return detection_results, frame

def tracking_person(detection_results, frame):
	global tracker_list
	global max_age
	global min_hits
	global track_id_list
	global identity_num

	x_box =[]
	if len(tracker_list) > 0:
		for trk in tracker_list:
			x_box.append([(trk.box[0],trk.box[1]),(trk.box[2],trk.box[3])]) #should be changed into the right format instead of the .box format
            
	matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, detection_results, iou_thrd = 0.2)  
	
	# Deal with matched detections     
	if matched.size >0:
		for trk_idx, det_idx in matched:
			z = detection_results[det_idx]
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk= tracker_list[trk_idx]
			tmp_trk.kalman_filter(z)
			xx = tmp_trk.x_state.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			x_box[trk_idx] = xx
			tmp_trk.box =xx
			tmp_trk.hits += 1
			tmp_trk.no_losses = 0
	
    # Deal with unmatched detections
    t1 = time.time()      
	if len(unmatched_dets)>0:
		persons = {}
		for idx in unmatched_dets:
			z = detection_results[idx]
			x1 = int(z[0][0])
			y1 = int(z[0][1])
			x2 = int(z[1][0])
			y2 = int(z[1][1])
			person = frame[y1:y2, x1:x2, :]
			#persons.append(person)
			persons[identity_num] = person				
			identify_name = "Unknown" + str(identity_num)
			identity_num += 1
			
            #generate a new tracker for the person
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk = Tracker() # Create a new tracker
			x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
			tmp_trk.x_state = x
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box = xx
			tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
			tmp_trk.personReID_info['personID'] = identify_name #assign the reidentified personID for the tracker
			tracker_list.append(tmp_trk)
			x_box.append(xx)

		#offload the Re-ID task
		arg_send = (persons,)
		offload_to_peer(2, next_task_args=persons, client_socket=client_socket)
	t2 = time.time()
	print("send consumes", (t2-t1))
	
	    # Deal with unmatched tracks       
	if len(unmatched_trks)>0:
		for trk_idx in unmatched_trks:
			tmp_trk = tracker_list[trk_idx]
			tmp_trk.no_losses += 1
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box =xx
			x_box[trk_idx] = xx
	
	# The list of tracks to be annotated and draw the figure
	good_tracker_list =[]
	for trk in tracker_list:
		if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
			good_tracker_list.append(trk)
			x_cv2 = trk.box
			trackerID_str="Unknown Person:"+str(trk.id)
			if trk.personReID_info['personID'] == "Unknown":
				frame= draw_box_label(frame, x_cv2,personReID_info={'personID':trackerID_str}) # Draw the bounding boxes for unknown person
			else:
				frame= draw_box_label(frame, x_cv2,personReID_info=trk.personReID_info) # Draw the bounding boxes for re-identified person
	#book keeping
	deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

	for trk in deleted_tracks:
		track_id_list.append(trk.id)

	tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

		
	return frame

def show_frame(frame):
	cv2.imshow("local frame", frame)
	cv2.waitKey(1)


def recv_data():
	while True:
		b = client_socket.recv(1024).decode()
		if len(b) > 0:
			identity_names = eval(b)
			assign_new_identity(identity_names)

def assign_new_identity(identity_names):

	for key, name in identity_names.items():
		identify_name = "Unknown" + str(key)
		for trk in tracker_list:
			if trk.personReID_info['personID'] == identify_name:
				trk.personReID_info['personID'] = name
	


def offload_to_peer(next_task_num, next_task_args, client_socket):
	send_data = b''
	next_arg_data = []

	if next_task_args is not None:
		if type(next_task_args) is tuple:
			for arg in next_task_args:
				next_arg_data.append(arg)
		else:
			next_arg_data.append(next_task_args)

	# Send number of args
	send_data += struct.pack("L", len(next_arg_data))

	# Send the next task's number
	send_data += struct.pack("L", next_task_num)

	if len(next_arg_data) > 0:
		for next_arg in next_arg_data:
			data = pickle.dumps(next_arg)
			arg_size = struct.pack("L", len(data))
			send_data += arg_size
			send_data += data

	print(len(send_data))
	client_socket.sendall(send_data)


if __name__ == '__main__':
	# init detection model
	img = cv2.imread('./test_img/one.jpg')
	detection_test = api.get_person_bbox(img, thr=0.5)
	print(detection_test)

	#create socket
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect((S_IP, S_port))
	print("Successfully connected to the server.")

	t = threading.Thread(target=recv_data)
	t.daemon = True
	t.start()
	print("listen thread starts")

	#cap = read_video()
	#cap = read_local_video()
	#while True:
		#t1 = time.time()

		#frame = get_frame(cap)
	file = 'edge1.jpg'
	frame = cv2.imread('./test_img/'+file)
	print("get one frame")
	# if frame is None:
	# 	break

	start = time.time()
	detection_results, frame = detect_person(frame)
	end = time.time()
	print("detection consumes {0:.2f}".format(end-start))

	start2 = time.time()
	img = tracking_person(detection_results, frame)
	end = time.time()
	print("tracking and send consumes {0:.2f}".format(end-start2))
	
	t2 = time.time()

	print("one frame takes {0:.2f}".format(t2-start))
	frame_rate_calc = 1 / (t2 - start)
	print("FPS is {0:.2f}".format(frame_rate_calc))
		#if frame_rate_calc < 15:
		#	frame_rate_calc = 2*frame_rate_calc

		# cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
		# 			cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		# #show video
		# show_frame(img)
