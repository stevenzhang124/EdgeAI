import sys
import subprocess
import cv2
import pickle
import socket
import struct
from pedestrian_detection_ssdlite import api
import time


S_IP = '192.168.1.111'
S_port = 8089

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

def get_frame(cap):

	
	ret, frame = cap.read()
	# counter=0

	# while (cap.isOpened()):
	# 	#t1 = cv2.getTickCount()
	# 	counter+=1
	# 	#if counter % 12 !=0:
	# 	#	print(counter)
	# 	#	continue
	# 	t1 = time.time()
	# 	print ("before read:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	# 	if counter % 5 != 0:
	# 		ret, frame = cap.read()
	# 		continue

	# 	break	
	
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
	img = cv2.imread('example.jpg')
	detection_test = api.get_person_bbox(img, thr=0.5)
	print(detection_test)

	#create socket
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect((S_IP, S_port))
	print("Successfully connected to the server.")

	cap = read_video()
	while True:
		frame = get_frame(cap)
		print("get one frame")

		start = time.time()
		data = detect_person(frame)
		end = time.time()
		print("detection consumes {0:.2f}".format(end-start))

		start = time.time()
		offload_to_peer(2, next_task_args=data, client_socket=client_socket)
		end = time.time()
		print("send consumes {0:.2f}".format(end-start))
