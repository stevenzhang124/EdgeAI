#Re-ID in this part
from utils import box_iou2, assign_detections_to_trackers, draw_box_label
import cv2
import struct
import pickle
import threading
from collections import deque
from tracker import Tracker
import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from reid import cam_reid
import socket
import time


reid_mode = cam_reid.reid_model()

# encode origin image
compare = cam_reid.Compare(model=reid_mode, origin_img="./image/origin")
origin_f, origin_name = compare.encode_origin_image()


#def server_inference(detection_results, persons):
def server_inference(persons):
	print("get persons", len(persons))
	identify_names = {}
	for key, person in persons.items():
		identify_name, score = compare.run(person, origin_f, origin_name)
		if(identify_name in [ "MJ1", "MJ2", "MJ3", "MJ4", "MJ5"]):
			identify_name = "Person_1"
		elif(identify_name in ["QY1", "QY2", "QY3", "QY4", "QY5"]):
			identify_name = "Person_2"
		else:
			identify_name = "Unknown"
		print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
		identify_names[key] = identify_name
		
	return identify_names


def on_new_client(conn):
	t1 = time.time()

	data = b''
	payload_size = struct.calcsize("L")

	while True:

		# Reset args list every loop
		next_task_args_list = []

		# Retrieve number of args for next task
		while len(data) < payload_size:
			data += conn.recv(4096)

		packed_num_next_task_args = data[:payload_size]
		data = data[payload_size:]
		num_next_task_args = struct.unpack("L", packed_num_next_task_args)[0]

		# Retrieve the next task index
		while len(data) < payload_size:
			data += conn.recv(4096)

		packed_next_task_num = data[:payload_size]
		data = data[payload_size:]
		next_task_num = struct.unpack("L", packed_next_task_num)[0]

		# Retrieve all args per task
		for i in range(num_next_task_args):
			# Retrieve each argument size
			while len(data) < payload_size:
			    data += conn.recv(4096)
			packed_argsize = data[:payload_size]
			data = data[payload_size:]
			argsize = struct.unpack("L", packed_argsize)[0]

			# Retrieve data based on arg size
			while len(data) < argsize:
			    data += conn.recv(4096)

			next_arg_data = data[:argsize]
			data = data[argsize:]
			# Extract next arg
			next_arg = pickle.loads(next_arg_data)

			next_task_args_list.append(next_arg)

		# Set variables and args for running tasks
		next_task_run_index = next_task_num

		next_task_args = tuple(next_task_args_list)

		
		identify_names = server_inference(*next_task_args)

		sendData = str(identify_names)

		conn.sendall(sendData.encode())

		t2 = time.time()
		print("Re-ID consumes", (t2-t1))


def main():
	#init models
	#frame = cv2.imread('example.jpg')
	#detection_results = [[(267, 62), (343, 270)], [(201, 65), (255, 227)], [(187, 64), (228, 169)], [(101, 73), (144, 202)]]
	#frame = server_inference(detection_results, frame)
	#print("init finished")

	#create socket
	HOST = '192.168.1.111'
	PORT = 8089
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print('Socket created')

	s.bind((HOST, PORT))
	print('Socket bind complete')
	s.listen(10)
	print('Socket now listening on port', PORT)

	while True:
		print('Waiting for client to connect')

		# Receive connection from client
		client_socket, (client_ip, client_port) = s.accept()
		print('Received connection from:', client_ip, client_port)

		# Start a new thread for the client. Use daemon threads to make exiting the server easier
		# Set a unique name to display all images
		t = threading.Thread(target=on_new_client, args=[client_socket], daemon=True)
		t.setName(str(client_ip) + ':' + str(client_port))
		t.start()
		print('Started thread with name:', t.getName())

if __name__ == '__main__':
	main()
