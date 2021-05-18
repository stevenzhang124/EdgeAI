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

max_age=5
min_hits=1

reid_mode = cam_reid.reid_model()

# encode origin image
compare = cam_reid.Compare(model=reid_mode, origin_img="./image/origin")
origin_f, origin_name = compare.encode_origin_image()

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

#def server_inference(detection_results, persons):
def server_inference(detection_results, frame):
	global tracker_list
	global max_age
	global min_hits
	global track_id_list

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
	if len(unmatched_dets)>0:
		for idx in unmatched_dets:
			z = detection_results[idx]
			x1 = int(z[0][0])
			y1 = int(z[0][1])
			x2 = int(z[1][0])
			y2 = int(z[1][1])
			person = frame[y1:y2, x1:x2, :]
			identify_name, score = compare.run(person, origin_f, origin_name)
			if(identify_name in [ "MJ1", "MJ2", "MJ3", "MJ4", "MJ5"]):
				identify_name = "Person_1"
			elif(identify_name in ["QY1", "QY2", "QY3", "QY4", "QY5"]):
				identify_name = "Person_2"
			print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
			
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
	cv2.imshow("remote frame", frame)
	cv2.waitKey(1)


def on_new_client(conn):
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

		
		frame = server_inference(*next_task_args)
		show_frame(frame)


def main():
	#init models
	frame = cv2.imread('example.jpg')
	detection_results = [[(267, 62), (343, 270)], [(201, 65), (255, 227)], [(187, 64), (228, 169)], [(101, 73), (144, 202)]]
	frame = server_inference(detection_results, frame)
	print("init finished")

	#create socket
	HOST = 'localhost'
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
