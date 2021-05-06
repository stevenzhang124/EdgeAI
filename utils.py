import numpy as np
import cv2
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

def box_iou2(a, b):
	'''
	Helper funciton to calculate the ratio between intersection and the union of
	two boxes a and b
	a[0], a[1], a[2], a[3] <-> left, up, right, bottom
	'''
	
	w_intsec = np.maximum (0, (np.minimum(a[1][0], b[1][0]) - np.maximum(a[0][0], b[0][0])))
	h_intsec = np.maximum (0, (np.minimum(a[1][1], b[1][1]) - np.maximum(a[0][1], b[0][1])))
	s_intsec = w_intsec * h_intsec
	s_a = (a[1][0] - a[0][0])*(a[1][1] - a[0][1])
	s_b = (b[1][0] - b[0][0])*(b[1][1] - b[0][1])
  
	return float(s_intsec)/(s_a + s_b -s_intsec)

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
	'''
	From current list of trackers and new detections, output matched detections,
	unmatchted trackers, unmatched detections.
	'''    
	
	IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
	for t,trk in enumerate(trackers):
		#trk = convert_to_cv2bbox(trk) 
		for d,det in enumerate(detections):
		 #   det = convert_to_cv2bbox(det)
			IOU_mat[t,d] = box_iou2(trk,det) 
	
	# Produces matches       
	# Solve the maximizing the sum of IOU assignment problem using the
	# Hungarian algorithm (also known as Munkres algorithm)
	

	matched_idx_tra, matched_idx_det = linear_assignment(-IOU_mat)        
	matched_idx = np.zeros((len(matched_idx_tra),2),dtype=np.int8)
	for i in range(len(matched_idx_tra)):
		matched_idx[i]=(matched_idx_tra[i],matched_idx_det[i])
	
	#matched_idx = linear_assignment(-IOU_mat)        

	unmatched_trackers, unmatched_detections = [], []
	for t,trk in enumerate(trackers):
		if(t not in matched_idx[:,0]):
			unmatched_trackers.append(t)

	for d, det in enumerate(detections):
		if(d not in matched_idx[:,1]):
			unmatched_detections.append(d)

	matches = []
   
	# For creating trackers we consider any detection with an 
	# overlap less than iou_thrd to signifiy the existence of 
	# an untracked object
	
	for m in matched_idx:
		if(IOU_mat[m[0],m[1]]<iou_thrd):
			unmatched_trackers.append(m[0])
			unmatched_detections.append(m[1])
		else:
			matches.append(m.reshape(1,2))
	
	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)
	
	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), personReID_info={'personID':'Unknown'}, show_label=True):
	'''
	Helper funciton for drawing the bounding boxes and the labels
	bbox_cv2 = [left, top, right, bottom]
	'''
	#box_color= (0, 255, 255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_size = 0.4
	font_color = (0, 0, 0)
	left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
	
	# Draw the bounding box
	cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
	
	if show_label:
		# Draw a filled box on top of the bounding box (as the background for the labels)
		cv2.rectangle(img, (left-2, top-30), (right+2, top), box_color, -1, 1)
		
		# Output the labels that show the x and y coordinates of the bounding box center.
		text_ID = 'personID:' + personReID_info['personID']
		cv2.putText(img,text_ID,(left,top-20), font, font_size, font_color, 1, cv2.LINE_AA)
		text_x= 'x='+str((left+right)/2)
		cv2.putText(img,text_x,(left,top-10), font, font_size, font_color, 1, cv2.LINE_AA)
		text_y= 'y='+str((top+bottom)/2)
		cv2.putText(img,text_y,(left,top), font, font_size, font_color, 1, cv2.LINE_AA)
			
	return img   