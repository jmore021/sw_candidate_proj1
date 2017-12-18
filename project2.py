import cv2
import numpy as np
import math

def main():
	cap = cv2.VideoCapture('sample_weld_video.mp4')
	
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
	prev_frame_data = [0, 0, 0, 0] # previous frames details [area, x, y, is_shrinking]
	
	while cap.isOpened():
		flag, frame = cap.read()
		if flag:
			# The frame is ready and already captured
		    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
		    print str(pos_frame)+" frames"		    
		    img, prev_frame_data = state(frame, prev_frame_data)
		    # cv2.imwrite("frame%d.jpg" % pos_frame, img)
		    cv2.imshow('video', img)
		else:
		    # The next frame is not ready, so we try to read it ag
		    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
		    print "frame is not ready"
		    cv2.waitKey(1000)

		if cv2.waitKey(10) == 27:
			break
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			# If the number of captured frames is equal to the total number of frames,
			# we stop
			break

	# When everything done, release the video capture object
	cap.release()
	# Closes all the frames
	cv2.destroyAllWindows()
	return


def state(curr_frame, prev_data):
	if is_welder_off(curr_frame, prev_data):
		img = add_lines(curr_frame, prev_data[1], prev_data[2])
		img = cv2.putText(img, 'Welder Off!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA);
		cur_frame_data = prev_data

	else: 
		contours = detect_weld_pool(curr_frame)
	
		if contours == []: # Not detected
			img, cur_frame_data = detect_motion(curr_frame, prev_data)
		else:
			img, cur_frame_data = detect_new_frame(curr_frame, contours, prev_data);

	return img, cur_frame_data

def is_welder_off(image, prev_data):
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
	height, width,channels = image.shape

	if sum(hist[0:60]) >= (height * width) /2: # Most of the pixels are dark
		return False
	return True

def detect_weld_pool(frame):   
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_img = cv2.GaussianBlur(gray_img, (5,5),0)

	ret, bw_img = cv2.threshold(gray_img,178,255,cv2.THRESH_BINARY)
	
	#Morphological Operation
	kernel = np.ones((5,5),np.uint8)
	bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
	bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
	bw_img = cv2.dilate(bw_img,kernel,iterations = 1)

	im2, contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	if contours != []:
		detected = max(contours, key = cv2.contourArea)
	else :
		detected = []

	return detected

def detect_new_frame(image, contours, prev_data):
	x,y,w,h = cv2.boundingRect(contours) # welding pool size
	
	# Check if points are close enough
	if is_close(prev_data[1], prev_data[2], (x+w), (y+h)):
		new_img = add_lines(image, prev_data[1], prev_data[2])
		if is_shrinking(prev_data[0], (w*h)):
			prev_data[3] = 1
	 
	else:
		new_img = add_lines(image, (x+w), (y+h))
		prev_data[1] = (x+w);
		prev_data[2] = (y+h);
		prev_data[3] = 0;
		
	prev_data[0] = w*h; # area stored to check if shrinking
	return new_img, prev_data
		
def detect_motion(image, prev_data):
	if(prev_data[3] == 1): # Shrinking means it moved behind an object
		new_img = add_lines(image, prev_data[1], prev_data[2])
	else: 
		new_img = add_lines(image, prev_data[1], prev_data[2])
		new_img = cv2.putText(new_img, 'Frame skipped!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA);

	return new_img, prev_data; 

def add_lines(image, x, y):
	height, width,channels = image.shape # image size
	# draw the book contour (in green)	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.line(image, (x,0), (x,height), (0,255,0),4)
	cv2.line(image, (0, y), (width, y), (0,255,0),4)
	# img_bb = np.hstack([image, output])
	return image

def is_close(prev_x, prev_y, curr_x, curr_y):
	distance = math.sqrt((prev_x - curr_x)**2 + (prev_y - curr_y)**2)
	if distance < 123:
		return True
	else:
		return False

def is_shrinking(prev_area, curr_area):
	if prev_area > curr_area:
		return True
	else:
		return False


main()