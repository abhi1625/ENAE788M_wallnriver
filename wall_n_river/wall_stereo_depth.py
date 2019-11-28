import cv2
import numpy as np
import os


def thres_img(cv_image):
	# thresh_b_min=56
	# thresh_g_min=128
	# thresh_r_min=137
	# thresh_b_max=255
	# thresh_g_max=255
	# thresh_r_max=255

	# river params
	thresh_r_min=87
	thresh_g_min=81
	thresh_b_min=133
	thresh_r_max=155
	thresh_g_max=205
	thresh_b_max=255


	cv_image[cv_image[:,:,0] < thresh_b_min]=0
	cv_image[cv_image[:,:,1] < thresh_g_min]=0
	cv_image[cv_image[:,:,2] < thresh_r_min]=0

	cv_image[cv_image[:,:,0] > thresh_b_max]=0
	cv_image[cv_image[:,:,1] > thresh_g_max]=0
	cv_image[cv_image[:,:,2] > thresh_r_max]=0
	return cv_image

def detect_blobs(im):
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	 
	# Change thresholds
	params.minThreshold = 200
	params.maxThreshold = 256
	params.blobColor = 255
	 
	# Filter by Area.
	params.filterByArea = True
	params.minArea = 2500
	params.maxArea = 1000000
	 
	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1
	 
	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.87
	 
	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.001
	 
	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
	    detector = cv2.SimpleBlobDetector(params)
	else : 
	    detector = cv2.SimpleBlobDetector_create(params)

	keypoints = detector.detect(im)
	# if keypoints:
	# 	print("keypoints = ",keypoints[0].pt)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob

	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Show blobs
	# cv2.imshow("Keypoints", im_with_keypoints)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return keypoints

def detect_bridge(im):
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	 
	# Change thresholds
	params.minThreshold = 200
	params.maxThreshold = 256
	params.blobColor = 255
	 
	# Filter by Area.
	params.filterByArea = True
	params.minArea = 500
	params.maxArea = 2500
	 
	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1
	 
	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.87
	 
	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.001
	 
	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
	    detector = cv2.SimpleBlobDetector(params)
	else : 
	    detector = cv2.SimpleBlobDetector_create(params)

	keypoints = detector.detect(im)
	return keypoints

data_path = "./img"
dirname = sorted(os.listdir(data_path))
curr_frame = None
prev_frame = None
status= False
for filename in dirname:
	curr_frame = cv2.imread(os.path.join(data_path,filename))
	if curr_frame is not None:
		img_center_x = float(curr_frame.shape[1]/2)
		# print("image center = ", img_center_x)
		river_seg = thres_img(curr_frame)
		# cv2.imshow('thresh img',curr_frame)
		# cv2.waitKey(10)
		river_mask = np.float32(cv2.cvtColor(river_seg,cv2.COLOR_BGR2GRAY))
		river_mask[river_mask[:]>0.0] = 255.0
		river_mask = np.uint8(river_mask)
		kernel = (5,5)
		river_mask = cv2.GaussianBlur(river_mask,kernel,0)

		kernel = np.ones((11,11),np.uint8)
		river_mask = cv2.morphologyEx(river_mask, cv2.MORPH_CLOSE, kernel)

		# kernel = np.ones((5,5),np.uint8)
		river_mask = cv2.erode(river_mask,kernel,iterations = 1)
		river_mask[river_mask[:]>0.0] = 255.0
		river_mask = np.uint8(river_mask)	
		# cv2.imshow('mask ',river_mask)
		# cv2.waitKey(10)
		# cv2.destroyAllWindows()
		river_mask_3d = np.dstack((river_mask,river_mask,river_mask))
		keypoints = detect_blobs(river_mask)
		if keypoints:
			# for i in range(len(keypoints)):
				# print("response of"+str(i)+"keypoint",keypoints[i].response)
			num_keypoints = len(keypoints)
			if num_keypoints == 1:
				status = False
				error = - img_center_x
				return status, error

			if num_keypoints >= 2:
				# w0 = keypoints[0].size
				# w1 = keypoints[1].size

				# print("w0 and w1", w0,w1)
				# print("x0 y0 = ",x0,y0)
				xerr = int(((keypoints[1].pt[0] + keypoints[0].pt[0] )/2))
				yerr = int(((keypoints[1].pt[1] + keypoints[0].pt[1] )/2))
				# bridge_center= cv2.circle(river_mask_3d,(xerr,yerr),10,(255,0,0),-1)
				bridge_center= cv2.circle(river_mask_3d,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,0,0),-1)
				bridge_center= cv2.circle(river_mask_3d,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,0,0),-1)
				cv2.imshow("bridge_center",bridge_center)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				center_keypoints = detect_bridge(bridge_center)
				if center_keypoints:
					# print("response",center_keypoints[0].pt)
					x_bridge = center_keypoints[0].pt[0]
					y_bridge = center_keypoints[0].pt[1]
					cv2.circle(bridge_center,(int(x_bridge),int(y_bridge)),int(center_keypoints[0].size),(0,234,32),3)
					cv2.imshow("TEST	",bridge_center)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

					error = img_center_x - x_bridge

					print("error = ", error)
					status = True
					return status, error
					
				else:
					print("bridge not detected")
					status = False
					return status, 0.0
	status = False
	return status, 0.0