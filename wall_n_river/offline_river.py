import cv2
import numpy as np
import os



def segment_river(cv_image):
	# river params dark night
	thresh_r_min=87
	thresh_g_min=81
	thresh_b_min=133
	thresh_r_max=155
	thresh_g_max=205
	thresh_b_max=255

	# # river params bright day
	# thresh_r_min=16
	# thresh_g_min=43
	# thresh_b_min=0
	# thresh_r_max=88
	# thresh_g_max=255
	# thresh_b_max=255

	# river params bright day
	# thresh_r_min=16
	# thresh_g_min=72
	# thresh_b_min=0
	# thresh_r_max=110
	# thresh_g_max=248
	# thresh_b_max=248

	cv_image[cv_image[:,:,0] < thresh_b_min]=0
	cv_image[cv_image[:,:,1] < thresh_g_min]=0
	cv_image[cv_image[:,:,2] < thresh_r_min]=0

	cv_image[cv_image[:,:,0] > thresh_b_max]=0
	cv_image[cv_image[:,:,1] > thresh_g_max]=0
	cv_image[cv_image[:,:,2] > thresh_r_max]=0
	return cv_image




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
	params.maxArea = 8000
	 
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



def detect_blobs(im):
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	 
	# Change thresholds
	params.minThreshold = 200
	params.maxThreshold = 256
	params.blobColor = 255
	 
	# Filter by Area.
	params.filterByArea = True
	params.minArea = 7000
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


def transform_img(img):
	per = 0.6
	h = 460
	w = 800
	# pts_src = np.array([[0.0, 0.0],[0.0, self.w],[self.h, self.w],[self.h, 0]],dtype = np.float32)
	# pts_dst = np.array([[self.h*per, 0],[0, self.w*per],[self.h, self.w],[self.h, 0]],dtype = np.float32)
	# print("src and dst = ",pts_src,pts_dst)
	# h, status = cv2.findHomography(pts_src, pts_dst)		 
	# im_dst = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))
	crop = img[int(h*per):,:]
	crop = cv2.resize(crop,(w,h),interpolation = cv2.INTER_AREA)
	
	return crop

def bridge_detect(curr_frame):
	if curr_frame is not None:
		print("inside curr frame")
		img_center_x = float(curr_frame.shape[1]/2)
		img_center_y = float(curr_frame.shape[0]/2)
		# print("image center = ", img_center_x)

		curr_frame = segment_river(curr_frame)
		cv2.imwrite("threshold.jpg",curr_frame)
		cv2.imshow('thresh img',curr_frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		river_seg = transform_img(curr_frame)
		cv2.imwrite("ROI.jpg",river_seg)
		cv2.imshow('after homography',river_seg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		river_mask = np.float32(cv2.cvtColor(river_seg,cv2.COLOR_BGR2GRAY))
		# river_mask[:int(img_center_y),:] = 0.0
		river_mask = np.uint8(river_mask)
		cv2.imwrite("before_blur.jpg",river_mask)
		cv2.imshow('before_blur ',river_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		kernel = (15,15)
		# river_mask = cv2.GaussianBlur(river_mask,kernel,0)
		river_mask = cv2.medianBlur(river_mask,15,0)
		cv2.imwrite("medianBlur.jpg",river_mask)
		cv2.imshow('median ',river_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		kernel = np.ones((7,7),np.uint8)
		river_mask = cv2.morphologyEx(river_mask, cv2.MORPH_OPEN, kernel)
		kernel = np.ones((20,20),np.uint8)
		river_mask = cv2.morphologyEx(river_mask, cv2.MORPH_CLOSE, kernel)

		# cv2.imshow('closing ',river_mask)
		# cv2.waitKey(10)
		# kernel = np.ones((5,5),np.uint8)
		# river_mask = cv2.dilate(river_mask,kernel,iterations = 1)
		river_mask[river_mask[:]>0.0] = 255.0
		#river_mask[river_mask[:]<] = 255.0

		river_mask = np.uint8(river_mask)
		cv2.imwrite("binary_mask.jpg",river_seg)

		cv2.imshow('mask ',river_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		river_mask_3d = np.dstack((river_mask,river_mask,river_mask))
		
		keypoints_first = detect_bridge(river_mask)
		# if keypoints_first:
		# 	# self.error = img_center_x - keypoints_first[0].pt[0]
		# 	# status = True
		# 	# return status, self.error
		# 	if len(keypoints_first) ==1:
		# 		print("first frame bridge detect")
		# 		im_with_key = cv2.circle(river_mask_3d.copy(),(int(keypoints_first[0].pt[0]),int(keypoints_first[0].pt[1])),int(keypoints_first[0].size),(0,0,255),4)

		# 		# Show blobs
		# 		cv2.imwrite("bridge-blob.jpg",im_with_key)
		# 		cv2.imshow("bridge-blob", im_with_key)
		# 		cv2.waitKey(0)
		# 		cv2.destroyAllWindows()
		# 	else:
		# 		print("multiple blobs found in first scene")
		# 		keypoints = detect_blobs(river_mask)

		# 		# if keypoints:
		# 		# 	num_keypoints = len(keypoints)
		# 		# 	if num_keypoints == 1:
		# 		# 		print("single blob after multiple bridge blobs")
		# 		# 		blob_img= cv2.circle(river_mask_3d.copy(),(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,255,0),4)
		# 		# 		cv2.imwrite("single_river_blob_m_b.jpg",blob_img)
		# 		# 		cv2.imshow("single_river_blob", blob_img)
		# 		# 		cv2.waitKey(0)
		# 		# 		cv2.destroyAllWindows()

		# 		# 	if num_keypoints >= 2:
		# 		# 		print("two blob after multiple bridge blobs")

		# 		# 		xerr = int(((keypoints[1].pt[0] + keypoints[0].pt[0] )/2))
		# 		# 		yerr = int(((keypoints[1].pt[1] + keypoints[0].pt[1] )/2))
		# 		# 		# bridge_center= cv2.circle(river_mask_3d,(xerr,yerr),10,(255,0,0),-1)
		# 		# 		cv2.circle(river_mask_3d,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,255,0),4)
		# 		# 		cv2.circle(river_mask_3d,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,255,0),4)
		# 		# 		cv2.imwrite("two_river_blob_m_b.jpg",river_mask_3d)
		# 		# 		cv2.imshow("two_river_blob_m_b", river_mask_3d)
		# 		# 		cv2.waitKey(0)
		# 		# 		cv2.destroyAllWindows()

		# 		# 		bridge_center= cv2.circle(river_mask_3d,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,0,0),-1)
		# 		# 		bridge_center= cv2.circle(river_mask_3d,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,0,0),-1)
		# 		# 		# cv2.i11mshow("bridge_center",bridge_center)
		# 		# 		# self.blobs_pub.publish()
		# 		# 		# cv2.waitKey(0)
		# 		# 		# cv2.destroyAllWindows()
		# 		# 		center_keypoints = detect_bridge(bridge_center)
		# 		# 		if center_keypoints:
		# 		# 			# print("response",center_keypoints[0].pt)
		# 		# 			x_bridge = center_keypoints[0].pt[0]
		# 		# 			y_bridge = center_keypoints[0].pt[1]
		# 		# 			cv2.circle(bridge_center,(int(x_bridge),int(y_bridge)),int(center_keypoints[0].size),(0,234,32),3)
		# 		# 			#cv2.imshow("TEST	",bridge_center)
		# 		# 			# pub_img = self.bridge.cv2_to_imgmsg(bridge_center)
		# 		# 			# self.bridge_pub.publish(pub_img)
		# 		# 			#cv2.waitKey(1)
		# 		# 			cv2.imwrite("bridge_after_two_river.jpg",bridge_center)
		# 		# 			cv2.imshow("bridge_after_two_river", bridge_center)
		# 		# 			cv2.waitKey(0)
		# 		# 			cv2.destroyAllWindows()
		# 		# 			# cv2.destroyAllWindows()
							
		# 		# 		else:
		# 		# 			print("bridge not detected")

		# else:
		# 	# self.error = -400.0

		keypoints = detect_blobs(river_mask)

		if keypoints:
			num_keypoints = len(keypoints)
			if num_keypoints == 1:
				
				blob_img= cv2.circle(river_mask_3d.copy(),(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,255,0),4)
				cv2.imwrite("single_river_blob.jpg",blob_img)
				cv2.imshow("single_river_blob", blob_img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			if num_keypoints >= 2:
				print("2 pts")
				xerr = int(((keypoints[1].pt[0] + keypoints[0].pt[0] )/2))
				yerr = int(((keypoints[1].pt[1] + keypoints[0].pt[1] )/2))
				# bridge_center= cv2.circle(river_mask_3d,(xerr,yerr),10,(255,0,0),-1)
				bridge_center= cv2.circle(river_mask_3d,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,0,0),-1)
				bridge_center= cv2.circle(river_mask_3d,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,0,0),-1)
				# cv2.i11mshow("bridge_center",bridge_center)
				# self.blobs_pub.publish()
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				center_keypoints = detect_bridge(bridge_center)
				if center_keypoints:
					# print("response",center_keypoints[0].pt)
					x_bridge = center_keypoints[0].pt[0]
					y_bridge = center_keypoints[0].pt[1]
					# bridge_center= cv2.circle(river_mask_3d,(xerr,yerr),10,(255,0,0),-1)
					river_mask_3d_dr = river_mask_3d.copy()
					cv2.circle(river_mask_3d_dr,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,255,0),4)
					cv2.circle(river_mask_3d_dr,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,255,0),4)
					cv2.imwrite("two_river_blob_m_b.jpg",river_mask_3d_dr)
					cv2.imshow("two_river_blob_m_b", river_mask_3d_dr)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

					river_mask_3d_aa = river_mask_3d.copy()
					cv2.circle(river_mask_3d_aa,(int(keypoints[0].pt[0]),int(keypoints[0].pt[1])),int(keypoints[0].size),(0,0,0),-1)
					cv2.circle(river_mask_3d_aa,(int(keypoints[1].pt[0]),int(keypoints[1].pt[1])),int(keypoints[1].size),(0,0,0),-1)
					# cv2.i11mshow("bridge_center",bridge_center)
					# self.blobs_pub.publish()
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
					center_keypoints = detect_bridge(river_mask_3d_aa)
					if center_keypoints:
						# print("response",center_keypoints[0].pt)
						x_bridge = center_keypoints[0].pt[0]
						y_bridge = center_keypoints[0].pt[1]
						cv2.circle(river_mask_3d_aa,(int(x_bridge),int(y_bridge)),int(center_keypoints[0].size),(0,234,32),3)
						#cv2.imshow("TEST	",bridge_center)
						# pub_img = self.bridge.cv2_to_imgmsg(bridge_center)
						# self.bridge_pub.publish(pub_img)
						#cv2.waitKey(1)
						cv2.imwrite("bridge_after_two_river.jpg",river_mask_3d_aa)
						cv2.imshow("bridge_after_two_river", river_mask_3d_aa)
						cv2.waitKey(0)
						cv2.destroyAllWindows()
						# cv2.destroyAllWindows()
					
				else:
					print("bridge not detected")
					# status = False
					# return status, 0.0
	status = False
	# return status, 0.0



def main():
	data_path = "./img"
	dirname = sorted(os.listdir(data_path))
	curr_frame = None
	prev_frame = None
	for filename in dirname:
		curr_frame = cv2.imread(os.path.join(data_path,filename))
		# print(os.path.join(data_path,filename))
		cv2.imshow("Original image",curr_frame)
		cv2.imwrite("Original_img.jpg",curr_frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# curr_frame = thres_img(curr_frame)
		# cv2.imshow('thresh img',curr_frame)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		bridge_detect(curr_frame)

if __name__ == '__main__':
	main()
