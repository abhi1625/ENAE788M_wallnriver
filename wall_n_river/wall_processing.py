import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

def detect_blobs(im):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
            
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 255
        params.blobColor = 255
        params.filterByColor = True
            
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2500
        params.maxArea = 10000000000000
            
        # # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 0.1
            
        # # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 0.9
            
        # # Filter by Inertia
        params.filterByInertia = False
        # params.minInertiaRatio = 0.1
            
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(im)
        # if keypoints:
        #     print("keypoints = ",keypoints[0].pt)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Show blobs
        # cv2.imshow("Keypoints", im_with_keypoints)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        return keypoints

def get_wall_center(cv_image):
    kernel = np.ones(30)
    h = cv_image.shape[0]
    w = cv_image.shape[1]
    closing = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
    thresh = np.mean(closing[:])
    closing[closing < thresh] = 0
    closing_resize = cv2.resize(closing, (w/4, h/4))
    Z = closing_resize.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)
    print(np.amax(label), np.amin(label), center.shape)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    max_center = np.amax(res)
    print(max_center)
    res[res<max_center] = 0
    res[res>=max_center] = 255
    res2 = res.reshape((h/4,w/4,3))
    closing = cv2.resize(res2,(w,h))
    closing = np.uint8(closing)
    keypoints = detect_blobs(closing)
    print("length",len(keypoints))
    if len(keypoints) >= 1:
        center_x = np.mean(np.array(keypoints[0].pt[0]))
        center_y = np.mean(np.array(keypoints[0].pt[1]))
        cv2.circle(cv_image,(int(center_x),int(center_y)),int(keypoints[0].size/2),(255,0,255),2)
    cv2.imshow("test_img",cv_image)
    cv2.waitKey(1)
    return keypoints



cv_image = None
bridge = CvBridge()
def flow_cb(data):
    try:
        global cv_image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        print(cv_image.shape)
    except CvBridgeError as e:
        print(e)
if __name__ == "__main__":
    global cv_image
    rospy.init_node("test", anonymous=True)
    rospy.Subscriber("/flow_img", Image, flow_cb)
    while(not rospy.is_shutdown()):
        if cv_image is not None:
            get_wall_center(cv_image)
            
