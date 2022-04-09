# (tensorflow_gpuenv) dhankar@dhankar-1:~/.../_book_3$ python test_2_openCV_SIFT_ORB.py 
# (tensorflow_gpuenv) dhankar@dhankar-1:~/.../_book_3$ python test_2_openCV_SIFT_ORB.py 
# (tensorflow_gpuenv) dhankar@dhankar-1:~/.../_book_3$ 
# (tensorflow_gpuenv) dhankar@dhankar-1:~/.../_book_3$ python test_1.py 


import cv2

#sift_feat = cv2.xfeatures2d.SIFT_create() ## gives a Warningg but seems to work ... 
#[ WARN:0] global /tmp/pip-req-build-xw6jtoah/opencv_contrib/modules/xfeatures2d/misc/python/shadow_sift.hpp (13) SIFT_create DEPRECATED: cv.xfeatures2d.SIFT_create() is deprecated due SIFT tranfer to the main repository. https://github.com/opencv/opencv/issues/16736

img = cv2.imread("./input_dir/image_0012.jpg", cv2.IMREAD_GRAYSCALE)
# img_0 = cv2.imread("./input_dir/image_0012.jpg", cv2.IMREAD_GRAYSCALE)
# img_1 = cv2.imread("./input_dir/image_0012.jpg", cv2.IMREAD_GRAYSCALE)
# #
#guitars

input_img_path1 = "./input_dir/guitars/image_0068.jpg"
input_img_path2 = "./input_dir/guitars/image_0070.jpg"

## AirPlanes
# input_img_path1 = "./input_dir/planes/image_0048.jpg"
# input_img_path2 = "./input_dir/planes/image_0049.jpg"

# img_className = input_img_path1.split('/')[0] # TODO -- Fix this like --->> face_1/face_recog22/pract_dl_1.py
# img_fileName = input_img_path1.split('/')[1] # TODO -- Fix this like --->> face_1/face_recog22/pract_dl_1.py

img_0 = cv2.imread(input_img_path1, cv2.IMREAD_GRAYSCALE)
img_1 = cv2.imread(input_img_path2, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create(nfeatures=150) # No Warning seems Ok 
#surf = cv2.SURF_create() #AttributeError: module 'cv2.cv2' has no attribute 'SURF_create'
orb = cv2.ORB_create(nfeatures=350) 
orb_small = cv2.ORB_create(nfeatures=50) 
#
keypoints_sift, descriptors = sift.detectAndCompute(img, None)
#keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb_1, descriptors_orb_1 = orb.detectAndCompute(img_0, None)
keypoints_orb_2, descriptors_orb_2 = orb.detectAndCompute(img_1, None)
#orb_small--- for the DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
keypoints_orb_small, descriptors = orb_small.detectAndCompute(img, None)
#
img_rich_keypoints_orb = cv2.drawKeypoints(img, keypoints_orb_small,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #None)
cv2.imshow("1_img_rich_keypoints_orb", img_rich_keypoints_orb)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite('./output_dir/orb_rich_keypoints_18_3.jpg',img_rich_keypoints_orb)
#
img_keypoints_orb = cv2.drawKeypoints(img, keypoints_orb_1, None)
cv2.imshow("2_img_keypoints_orb", img_keypoints_orb)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite('./output_dir/orb_keypoints_18_3.jpg',img_keypoints_orb)
#
img_keypoints_sift = cv2.drawKeypoints(img, keypoints_sift, None)
cv2.imshow("3_Image_keypoints_sift", img_keypoints_sift)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite('./output_dir/sift_keypoints_18_3.jpg',img_keypoints_sift)
#

# As an OpenCV enthusiast, the most important thing about the ORB is that it came from “OpenCV Labs”. This
# algorithm was brought up by Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary R. Bradski in their paper ORB:
# An efficient alternative to SIFT or SURF in 2011. As the title says, it is a good alternative to SIFT and SURF
# in computation cost, matching performance and mainly the patents. Yes, SIFT and SURF are patented and you are
# supposed to pay them for its use. But ORB is not !!!
#

# create BFMatcher object
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = brute_force_matcher.match(descriptors_orb_1,descriptors_orb_2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
bf_img_1 = cv2.drawMatches(img_0,keypoints_orb_1,img_1,keypoints_orb_2,matches[:15],None,flags=2) # #
# ERROR if no NONE --->> drawMatches() missing required argument 'outImg' #
# SO -- https://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found

cv2.imshow("Brute_Force_Match", bf_img_1)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite('./output_dir/bf_img_1.jpg',bf_img_1)

import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dt_time_now = datetime.now()
hour_now = dt_time_now.strftime("_%m_%d_%Y_%H%M%s")
print(str(hour_now))
cv2.imwrite('./output_dir/bf_img_'+str(hour_now)+'_.jpg',bf_img_1)

# img_output_dir_path = './output_dir/_SimilarImg_/'+str(img_className)+'_similar_'+ str(hour_now)
# if not os.path.exists(img_output_dir_path):
#     os.makedirs(img_output_dir_path)
# plt.imsave(str(img_output_dir_path)+str(img_fileName)+"_.png", bf_img_1)     
# #cv2.imwrite(str(img_fileName)+"_.png",bf_img_1)






