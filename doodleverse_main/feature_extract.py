import numpy as np 
import cv2 #openCV library
from matplotlib import pyplot as plt


def main():
	 #import image as grayscale, if image is drawn by hand, cv2.threshold THRESH_BINARY assures it is blk/white
	 #example shapes: shape1.png (odd,no int), shape2.png (odd,no int), shape3.png (rounded, no int)
	raw_img = cv2.imread('shape2.png',0)
	#ret,raw_img = cv2.threshold(raw_img,127,255,cv2.THRESH_BINARY) #incorrect parameters

	#create grayscale (uint8) all white image to draw features onto
	draw_img = 1*np.ones((raw_img.shape[0],raw_img.shape[1],1), np.uint8)

	#find contours and select outermost contour that is not the border, input image must be black/white
	#this only finds continuous contours, would need to change a setting if the image is U-shapped or something
	image, contours, hierarchy = cv2.findContours(raw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	cnt = contours[1] 
	draw_img = cv2.drawContours(draw_img, cnt, -1, 255, 1) #draw_img now has the contour drawn on it in black

	#build mask for contour
	mask = cv2.minAreaRect(cnt)

	#grab extreme points of contour
	extrema = getExtrema(cnt)

	#use harris corner detector 
	corners = getCorners(draw_img,10,0.1,50)

	#consolidate to best features 
	max_error = 0.2
	nom_error = 0.05
	features = consolidateFeatures(cnt,extrema,corners,max_error,nom_error)

	#plot feature points
	fig1 = plt.figure(1)
	plt.imshow(draw_img.squeeze(),cmap='Greys')
	plt.scatter(corners[:,0],corners[:,1],s=10,c='b',marker='x')
	#plt.plot(corners[:,0],corners[:,1])
	#plt.scatter(extrema[:,0],extrema[:,1],s=20,c='r',marker='o')
	plt.title('Feature Map')
	plt.show()

	print(corners.shape, '\nCorners:\n', corners, '\n\nExtrema:\n',extrema)


"""
getExtrema(contour)
Function purpose: Takes a contour and returns a 4x1 array with the left,right,top,bottom extreme points

INPUTS: 
contour = the contour (x,y) points that the extrema will be found on

OUTPUTS:
extrema = an array of (x,y) points that contains the left-most, right-most, top-most, and bottom-most points of the contour

PROBLEMS:
none
"""
def getExtrema(contour):
	left  = list(contour[contour[:,:,0].argmin()][0])
	right = list(contour[contour[:,:,0].argmax()][0])
	top   = list(contour[contour[:,:,1].argmin()][0])
	btm   = list(contour[contour[:,:,1].argmax()][0])
	extrema = np.array(list((left,right,top,btm)))
	return extrema

"""
getCorners(image,num_corners,quality_factor,min_dist)
Function purpose: Takes an image and parameters to return normers in a 10x2 array of coordinates using Harris Corner Detector

INPUTS:
image = image to find corners on
num_corners = maximum number of corners to find
quality_factor = level that determines a quality corner, if max corner has quality 1500, quality_factor = 0.1, then only corners with 150 quality are considered
min_dist = minimum distace from a quality corner to consider adding a new unique corner

OUTPUTS:
corners = an nx2 array of (x,y) coordinates for the strongest corners that fit the input critera, where n is the number of valid corners found

PROBLEMS:
1. does not order the corners in a good way (i.e. clockwise) - need to implement code to reorder corner points so that they are in some kind of order
2. if there exists an interior line with corners or a dot or some extra shape, this function will find corners that are not on the outline contour, so ordering the contour directly from this function does not seem feasible
"""
def getCorners(image,num_corners,quality_factor,min_dist):
	corners = cv2.goodFeaturesToTrack(image,num_corners,quality_factor,min_dist)
	corners = np.int0(corners)
	corners = corners.reshape(corners.shape[0],2)
	return corners 

"""
consolidateFeatures(contour,extrema,corners,error_bound)
Function purpose: return an array of features that includes a minimum number of features that meet the error_bound when linearly splined

INPUTS: 
contour = contour point array that will be used as the basis of the error_bound
extrema = array of [left-most, right-most, top-most, bottom-most] (x,y) coordinates of the contour
corners = corner features returned by getCorners() function
max_error = maximum acceptable error (decimal 0 to 1) between linearly splined feature points and true contour connecting those feature points, normalized by distace between feature points
nominal_error = small error level (decimal 0 to 1) between three feature points that is considered essentially nothing - ie the central feature point can be removed without any issues, normalized by distance between feature points

OUTPUTS: 
features = an ordered array of consolidated feature points in (x,y) coordinates

PROBLEMS:
1. not yet implemented
"""
def consolidateFeatures(contour,extrema,corners,max_error,nominal_error):
	#order the extrema within the set of corners points so that the order of corners is preserved

	#REMOVE midpoints, 
	#for each point, 
	#	look at the point to the clockwise direction and check if the next two points are both on the same line (ie nominal error from pt1 to pt3 is not exceeded)
	#	remove mid point as necessary and restart iteration until all midpoints are above nominal error

	#ADD new points
	#for each point, if error between point > max_error, perform recursive method:
	#	add midpoint on contour connecting pt1 and pt2 = ptA
	#	test errors e1 = error between pt1 and ptA, e2 = error between pt2 and ptA
	#	if e1 and e2 both within max error, return. 
	#	else, add midpoint between ptA and larger error point and test error between ptB and pt2 (if that had the high error)
	#		continue testing error between new midpoint and pt2 until within max_error
	#	add new midpoint to full feature set and restart iteration to add new points

	return 

if __name__ == '__main__': main()
