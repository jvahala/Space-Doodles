import numpy as np 
import cv2 #openCV library
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
np.set_printoptions(threshold=np.nan)

def main():
	 #import image as black/white 
	 #example shapes: shape1.png (odd,no int), shape2.png (odd,no int), shape3.png (rounded, no int)
	raw_img, image, contours, hierarchy = importImage('shape5.png')
	img_main=mpimg.imread('shape5.png')
	cnt = contours[1] #contour zero is border, contour 1 is outermost contour, ...etc
	cnt2 = cleanContour(cnt) #makes distances between contour points more even

	#create grayscale (uint8) all white image to draw features onto
	draw_img = drawImage(raw_img,cnt2)

	#grab extreme points of contour
	extrema = getExtrema(cnt2)

	#use harris corner detector 
	corners = getCorners(draw_img,10,0.1,50)
	features = orderFeatures(cnt2,extrema,corners)

	#consolidate features
	add_threshold = 0.01 #smaller values add more points (0.01 default)
	remove_threshold = 0.01 #larger values mean less points (0.01 default)
	n = 5#number of divisions for determining normalized error (5 default)
	index = 0 #default starting index (0 default)

	count = 0
	new_features = features

	#add a bunch of features 
	new_features = addFeatures(index,new_features,cnt2,n,add_threshold)
	new_features = addFeatures(index,new_features,cnt2,n,add_threshold*0.1)

	#remove them slowly
	new_features = removeMidpoints(index,new_features,cnt2,n,remove_threshold)
	new_features = removeMidpoints(index,new_features,cnt2,n,remove_threshold*10)
	new_features = removeMidpoints(index,new_features,cnt2,n,remove_threshold*30)
	new_features = removeMidpoints(index,new_features,cnt2,n,remove_threshold*50)
\
	print('Original/New/difference',features.shape[0],'/',new_features.shape[0],'/',new_features.shape[0]-features.shape[0])
	best_features_sorted = findKeyFeatures(new_features)
	print(best_features_sorted)
	print(new_features)

	#plot feature points
	fig1 = plt.figure(1)
	plt.subplot(221)
	plt.imshow(img_main)
	plt.title('(a) Original Image', fontsize=10)
	frame = plt.gca()
	frame.axes.get_xaxis().set_ticks([])
	frame.axes.get_yaxis().set_ticks([])
	plt.subplot(222)
	plt.imshow(draw_img.squeeze(),cmap='Greys')
	plt.title('(b) Contour', fontsize=10)
	frame = plt.gca()
	frame.axes.get_xaxis().set_ticks([])
	frame.axes.get_yaxis().set_ticks([])
	plt.subplot(223)
	plt.imshow(draw_img.squeeze(),cmap='Greys')
	plt.hold(True)
	plt.scatter(features[:,0],features[:,1],s=20,c='b',marker='x')
	plt.plot(features[:,0],features[:,1])
	plt.title('(c) Harris Corner Detector Features', fontsize=10)
	plt.axis('image')
	frame = plt.gca()
	frame.axes.get_xaxis().set_ticks([])
	frame.axes.get_yaxis().set_ticks([])
	plt.subplot(224)
	plt.imshow(draw_img.squeeze(),cmap='Greys')
	plt.hold(True)
	plt.scatter(new_features[:,0],new_features[:,1],s=20,c='r',marker='x')
	plt.plot(new_features[:,0],new_features[:,1],'r-')
	best_index = best_features_sorted[0,1]
	best_triangle = new_features[2:5,:]
	plt.scatter(best_triangle[:,0],best_triangle[:,1],s=30,facecolors='none',edgecolors='g',marker='o')
	plt.title('(d) Optimized Features', fontsize=10)
	plt.axis('image')
	frame = plt.gca()
	frame.axes.get_xaxis().set_ticks([])
	frame.axes.get_yaxis().set_ticks([])
	plt.show()

"""
raw_img, image, contours, hierarchy = importImage(image)

Function purpose: imports image as black/white with contours 

INPUTS: 
image = typically a .png file (string)

OUTPUTS:
raw_img = raw image data points
image = dummy
contours = all contours in the image 
hierarchy = ranking of the contours 

PROBLEMS:
none
"""
def importImage(image):
	raw_img = cv2.imread(image,0)
	image, contours, hierarchy = cv2.findContours(raw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	return raw_img, image, contours, hierarchy

"""
draw_img = drawImage(raw_img,contour)

Function purpose: creates draw image variable that holds black/white data and the contours you pass to the function

INPUTS: 
raw_img = raw image from import image function
contour = contours you pass to the function

OUTPUTS:
draw_img = black/white image of same shape as raw image with contours drawn on it 

PROBLEMS:
none
"""
def drawImage(raw_img,contour):
	draw_img = 1*np.ones((raw_img.shape[0],raw_img.shape[1],1), np.uint8)
	draw_img = cv2.drawContours(draw_img, contour, -1, 255, 1) #draw_img now has the contour drawn on it in black
	return draw_img 

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
orderFeatures(contour,extrema,corners)

Function purpose: return an ordered array of features 

INPUTS: 
contour = contour point array that will be used as the basis of the error_bound
extrema = array of [left-most, right-most, top-most, bottom-most] (x,y) coordinates of the contour
corners = corner features returned by getCorners() function

OUTPUTS: 
features = an ordered array of feature points in (x,y) coordinates

PROBLEMS:
1. 
"""
def orderFeatures(contour,extrema,corners):
	features_temp = np.vstack((extrema,corners))
	features_temp.shape = (features_temp.shape[0],2)
	features = np.zeros((2,2))
	contour.shape = (contour.shape[0],2)
	#print(contour.shape, features_temp.shape)
	#go over contour and add features as they are reached in the iteration
	feat_it = 0 #initialization parameter
	for i in range(contour.shape[0]): 
		for j in range(features_temp.shape[0]):
			dist = distance(contour[i,:],features_temp[j,:])
			if dist < 3:
				if feat_it <= 1: 
					features[feat_it,:] = features_temp[j,:]
					feat_it = feat_it + 1
				else:
					if distance(features[-1,:],features[-2,:]) < 5: 
						features[-1,:] = features_temp[j,:]
					else:
						features = np.vstack((features,features_temp[j,:]))
					feat_it = 2
	return features

""" 
contour = cleanContour(contour)

Function purpose: turn contour into a set of points the same average distance apart

INPUTS: 
contour = (x,y) ordered coordinates of the contour shape

OUTPUTS: 
contour = (x,y) ordered coordinates with similar distance between points, slightly less error invloved

PROBLEMS:
1. 
"""
def cleanContour(contour):
	total_dist = 0
	cnt_length = contour.shape[0]-1
	new_contour = contour[0,:]
	for k in range(cnt_length):
		dist_pts = distance(contour[k,:],contour[k+1,:])
		if dist_pts > .99 and dist_pts < 1.01:
			new_contour = np.vstack((new_contour,contour[k,:]))
		elif dist_pts > 1.41 and dist_pts <1.42:
			point_1 = contour[k,:]
			point_2 = contour[k,:]
			point_1.shape = (1,2)
			point_2.shape = (1,2)
			percent_bisect = 0.5
			x_spline = int(point_1[0,0] - (point_1[0,0] - point_2[0,0])*(1-percent_bisect))
			y_spline = int(point_1[0,1] - (point_1[0,1] - point_2[0,1])*(1-percent_bisect))
			spline_bisect = np.array([x_spline,y_spline])
			spline_bisect.shape = (1,2)
			new_contour = np.vstack((new_contour,spline_bisect,contour[k,:]))
	new_contour.shape = (new_contour.shape[0],1,2)
	return new_contour 

""" 
dist = distance(point_1,point_2)

Function purpose: return euclidian distance between points

INPUTS: 
point_1 = [x,y] array that can be shaped into a 1 by 2 array
point_2 = [x,y] array that can be shaped into a 1 by 2 array

OUTPUTS: 
dist = euclidian distance between points, absolute scalar value

PROBLEMS:
1. 
"""
def distance(point_1,point_2):
	# reshape points
	point_1.shape = (1,2)
	point_2.shape = (1,2)
	# calculate distance 
	dist = np.sqrt((point_1[0,0]-point_2[0,0])**2 + (point_1[0,1]-point_2[0,1])**2)
	return np.abs(dist)

""" 
cnt_bisect,spline_bisect = findBisect(point_1,point_2,percent_bisect,contour)

Function purpose: return the bisection point on the contour between two points along with the [x,y] midpoint between the points themselves

INPUTS: 
point_1 = [x,y] array that can be shaped into a 1 by 2 array
point_2 = [x,y] array that can be shaped into a 1 by 2 array
percent_bisect = decimal between 0 and 1 representing how close to point_1 to make the bisect (0.3 splits the spline 3/10 of the way from point 1 to point 2)
contour = contour point array to find the bisection point on 

OUTPUTS: 
cnt_bisect = 1x2 [x,y] point on the contour located between the two points
spline_bisect = 1x2 [x,y] point located at the midpoint between the points themselves

PROBLEMS:
1. points on the contour are not linearly spaced, so direct (add the percent of total points between feature points along the contour does not work great)
"""
def findBisect(point_1,point_2,percent_bisect,contour):
	#reshape points
	point_1.shape = (1,2)
	point_2.shape = (1,2)
	contour.shape = (contour.shape[0],2)
	contour_long = np.vstack((contour,contour))
	contour_long.shape = (contour_long.shape[0],2)
	#initialize 
	skip_1 = 0
	skip_2 = 0
	#get split point between spline of points
	x_spline = int(point_1[0,0] - (point_1[0,0] - point_2[0,0])*(1-percent_bisect))
	y_spline = int(point_1[0,1] - (point_1[0,1] - point_2[0,1])*(1-percent_bisect))
	spline_bisect = np.array([x_spline,y_spline])
	spline_bisect.shape = (1,2)
	#get contour bisect between the two points
	#move along contour to find contour point numbers for both point 1 and 2. Set flag that point has been found when done
	for i in range(contour.shape[0]-1): 
		dist1 = distance(point_1,contour[i,:])
		dist2 = distance(point_2,contour[i,:])
		#print('dist1/dist2: ',dist1,' / ', dist2,'    skip1/skip2: ',skip_1, ' / ', skip_2)
		if (dist1 < 10) and (skip_1 == 0):
			tmp_dist = distance(point_1,contour[i,:])
			nxt_dist = distance(point_1,contour[i+1,:])
			if tmp_dist < nxt_dist:
				cnt_pt_1 = i 
				skip_1 = 1
				#print("\npoint 1 found. \n")
		if (dist2 < 10) and (skip_2 == 0):
			tmp_dist = distance(point_2,contour[i,:])
			nxt_dist = distance(point_2,contour[i+1,:])
			if tmp_dist < nxt_dist:
				cnt_pt_2 = i 
				skip_2 = 1
				#print('\npoint 2 found. \n')
	#if flag for points being found are valid, count number of points and return [x,y] coordinate of that bisecting point, else print error that both points not found
	if (skip_1 == 1) and (skip_2 == 1):
		cnt_1_larger = cnt_pt_1 > cnt_pt_2
		#print(cnt_1_larger, cnt_pt_1, cnt_pt_2, contour.shape[0])
		cnt_1_shift = cnt_pt_1 + contour.shape[0]
		cnt_2_shift = cnt_pt_2 + contour.shape[0]
		if cnt_1_larger:
			total_points = cnt_1_shift - cnt_2_shift 
			cnt_index = cnt_2_shift - contour.shape[0]
			if total_points > contour.shape[0]/2:
				total_points = 2*contour.shape[0] - cnt_1_shift
				cnt_index = cnt_1_shift - contour.shape[0]
			
		else:
			total_points = cnt_2_shift - cnt_1_shift 
			cnt_index = cnt_1_shift - contour.shape[0]
			if total_points > contour.shape[0]/2:
				total_points = 2*contour.shape[0] - cnt_2_shift
				cnt_index = cnt_2_shift - contour.shape[0]
			
		#print(cnt_index)
		cnt_index = cnt_index + int(total_points*(1-percent_bisect))
		#print(cnt_index)
		cnt_bisect = contour[cnt_index,:]
		cnt_bisect.shape = (1,2)
	return cnt_bisect, spline_bisect

""" 
normError = getNormError(point_1,point_2,contour,n)

Function purpose: return the error between the contour and two features points normalized by the distance between feature points

INPUTS: 
point_1 = [x,y] array that can be shaped into a 1 by 2 array
point_2 = [x,y] array that can be shaped into a 1 by 2 array
contour = contour point array to find the bisection point on 
n = number of bisections to perform when calculation total error (ie n = 5 means bisections are used)

OUTPUTS: 
normError = value of total error normalized by the distance between the two points

PROBLEMS:
1. 
"""
def getNormError(point_1,point_2,contour,n):
	#scan through percent bisects 0 to 1 using n points, sum errors, normalize by the distance between feature points
	dist_p1p2 = distance(point_1,point_2)
	start = 1/n 
	percent = np.linspace(start,1.0-start,num=n,endpoint=True)
	absError = 0
	for i in range(n):
		cnt_bisect, spline_bisect = findBisect(point_1,point_2,percent[i],contour)
		absError = absError + distance(cnt_bisect,spline_bisect)
	if dist_p1p2 > 5:
		normError = absError / dist_p1p2
	else:
		normError = 0; 
	#print('absError/dist_p1p2/normError: ', absError,' / ', dist_p1p2, ' / ', normError) 
	return normError 

""" 
features = removeMidpoints(pt1_index, features, contour, n, threshold)

Function purpose: remove midpoints between features points that approximate the contour within some error threshold

INPUTS: 
pt1_index = index of feature array to start iteration 
features = set of feature points [x,y] indexed by pt1_index and pt2_index
contour = contour point array [x,y]
n = number of bisections to perform when calculation total error (ie n = 5 means bisections are used)
threshold = error threshold for comparing iterative errors along the contour 

OUTPUTS: 
features = new array of feature points with redundant features removed

PROBLEMS:
1. 
"""
def removeMidpoints(pt1_index, features, contour, n, threshold):
	#will need to add in wrap around ability and other indexing stuff
	if pt1_index < 0:
		pt1_index = abs(pt1_index)
		pt2_index = pt1_index - 1
		if pt2_index > 0:
			normErr_12 = getNormError(features[pt1_index,:],features[pt2_index,:], contour, n)
			normErr_13 = getNormError(features[pt1_index,:],features[pt2_index+1,:], contour, n)
			if normErr_13-normErr_12 < threshold:
				features  = np.delete(features,(pt2_index),(0))
				pt1_index = -1*(pt1_index - 1)
				features = removeMidpoints(pt1_index,features,contour,n,threshold)
			else: 
				pt1_index = -1*(pt1_index-1)
				features = removeMidpoints(pt1_index,features,contour,n,threshold)
	elif pt1_index >= 0:
		pt2_index = pt1_index + 1
		if pt2_index < features.shape[0] - 1:
			normErr_12 = getNormError(features[pt1_index,:],features[pt2_index,:], contour, n)
			normErr_13 = getNormError(features[pt1_index,:],features[pt2_index+1,:], contour, n)
			if normErr_13-normErr_12 < threshold:
				features  = np.delete(features,(pt2_index),(0))
				features = removeMidpoints(pt1_index,features,contour,n,threshold)
			else: 
				pt1_index = pt1_index+1
				features = removeMidpoints(pt1_index,features,contour,n,threshold)
	return features 

""" 
features = addFeatures(pt1_index, features, contour, n, threshold)

Function purpose: adds features by bisecting feature pairs that have high normalized error

INPUTS: 
pt1_index = index of feature array to start iteration (use 0 to iterate through whole contour)
features = set of feature points [x,y] indexed by pt1_index and pt2_index
contour = contour point array [x,y]
n = number of bisections to perform when calculation total error (ie n = 5 means bisections are used)
threshold = error threshold for determining whether to add a bisection 

OUTPUTS: 
features = new array of feature points with redundant features removed

PROBLEMS:
1. 
"""
def addFeatures(pt1_index, features, contour, n, threshold):
	pt2_index = pt1_index + 1
	if pt2_index < features.shape[0]:
		#print(features.shape[0], ' wow')
		normErr = getNormError(features[pt1_index,:],features[pt2_index,:], contour, n)
		#print('normErr: ', normErr)
		if normErr > threshold and normErr < 1000:
			#print('whoops ')
			#do stuff 
			cnt_bisect,spline_bisect = findBisect(features[pt1_index,:],features[pt2_index,:],0.5,contour)
			features = np.insert(features, pt2_index, cnt_bisect, axis = 0 )
			pt1_index = pt1_index+2
			features = addFeatures(pt1_index, features, contour, n, threshold)
		else:
			#print('wheeee')
			pt1_index = pt1_index + 1
			features = addFeatures(pt1_index, features, contour, n, threshold)
	return features

""" 
most_important_sorted = findKeyFeatures(features)

Function purpose: finds the key features based on how much they account for the image correctness. Large distances betwen adjacent features and angles near pi/2 rad are considered most important. 

INPUTS: 
features = set of features points to determine most important features from 

OUTPUTS: 
most_important_sorted = returns nx2 [importance value, feature index] of sorted values by important

PROBLEMS:
1. 
"""
def findKeyFeatures(features):
	most_important = np.zeros((features.shape[0]-1,2))
	for k in range(features.shape[0] - 1):
		kplus1 = k + 1
		if k == 0:
			kmin1 = features.shape[0] - 2 #k-1 term is the first term at the backend of the closed feature set that is not the 0th term
		else:
			kmin1 = k - 1
		lmin1 = distance(features[k,:],features[kmin1,:])
		lplus1 = distance(features[k,:],features[kplus1,:])
		vec1 = features[kmin1,:] - features[k,:]
		vec2 = features[kplus1,:] - features[k,:]
		vec1_u = vec1/np.linalg.norm(vec1)
		vec2_u = vec2/np.linalg.norm(vec2)
		angle = np.arccos(np.dot(vec1_u,vec2_u))
		if np.isnan(angle):
			if vec1_u.all() == vec2_u.all():
				angle = 0.0
			else:
				angle = np.pi
		most_important[k,0] = int((lmin1 + lplus1)*np.sin(angle))
		most_important[k,1] = int(k) 
		most_important_sorted = sorted(most_important,key=lambda x: x[0], reverse = True)
		most_important_sorted = np.array(most_important_sorted)
	return most_important_sorted


def chooseNumFeatures(old_features, features, num_features,contour,last, n, add_thresh, remove_thresh, count):
	index = 0
	count = count + 1
	if count > 10:
		return features
	print('features/addthresh/removethresh: ', features.shape[0], add_thresh, remove_thresh, last)
	if features.shape[0] == num_features:
		return features
	if features.shape[0] < num_features:
		#if last == 1: 
			#add_thresh = add_thresh /2 
			#add_thresh = add_thresh + 0.1
		if last == -1: 
			remove_thresh = remove_thresh - 0.1
	elif features.shape[0] > num_features:
		#if last == 1: 
			#add_thresh = add_thresh - 0.1
		if last == -1: 
			#add_thresh = add_thresh/2
			remove_thresh = remove_thresh + 0.25
	last = -1*last 
	features = addFeatures(index,old_features,contour,n,add_thresh)
	features = removeMidpoints(index,old_features,contour,n,remove_thresh)
	features = chooseNumFeatures(old_features, features, num_features, contour, last, n, add_thresh, remove_thresh,count)
	return features 


if __name__ == '__main__': main()
