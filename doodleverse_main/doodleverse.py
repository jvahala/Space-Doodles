
import numpy as np
import search
import feature_extract as f_e
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import cv2

def main():
    
    plt.close("all")    
    
    [feature_points, bestpoint] = get_features('shape1.png')
    
    print(feature_points,bestpoint)
    
    star_tab = search.StarTable()

    featpoints = []
    
    for point in feature_points:
        featpoints.append(search.Point(point))
    
    featset = search.SetOfPoints()
    
    featset.append(featpoints)
    
    num_tries = 1
    
    mag_constant = .5
    
    searches = []
    searchscores = []
    
    for i in range(num_tries):
        searchdata = Search(star_tab,featset, mag_constant)
        searches.append(searchdata)
        searchdata.evaluate(mag_constant)
        searchscores.append(searchdata.score)
        
    bestsearchID = np.argmin(searchscores)
    
    bestsearch = searches[bestsearchID]
            
    PlotEverything(bestsearch)
    
    print('Average mag is:',bestsearch.avgmag)
    print('Score is:',bestsearch.score)

def get_features(image_name):    
    
	 #import image as black/white 
	 #example shapes: shape1.png (odd,no int), shape2.png (odd,no int), shape3.png (rounded, no int)
    raw_img, image, contours, hierarchy = f_e.importImage(image_name)
    img_main = mpimg.imread(image_name)
    cnt1 = contours[1] #contour zero is border, contour 1 is outermost contour, ...etc
    cnt = f_e.cleanContour(cnt1)

    #create grayscale (uint8) all white image to draw features onto
    draw_img = f_e.drawImage(raw_img,cnt)

    #grab extreme points of contour
    extrema = f_e.getExtrema(cnt)

    #use harris corner detector 
    corners = f_e.getCorners(draw_img,10,0.1,50)
    features = f_e.orderFeatures(cnt,extrema,corners)


    #consolidate features
    add_threshold = 0.01 #smaller values add more points (0.01 default)
    remove_threshold = 0.01 #larger values mean less points (0.01 default)
    clumpThresh = +70 #set negative to make it based on the 1/4 the best feature value, otherwise 70+ is a good value, higher values mean less points
    n = 20 #number of divisions for determining normalized error (5 default)
    index = 0 #default starting index (0 default)
    count = 0

    #find feature location on contour
    new_features = f_e.featuresOnContour(features, cnt)

    #add a bunch of features 
    new_features = f_e.addFeatures(index,new_features,cnt,n,add_threshold)
    new_features = f_e.addFeatures(index,new_features,cnt,n,add_threshold*0.1)

    #remove them slowly
    new_features = f_e.removeMidpoints(index,new_features,cnt,n,remove_threshold)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n+10,remove_threshold*20)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n+20,remove_threshold*50)

    #finalize features
    best_features_sorted = f_e.findKeyFeatures(new_features)
    new_features, best_features_sorted = f_e.removeClumps(new_features,best_features_sorted,clumpThresh)

    best_features = new_features[:,0:2]
    best_features.shape = (best_features.shape[0],2)
    #best_features[:,[0, 1]] = best_features[:,[1, 0]] #switches columns 

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
    best_triangle = new_features[(best_index-1):(best_index+2),:]
    plt.scatter(best_triangle[:,0],best_triangle[:,1],s=30,facecolors='none',edgecolors='g',marker='o')
    plt.title('(d) Optimized Features', fontsize=10)
    plt.axis('image')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.show()

    return [best_features, best_features_sorted[0,1]]




if __name__ == '__main__': main()