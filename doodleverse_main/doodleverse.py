
import numpy as np
import search
import feature_extract as f_e
import matplotlib.pyplot as plt
import cv2

def main():
    
    plt.close("all")    
    
    [feature_points, bestpoint] = get_features()
    
    print(feature_points,bestpoint)
    
    star_tab = search.StarTable()

    featpoints = []

    for i in range(feature_points.shape[0]-1):
        featpoints.append(search.Point(feature_points[i]))   
    
    featset = search.FeatureSet(data=featpoints)
    
    search.Search(star_tab, featset)

def get_features():    
    
	 #import image as black/white 
	 #example shapes: shape1.png (odd,no int), shape2.png (odd,no int), shape3.png (rounded, no int)
    raw_img, image, contours, hierarchy = f_e.importImage('shape5.png')
    cnt = contours[1] #contour zero is border, contour 1 is outermost contour, ...etc


    #create grayscale (uint8) all white image to draw features onto
    draw_img = f_e.drawImage(raw_img,cnt)

    #grab extreme points of contour
    extrema = f_e.getExtrema(cnt)

    #use harris corner detector 
    corners = f_e.getCorners(draw_img,10,0.1,50)
    features = f_e.orderFeatures(cnt,extrema,corners)
    cnt = f_e.cleanContour(cnt)

    #consolidate features
    add_threshold = 0.01 #any normalized Error between features must be greater than this value for a new point to be added
    remove_threshold = 0.01 #larger values mean less features will make it through
    n = 5#number of divisions for determining normalized error
    index = 0 #default starting index 
    #num_features = 7
    count = 0
    new_features = features
    new_features = f_e.addFeatures(index,new_features,cnt,n,add_threshold)
    new_features = f_e.addFeatures(index,new_features,cnt,n,add_threshold*0.1)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n,remove_threshold)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n,remove_threshold*10)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n,remove_threshold*30)
    new_features = f_e.removeMidpoints(index,new_features,cnt,n,remove_threshold*50)
    new_features[:,[0, 1]] = new_features[:,[1, 0]]
    #new_features = chooseNumFeatures(features, features, num_features, cnt, -1, n, add_threshold, remove_threshold, count)
    #print('Original/New/difference',features.shape[0],'/',new_features.shape[0],'/',new_features.shape[0]-features.shape[0])
    best_features_sorted = f_e.findKeyFeatures(new_features)

    #print(best_features_sorted)
    
    #print(new_features)

    return [new_features, best_features_sorted[0,1]]
    #plot feature points
    plt.figure(1)
    plt.imshow(draw_img.squeeze(),cmap='Greys')
    plt.scatter(corners[:,0],corners[:,1],s=20,c='b',marker='x')
    plt.plot(features[:,0],features[:,1])
    plt.plot(new_features[:,0],new_features[:,1],'r-')
    plt.title('Feature Map')
    plt.show()

if __name__ == '__main__': main()