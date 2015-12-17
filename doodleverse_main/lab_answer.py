import numpy as np
import scipy.io as sio
from scipy import spatial

def main():
    
    plt.close('all')
    
    # Question 3
    featpoints = np.array([[290,116],[398,203],[290,288],[182,202]])
    
    # Pick 3 points:
    feat_tri = featpoints[0:3,:]
    
    # Compute angles:    
    feat_angles = GetAngles(feat_tri)
    
    # Load cluster centers
    clustercenters = sio.loadmat('cluster_data/clustercenters.mat')['clustercenter']    
    
    # Build tree to find closest cluster(s), find it
    clustertree = spatial.KDTree(clustercenters)
    bestcluster = clustertree.query(feat_angles, k=3)[1]
    
    # Find all triangles in cluster
    clusterlabels = sio.loadmat('cluster_data/starlabels.mat')['label']
    triangles = sio.loadmat('cluster_data/starlabels.mat')['startrio']
    match_IDs = np.argwhere(clusterlabels==bestcluster)[:,0]
    match_triangles = triangles[match_IDs]
    
    # Find which of these triangles are in our subset    
    starmat = sio.loadmat('stars.mat')
    starIDs = set([starID for starID in starmat['Index'][0]])
    intersection = []
    for triangle in match_triangles:    
        if triangle[0] in starIDs and triangle[1] in starIDs and triangle[2] in starIDs:
            intersection.append(triangle)
    
    # Get mollweide projection of our starset
    starsxy = []
    for i in range(len(starmat['RA'][0])):
        xy = Project(starmat['RA'][0][i],starmat['Dec'][0][i],263,-37)
        xy.append(starmat['Index'][0][i])
        starsxy.append(xy)
    starsxy = np.array(starsxy)
        
    # Question 4
        
    scores = []    
    matches = []

    # Go through possible matches
    for possible_match in intersection:
        match_xy = []
        match_mags = []
        
        # Get data for each star
        for star in possible_match:
            starID = np.argwhere(starsxy[:,2]==star)[0][0]
            match_mags.append(starmat['Mag'][0][starID])
            match_xy.append(list(starsxy[starID,0:2]))
        
        match_xy = np.array(match_xy)

        # Procrustes transformation to find last star        
        R, T, scale = GetProcrustes(feat_tri,match_xy)
        transformed_feat = scale*np.dot(featpoints,R)+T
        
        # Question 5
        # Get star closest to last point
        startree = spatial.KDTree(starsxy[:,0:2])
        laststarID = startree.query(transformed_feat[3])[1]
        match_xy = np.vstack((match_xy,starsxy[laststarID,0:2]))
        fullmatch = list(possible_match)        
        fullmatch.append(starmat['Index'][0][laststarID])
        matches.append(fullmatch)
        
        # Get magnitude of last star
        match_mags.append(starmat['Mag'][0][laststarID])
        
        # Evaluate final match... get Procrustes transformation again
        R, T, scale = GetProcrustes(featpoints,match_xy)        
        transformed_feat = scale*np.dot(featpoints,R)+T
        
        # How much to weight magnitudes
        mag_constant = .1
        
        # Cost function
        scores.append(np.linalg.norm(transformed_feat-match_xy)+mag_constant*sum(match_mags))        
    
    bestID = np.argmin(scores)
    print('Best match were stars:',matches[bestID])
    
def GetAngles(points):
    
        d = np.zeros((3))
        ab = points[0]-points[1]
        ac = points[0]-points[2]
        cb = points[1]-points[2]
        d[0] = np.linalg.norm(ab)
        d[1] = np.linalg.norm(ac)
        d[2] = np.linalg.norm(cb)
        
        a = np.zeros((3))
        a[0] = np.arccos(np.dot(ab,ac)/(d[0]*d[1]))
        a[1] = np.arccos(np.dot(ab,cb)/(d[0]*d[2]))
        a[2] = np.arccos(np.dot(ac,cb)/(d[1]*d[2]))
        
        return sorted(a)
        
def GetProcrustes(A,B):
    '''
    This is transpose of version proposed in lab, because
    more compatible with how python handles lists
    '''
    
    # Get mean and center matrices around origin
    meanA = A.mean(0)
    meanB = B.mean(0)
    A0 = A-meanA
    B0 = B-meanB
    
    # Get Frobenius norms and normalize
    Anorm = np.sqrt(np.sum(A0**2))
    Bnorm = np.sqrt(np.sum(B0**2))
    A0 /= Anorm
    B0 /= Bnorm

    # Get Procrustes rotation matrix.
    M = np.dot(B0.T,A0)
    [U,s,VT] = np.linalg.svd(M)
    R = np.dot(VT.T,U.T)
    
    # Get scaling factor
    scale = Bnorm/Anorm

    # Get translation        
    T = meanB - scale*np.dot(meanA,R)
    
    return R, T, scale
        
def Project(ra, dec, c_ra, c_dec):
    '''
    Finds the Mollweide projection coordinates (x,y) for the point (ra,dec) around
    point (c_ra,c_dec).
    '''
    # Convert to degrees
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)    
    c_ra = np.deg2rad(c_ra)
    c_dec = np.deg2rad(c_dec)
    
    # Find theta
    theta_0 = dec-c_dec
    epsilon = 10**-6
    error = 1+epsilon
    
    while error > epsilon:
        m = (2*theta_0+np.sin(2*theta_0)-np.pi*np.sin(dec-c_dec))/(2+2*np.cos(2*theta_0))
        theta_1 = theta_0-m
        error = np.abs(theta_1-theta_0)
        theta_0 = theta_1
        
    # Compute (x,y) coordinates
    x = 2*np.sqrt(2)*(ra-c_ra)*np.cos(theta_0)/np.pi
    y = np.sqrt(2)*np.sin(theta_0)

    return [x,y]
    

if __name__ == '__main__': main()