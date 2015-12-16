
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
import pydl.pydlutils.spheregroup.spherematch as spherematch
from scipy import spatial
import scipy.io as sio

def main():
    plt.close("all")
    
    # Load star data.
    star_tab = StarTable()

    # Specify feature points.
    featdata = [[290,116],[398,203],[290,288],[182,202]]
    
    # Turn feature points into "SetOfPoints" object
    featpoints = []
    for data in featdata:
        featpoints.append(Point(data))
    featset = SetOfPoints()
    featset.append(featpoints)
    
    # Specify weight on star magnitude for cost function
    # <~.5 mostly care about fit
    # >~3 mostly care about magnitude
    mag_constant = 10
    
    searchdata = Search(star_tab,featset, mag_constant)
            
    PlotEverything(searchdata)
    
    print('Average mag is:',searchdata.avgmag)
    print('Score is:',searchdata.score)
    
def Search(star_tab, featset, best_point=None, mag_constant=1):
    '''
    Main search algorithm.
    '''

    # Pick 3 feature points for initial triangle to search for.
    if best_point is None:
        feat_subindices = np.random.choice(featset.length,3,replace=False)
    else:    
        feat_subindices = [best_point-1, best_point, best_point+1]
    featsub = featset.GetSubset(indices = feat_subindices)
    
    # Use cluster search to get a list of possible star trio matches.
    possible_matches = ClusterSearch(featsub)
    
    print('Found',len(possible_matches),'possible matches from initial cluster search')
    
    # Go through each match and find best one.
    clustersearchdata = []
    clustersearchscores = []    
    for possible_match in possible_matches:

        # Turn star trio into "SubTable" object and project onto xy-plane.
        possible_match_subtable = SubTable(star_tab, possible_match)
        possible_match_subset = possible_match_subtable.MollProject()
        
        # Get transformation to best fit feature points.
        [R,T,scale] = featset.Procrustes(possible_match_subset, selfindices = feat_subindices)
        featprime = featset.Transform(R,T,scale)
        
        # Get subset of stars centered around the center of the transformed
        # feature points, project onto xy-plane.
        center = np.mean(featprime.matrix,axis=0)
        centerset = SetOfPoints([Point(center)])
        centerstar_index = possible_match_subset.GetClosestStar(centerset)[0]
        second_search_subtable = star_tab.ClosestStars(centerstar_index,40)
        second_search_subset = second_search_subtable.MollProject()
        
        # Gets indices of the possible match stars for the new subset.
        second_match_indices = second_search_subset.LookUp(possible_match)
        second_match_subset = second_search_subset.GetSubset(indices = second_match_indices)
        
        # Gets transformation to best fit feature points.
        [R,T,scale] = featset.Procrustes(second_match_subset, selfindices = feat_subindices)
        second_featprime = featset.Transform(R,T,scale)
        
        # Gets closest stars to all of the points in transformed set
        # for final match, turn into "SetOfPoints" object.
        final_match_indices = second_search_subset.GetClosestStar(second_featprime)
        final_match_subset_indices = second_search_subset.LookUp(final_match_indices)
        final_match = second_search_subset.GetSubset(final_match_subset_indices)    
        
        # Evaluate final match:
        # Get transformation to best fit all feature points to
        # all stars in match
        [R,T,scale] = featset.Procrustes(final_match, range(final_match.length))
        final_featprime = featset.Transform(R,T,scale)
        
        # Store all data as "SearchData" object to evaluate/plot later
        searchdata = SearchData(featset, feat_subindices, possible_match, second_search_subset, final_featprime, final_match, final_match_subset_indices)
        clustersearchdata.append(searchdata)
        searchdata.evaluate(mag_constant)
        clustersearchscores.append(searchdata.score)
        
    # Get index of search data with lowest score
    bestsearchID = np.argmin(clustersearchscores)
    
    # Return corresponding search data 
    return clustersearchdata[bestsearchID]
    
def PlotEverything(searchdata):
    '''
    A bunch of plots displaying search results.
    '''
    
    # Original feature points connected
    featsetfull = np.vstack((searchdata.featset.matrix,searchdata.featset.matrix[0]))
    lbound, ubound = GetBounds(searchdata.featset.matrix)        
    plt.figure()
    plt.plot(featsetfull[:,0],featsetfull[:,1])        
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Original Feature Points Connected')
    plt.show()
    

    # Original feature points
    lbound, ubound = GetBounds(searchdata.featset.matrix)    
    plt.figure()
    for i in range(searchdata.featset.length):
        if i in searchdata.feat_subindices:
            plt.scatter(searchdata.featset.matrix[i,0],searchdata.featset.matrix[i,1],s=50,c='r')
        else:
            plt.scatter(searchdata.featset.matrix[i,0],searchdata.featset.matrix[i,1])
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Original Feature Points, triangle in red')
    plt.show()

    # Plot second search set
    lbound, ubound = GetBounds(searchdata.second_search_subset.matrix)
    plt.figure()
    for i in range(searchdata.second_search_subset.length):
        if i in searchdata.final_match_subset_indices:
            plt.scatter(searchdata.second_search_subset.matrix[i,0],searchdata.second_search_subset.matrix[i,1],s=50,c='r',alpha = 1-searchdata.second_search_subset.mags[i])
        else:
            plt.scatter(searchdata.second_search_subset.matrix[i,0],searchdata.second_search_subset.matrix[i,1],c=[0,0,0], alpha = 1-searchdata.second_search_subset.mags[i])
    # Plot dotted line connecting stars
    x = searchdata.second_search_subset.matrix[searchdata.final_match_subset_indices,0]
    y = searchdata.second_search_subset.matrix[searchdata.final_match_subset_indices,1]
    x = np.hstack((x,x[0]))
    y = np.hstack((y,y[0]))
    plt.plot(x,y,'b--')
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Second search space, match in red')
    plt.show()
    
    # Final match vs original points
    lbound, ubound = GetBounds(np.vstack((searchdata.final_featprime.matrix,searchdata.final_match.matrix)))
    plt.figure()
    for i in range(searchdata.final_featprime.length):
        plt.scatter(searchdata.final_featprime.matrix[i,0],searchdata.final_featprime.matrix[i,1],s=50,c='b')
        plt.scatter(searchdata.final_match.matrix[i,0],searchdata.final_match.matrix[i,1],s=50,c='r')
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Final match stars in Red, Original Feature Points in Blue')
    plt.show()
    
    # Final match connected
    finalmatchfull = np.vstack((searchdata.final_match.matrix,searchdata.final_match.matrix[0]))
    lbound, ubound = GetBounds(searchdata.final_match.matrix)        
    plt.figure()
    plt.plot(finalmatchfull[:,0],finalmatchfull[:,1])        
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Match Stars Connected')
    plt.show()

def GetBounds(data):
    '''
    A helper function for getting bounds for plots
    '''
    
    lbound = min(min(data[:,0]),min(data[:,1]))
    ubound = max(max(data[:,0]),max(data[:,1]))
    
    return lbound, ubound
    
    
class SearchData:
    '''
    A class for storing search data
    '''
    def __init__(self, featset, feat_subindices, first_match_indices, second_search_subset, final_featprime, final_match, final_match_subset_indices):
        self.featset = featset
        self.feat_subindices = feat_subindices
        self.first_match_indices = first_match_indices
        self.second_search_subset = second_search_subset
        self.final_featprime = final_featprime
        self.final_match = final_match
        self.final_match_subset_indices = final_match_subset_indices

    def evaluate(self, mag_constant = .05):
        '''
        Evaluates the cost function for this SearchData. Adds up:
        Errors and a weighted L1 norm of star magnitudes. Smaller is better!
        Also computes average star magnitude.
        '''
        score = 0 
        for i in range(self.final_featprime.length):
            score += np.linalg.norm(self.final_featprime.points[i].xy-self.final_match.points[i].xy) + mag_constant*self.final_match.points[i].mag
        
        final_mags = []
        for point in self.final_match.points:
            final_mags.append(point.mag)
            
        self.score = score
        self.avgmag = np.mean(final_mags)


class StarTable:
    '''
    Main star data.
    '''
    def __init__(self, file = 'hyg_catalog.fits'):
        self.tab = table.Table.read(file, format = 'fits')
        
        # Assign index to each star.
        self.tab['Index'] = np.arange(len(self.tab))

        # Convert hours to degrees.
        self.tab['RA'] = self.tab['RA']*15
        
        # Get number of stars.
        self.num_stars = len(self.tab)
        
        # Normalize magnitudes so they are 0-1 exclusive
        maxmag = max(self.tab['Mag'])
        minmag = min(self.tab['Mag'])
        self.tab['Mag'] = (self.tab['Mag']-minmag)/(maxmag-minmag+.01)+10**-6
        print('Normalized magnitudes')
        print('Magnitude 3 corresponds to:',(3-minmag)/(maxmag-minmag+.01)+10**-6)
        
    def ClosestStars(self, center_index, radius):
        """
        Returns "SubTable" object of stars that is
        within radius of the center_index star
        """
        
        tRA = self.tab['RA']    
        tDec = self.tab['Dec']
        cRA = np.array([self.tab['RA'][center_index]])
        cDec = np.array([self.tab['Dec'][center_index]])
        
        indices = spherematch(tRA, tDec, cRA, cDec, maxmatch = 0, matchlength = radius)[0]
        
        return SubTable(self,self.tab['Index'][indices])

    def LookUp(self,indices):
        '''
        Given original index, finds index in this table.
        (i.e. if you have a subtable and need original table index)
        '''
        output = []
        
        for starindex in indices:
            output.append(np.where(self.tab['Index'] == starindex)[0][0])
        
        return output
        
class SubTable(StarTable):
    """
    An object for storing a subtable of stars.
    """
    def __init__(self,supertable,indices):
        self.tab = supertable.tab[indices]
        self.indices = indices
        self.num_stars = len(self.tab)


    def MollProject(self, center_index = 0):
        """
        Returns a "StarSet" object that contains stars with xy-coordinates
        for all stars in this subtable centered at star with center_index
        (defaults to first star in sub_table)
        """
    
        c = self.tab['RA'][center_index]
        
        # If close to edges, shift everything so they are not.
        if c-90<0 or c+90>360:
            self.tab['RA'] = (self.tab['RA']+ 180)%360
    
        # Get center star spherical coordinates.
        c_ra = np.deg2rad(self.tab['RA'][center_index])
        c_dec = np.deg2rad(self.tab['Dec'][center_index])

        xy = np.zeros((self.num_stars,2))     
        stars = []
        
        # Go through each star and project
        for i in range(self.num_stars):
            # Get spherical coordinates, convert hours/degrees to radians, 
            # shift dec to align with center.
            ra = np.deg2rad(self.tab['RA'][i])
            dec = np.deg2rad(self.tab['Dec'][i]) - c_dec
            
            # Mollweide projection algorithm...
            t_0 = dec
            epsilon = 10**-6
            error = 1+epsilon        
            while error > epsilon:
                t_1 = t_0 - (2*t_0+np.sin(2*t_0)-np.pi*np.sin(dec))/(2+2*np.cos(2*t_0))
                error = np.abs(t_1 - t_0)
                t_0 = t_1
            xy[i,0] = 2*np.sqrt(2)*(ra-c_ra)*np.cos(t_0)/np.pi
            xy[i,1] = np.sqrt(2)*np.sin(t_0)
            
            # Create "Star" object with parameters, append to output.
            stars.append(Star(xy=xy[i], mag=self.tab['Mag'][i], index = self.indices[i]))

        # Return a "StarSet" object of stars
        return StarSet(stars)
        
class SetOfPoints:
    '''
    Object for storing feature points and stars.
    '''
    def __init__(self, points = []):
        self.points = []
        self.append(points)

    def UpdateMatrix(self):
        xys = []
        for point in self.points:
            xys.append(point.xy)
        self.matrix = np.array(xys)
        self.length = self.matrix.shape[0]
        
    def append(self,points):
        for point in points:
            self.points.append(point)
        self.UpdateMatrix()
    
    def GetAngles(self, verts = [0,1,2]):
        '''
        Get angles between vertices specified by
        verts (defaults to first three points)
        '''
        d = np.zeros((3))
        ab = self.matrix[verts[0]]-self.matrix[verts[1]]
        ac = self.matrix[verts[0]]-self.matrix[verts[2]]
        cb = self.matrix[verts[1]]-self.matrix[verts[2]]
        d[0] = np.linalg.norm(ab)
        d[1] = np.linalg.norm(ac)
        d[2] = np.linalg.norm(cb)
        
        a = np.zeros((3))
        a[0] = np.arccos(np.dot(ab,ac)/(d[0]*d[1]))
        a[1] = np.arccos(np.dot(ab,cb)/(d[0]*d[2]))
        a[2] = np.arccos(np.dot(ac,cb)/(d[1]*d[2]))
        
        return a    
        
    def GetSubset(self, indices):
        '''
        Returns a "SetOfPoints" object containing points specified by indices.
        '''        
        subset = [self.points[i] for i in indices]
        return SetOfPoints(points = subset)
            
    def Procrustes(self, target, selfindices = [0,1,2]):
        '''
        Finds the transformation that best fits points of self (specified by
        self indices) to points of target.
        Returns [R,T,scale] where:
        R is rotation matrix
        T is translation matrix
        scale is scaling factor
        '''
        
        A = self.matrix[selfindices,0:2]
        B = target.matrix[:,0:2]
        
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
        V = VT.T
        R = np.dot(V,U.T)
        
        # Get scaling factor
        scale = Bnorm/Anorm

        # Get translation        
        T = meanB - scale*np.dot(meanA,R)
        
        return [R, T, scale]
        
    def Transform(self, R, T, scale):
        '''
        Outputs "SetOfPoints" object after
        applying given Procrustes transformation.
        '''
        transformed = scale*np.dot(self.matrix,R)+T
        output = []
        for point in transformed:
            output.append(Point(point))
        return SetOfPoints(output)
        
class Point:
    def __init__(self, xy=None):
        self.xy = xy

class Star:
    def __init__(self, xy = None, mag = None, index=None):
        index = index
        self.index = index
        self.xy = xy
        self.mag = mag
    
class StarSet(SetOfPoints):
    def __init__(self,points):
        self.points = []
        self.append(points)
        
    def append(self,points):
        for point in points:
            self.points.append(point)
        self.UpdateData()

    def UpdateData(self):
        indices = []
        mags = []
        xys = []
        for star in self.points:
            indices.append(star.index)
            xys.append(star.xy)
            mags.append(star.mag)
        self.indices = np.array(indices)
        self.matrix = np.array(xys)
        self.mags = np.array(mags)
        self.length = self.matrix.shape[0]
        
    def LookUp(self,indices):
        '''
        Given a set of original startable indices,
        gets this indices within this starset.
        '''
        output = []        
        for starindex in indices:
            output.append(np.where(self.indices == starindex)[0][0])        
        return output
        
    def GetClosestStar(self, setofpoints):
        '''
        Does a tree search to get a list of original starset indices
        that are closest to each point in setofpoints.
        '''
        tree = spatial.KDTree(self.matrix)
        closeststars = tree.query(setofpoints.matrix)[1]
        output = []
        for star in closeststars:
            output.append(self.points[star].index)
        return output
        
    def GetSubset(self, indices = None):
        subset = [self.points[i] for i in indices]
        return StarSet(subset)
        
def ClusterSearch(featset):
    '''
    Uses cluster data to find trios of stars that are possible matches for
    triangle formed by points in featset. Returns an array of lists of 3.
    '''
    
    # Load cluster data.
    clustercenters = sio.loadmat('cluster_data/clustercenters.mat')['clustercenter']
    clusterlabels = sio.loadmat('cluster_data/starlabels.mat')['label']
    startrios = sio.loadmat('cluster_data/starlabels.mat')['startrio']
    
    # Get and sort feature angles.
    feat_angles = np.sort(featset.GetAngles())
    
    # Do a tree search to find 3 closest cluster centers to feat_angles.
    tree = spatial.KDTree(clustercenters[:,0:2])
    bestcluster = tree.query(feat_angles[0:2],k=3)[1]
    
    # Get all trios of stars that are in the 3 clusters.
    best1 = startrios[np.argwhere(clusterlabels == bestcluster[0])[:,0]]
    best2 = startrios[np.argwhere(clusterlabels == bestcluster[1])[:,0]]
    best3 = startrios[np.argwhere(clusterlabels == bestcluster[2])[:,0]]
    
    # Stack into one list and return.
    possible_matches = np.vstack((best1,best2,best3))    
    return possible_matches
            
    
if __name__ == '__main__': main()