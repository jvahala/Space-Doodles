
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
import pydl.pydlutils.spheregroup.spherematch as spherematch
from scipy import spatial

def main():
    
    plt.close("all")    
    
    star_tab = StarTable()

    featdata = [[290,116],[398,203],[290,288],[182,202]]
    
    featpoints = []
    
    for data in featdata:
        featpoints.append(Point(data))
    
    featset = SetOfPoints()
    
    featset.append(featpoints)

    num_tries = 50
    
    mag_constant = .25
    
    searches = []
    searchscores = []
    
    for i in range(num_tries):
        searchdata = Search(star_tab,featset)
        searches.append(searchdata)
        searchdata.evaluate(mag_constant)
        searchscores.append(searchdata.score)
        
    bestsearchID = np.argmin(searchscores)
    
    bestsearch = searches[bestsearchID]
            
    PlotEverything(bestsearch)
    
    print('Average mag is:',bestsearch.avgmag)
    print('Score is:',bestsearch.score)
    
def Search(star_tab, featset):

    # Pick random index of star to search around.
    center = np.random.randint(star_tab.num_stars)
    #center = 1369
    print(center)
    # Get subtable of stars near center star.
    search_radius = 20
    first_search_subtable = star_tab.ClosestStars(center,search_radius)

    # Convert from spherical to cartesian using mollweide projection
    first_search_subset = first_search_subtable.MollProject()
    
    # Pick 3 random feature points of feature set, get angles
    feat_subindices = np.random.choice(featset.length,3,replace=False)
    featsub = featset.GetSubset(indices = feat_subindices)
    
    #match = featsub.RandomSearch(star_subset)
    first_match_subset = ClusterSearch(featsub, first_search_subset)

    first_match_indices = []
    for star in first_match_subset.points:
        first_match_indices.append(star.index)
    
    # Get procrustes transformation, apply to all feature points
    [R,T,scale] = featset.Procrustes(first_match_subset, selfindices = feat_subindices)
    featprime = featset.Transform(R,T,scale)
    
    # Find center of transformed feature points
    center = np.mean(featprime.matrix,axis=0)
    #turn center into a set of points so GetClosestStar can take it
    centerset = SetOfPoints([Point(center)])
    centerstar_index = first_search_subset.GetClosestStar(centerset)[0]
    
    #get new table of stars around center star
    # twice as big as original search radius
    second_search_subtable = star_tab.ClosestStars(centerstar_index,search_radius*2)

    #convert from spherical to mollweide projection
    second_search_subset = second_search_subtable.MollProject()
    
    second_match_indices = second_search_subset.LookUp(first_match_indices)
    
    second_match_subset = second_search_subset.GetSubset(indices = second_match_indices)
    
    [R,T,scale] = featset.Procrustes(second_match_subset, selfindices = feat_subindices)
    
    second_featprime = featset.Transform(R,T,scale)
        
    final_match_indices = second_search_subset.GetClosestStar(second_featprime)
    
    final_match_subset_indices = second_search_subset.LookUp(final_match_indices)
    
    final_match = second_search_subset.GetSubset(final_match_subset_indices)    
    
    #evaluate final match:
    [R,T,scale] = featset.Procrustes(final_match, range(final_match.length))
    final_featprime = featset.Transform(R,T,scale)
        
    searchdata = SearchData(featset, feat_subindices, first_search_subset, first_match_indices, second_search_subset, final_featprime, final_match, final_match_subset_indices)
    
    return searchdata
    
def PlotEverything(searchdata):
    # Plot everything...    
    
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
    
    # Plot first search set
    
    

    lbound, ubound = GetBounds(searchdata.first_search_subset.matrix)
    
    matches = searchdata.first_search_subset.LookUp(searchdata.first_match_indices)

    plt.figure()
    for i in range(searchdata.first_search_subset.length):
        if i in matches:
            plt.scatter(searchdata.first_search_subset.matrix[i,0],searchdata.first_search_subset.matrix[i,1],s=50,c='r',alpha = 1-searchdata.first_search_subset.mags[i])
        else:
            plt.scatter(searchdata.first_search_subset.matrix[i,0],searchdata.first_search_subset.matrix[i,1],c=[0,0,0],alpha = 1-searchdata.first_search_subset.mags[i])
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('First search space, triangle match in red')
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
    
    
    # final match connected
    
    finalmatchfull = np.vstack((searchdata.final_match.matrix,searchdata.final_match.matrix[0]))
    lbound, ubound = GetBounds(searchdata.final_match.matrix)        
    plt.figure()
    plt.plot(finalmatchfull[:,0],finalmatchfull[:,1])        
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.title('Match Stars Connected')
    plt.show()
    
    

def GetBounds(data):
    
    lbound = min(min(data[:,0]),min(data[:,1]))
    ubound = max(max(data[:,0]),max(data[:,1]))
    
    return lbound, ubound
    
    
class SearchData:
    def __init__(self, featset, feat_subindices, first_search_subset, first_match_indices, second_search_subset, final_featprime, final_match, final_match_subset_indices):
        self.featset = featset
        self.feat_subindices = feat_subindices
        self.first_search_subset = first_search_subset
        self.first_match_indices = first_match_indices
        self.second_search_subset = second_search_subset
        self.final_featprime = final_featprime
        self.final_match = final_match
        self.final_match_subset_indices = final_match_subset_indices

    def evaluate(self, mag_constant = .05):
    
        score = 0
        
        for i in range(self.final_featprime.length):
            score += np.linalg.norm(self.final_featprime.points[i].xy-self.final_match.points[i].xy) + mag_constant*self.final_match.points[i].mag
        
        final_mags = []
        for point in self.final_match.points:
            final_mags.append(point.mag)
            
        self.score = score
        self.avgmag = np.mean(final_mags)


class StarTable:
    def __init__(self, file = 'hyg_catalog.fits'):
        self.tab = table.Table.read(file, format = 'fits')
        # convert hours to degrees
        self.tab['RA'] = self.tab['RA']*15
        self.num_stars = len(self.tab)
        
        # normalize magnitudes so they are 0-1 exclusive
        maxmag = max(self.tab['Mag'])
        minmag = min(self.tab['Mag'])
        self.tab['Mag'] = (self.tab['Mag']-minmag)/(maxmag-minmag+.01)+10**-6
        
    def ClosestStars(self, center_index, radius):
        """
        Returns indices of stars that is
        within radius of the center_index star
        """
        
        tRA = self.tab['RA']    
        tDec = self.tab['Dec']
        cRA = np.array([self.tab['RA'][center_index]])
        cDec = np.array([self.tab['Dec'][center_index]])
        
        indices = spherematch(tRA, tDec, cRA, cDec, maxmatch = 0, matchlength = radius)[0]
        
        return SubTable(self,indices)
        
        
class SubTable(StarTable):
    def __init__(self,supertable,indices):
        self.tab = supertable.tab[indices]
        self.indices = indices
        self.num_stars = len(self.tab)

    def MollProject(self, center_index = 0):
        """
        Returns set of xy coordinates centered at center_index?
        Implements conversion given at:
        https://en.wikipedia.org/wiki/Mollweide_projection#Mathematical_formulation
        """
    
        c = self.tab['RA'][center_index]
        
        if c-90<0 or c+90>360:
            self.tab['RA'] = (self.tab['RA']+ 180)%360
    
        c_ra = np.deg2rad(self.tab['RA'][center_index])
        c_dec = np.deg2rad(self.tab['Dec'][center_index])
        
        #c_ra = (c_ra - np.pi)

        #initialize outputs        
        xy = np.zeros((self.num_stars,2))     
        stars = []
        
        for i in range(self.num_stars):
            #convert hours/degrees to radians
            ra = np.deg2rad(self.tab['RA'][i])
            dec = np.deg2rad(self.tab['Dec'][i]) - c_dec
            
            #ra = (ra + np.pi)
            #dec = (dec + np.pi)%(np.pi)-np.pi/2
            
            #longitude = RA = lambda ?
            #latitude = dec = phi ?
            
            t_0 = dec
            epsilon = 10**-6
            
            #iterate to find theta
            error = 1+epsilon        
            while error > epsilon:
                t_1 = t_0 - (2*t_0+np.sin(2*t_0)-np.pi*np.sin(dec))/(2+2*np.cos(2*t_0))
                error = np.abs(t_1 - t_0)
                t_0 = t_1
            
            xy[i,0] = 2*np.sqrt(2)*(ra-c_ra)*np.cos(t_0)/np.pi
            xy[i,1] = np.sqrt(2)*np.sin(t_0)
            
            #create star object with parameters
            #print(subtab[i])
            stars.append(Star(xy=xy[i], mag=self.tab['Mag'][i], index = self.indices[i]))

        #return a starset of stars in starsxy
        return StarSet(stars)
        
class SetOfPoints:
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
        get angles between vertices specified by
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
        
    def GetSubset(self, indices = None):
        if indices is None:
            subset = np.random.choice(bright_points, subsize, replace=False)
        else:
            subset = [self.points[i] for i in indices]
            
        return SetOfPoints(points = subset)
            
            
    def Procrustes(self, target, selfindices = [0,1,2]):
        '''
        Finds the matrix that ideally maps points of self to points of target
        For self, chooses points indexed by "selfindices" (defaults
        to first 3 points)
        '''
        
        A = self.matrix[selfindices,0:2]
        B = target.matrix[:,0:2]
        
        #get mean and center matrices around origin
        meanA = A.mean(0)
        meanB = B.mean(0)
        A0 = A-meanA
        B0 = B-meanB
        
        #get frobenius norms
        Anorm = np.sqrt(np.sum(A0**2))
        Bnorm = np.sqrt(np.sum(B0**2))

        A0 /= Anorm
        B0 /= Bnorm

        M = np.dot(B0.T,A0)
        
        [U,s,VT] = np.linalg.svd(M)
        V = VT.T
        
        #get optimal scaling
        scale = Bnorm/Anorm
        
        R = np.dot(V,U.T)
        
        T = meanB - scale*np.dot(meanA,R)
        
        return [R, T, scale]
        
    def Transform(self, R, T, scale):
        '''
        Outputs coordinates in matrix form after applying the given
        procrustes transformation
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
        test = self.indices
        self.matrix = np.array(xys)
        self.mags = np.array(mags)
        self.length = self.matrix.shape[0]
        
    def LookUp(self,indices):
        '''
        Given a set of original startable indices, gets this indices within this starset
        '''
        output = []
        
        test = self.indices
        
        for starindex in indices:
            output.append(np.where(self.indices == starindex)[0][0])
        
        return output
        
    def GetClosestStar(self, setofpoints):
        '''
        Returns starset index of closest star to point.
        '''
        tree = spatial.KDTree(self.matrix)
        closeststars = tree.query(setofpoints.matrix)[1]
        output = []
        for star in closeststars:
            output.append(self.points[star].index)
        return output
        
    def GetSubset(self, indices = None):
        if indices is None:
            subset = np.random.choice(bright_points, subsize, replace=False)
        else:
            subset = [self.points[i] for i in indices]

        return StarSet(subset)
    
def ClusterSearch(featset, starset):
    """
    Does a smarter search than RandomSearch? Finds a good BRIGHT match
    """
    
    # Get angles formed by first 3 points in self
    feat_angles = np.sort(featset.GetAngles())
    print('feature:',feat_angles)
    
    # Sort starset by its brightness (add first column to remember index)
    mat = starset.matrix
    n = starset.length
    stars = np.hstack((np.arange(n).reshape(n,1),mat,starset.mags.reshape(n,1)))
    s_sorted = stars[stars[:,3].argsort()]
    
    #print(s_sorted)

    # Set error tolerance, initialize counter
    epsilon = .01
    num_tries = 0
    brk = False
    best_diff_so_far = epsilon+1
    
    # iterate through all sets of 3 stars in order of brightness
    # until a match is found
    for i in range(2,n):
        for j in range(1,i):
            for k in range(j):
                v = [int(s_sorted[i,0]),int(s_sorted[j,0]),int(s_sorted[k,0])]
                a = np.sort(starset.GetAngles(verts = v))
                num_tries += 1
                #print(num_tries)
                if num_tries%500 == 0:
                    epsilon *= 1.1
                    if best_diff_so_far < epsilon:
                        a = best_so_far
                        v = best_v_so_far
                        brk = True
                        break
                diff = np.linalg.norm(a - feat_angles)
                if diff < epsilon:
                    brk = True
                    break
                if diff < best_diff_so_far:
                    best_so_far = a
                    best_diff_so_far = diff
                    best_v_so_far = v
            if brk is True:
                break
        if brk is True:
            break
            
    print('feat', feat_angles)
    print('match',a)
    print('verts',np.sort(starset.GetAngles(verts = v)))
    print('mags',starset.mags[v])
    print('done')
            
    # Return starset object of matching points.
    return starset.GetSubset(indices = v)

            
    
if __name__ == '__main__': main()