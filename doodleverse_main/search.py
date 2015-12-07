
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
import pydl.pydlutils.spheregroup.spherematch as spherematch
from scipy import spatial
from astropy.io import ascii
from astropy.io import fits
import scipy.io as sio

class StarTable:
    def __init__(self, file = 'hyg_catalog.fits'):
        self.tab = table.Table.read(file, format = 'fits')
        # convert hours to degrees
        self.tab['RA'] = self.tab['RA']*15
        
    def ClosestStars(self, center_index, radius):
        """
        Returns indices stars that is
        within radius of the center_index star
        """
        
        tRA = self.tab['RA']    
        tDec = self.tab['Dec']
        cRA = np.array([self.tab['RA'][center_index]])
        cDec = np.array([self.tab['Dec'][center_index]])
        
        # Shift stars over if near edge (spherematch doesn't wrap)
        if cRA-radius < 0 or cRA+radius > 360:
            tRA = (tRA + 180)%360
            cRA = (cRA + 180)%360
        if cDec-radius < -90 or cDec+radius > 90:
            cDec = (cDec+180)%180-90
            tDec = (tDec+180)%180-90
        
        m = spherematch(tRA, tDec, cRA, cDec, maxmatch = 0, matchlength = radius)

        self.subtable_indices = m[0]
        return m[0]
        
    def MollProject(self, subtab, center_index = 0):
        """
        Returns set of xy coordinates centered at center_index?
        Implements conversion given at:
        https://en.wikipedia.org/wiki/Mollweide_projection#Mathematical_formulation
        """
    
        c_ra = np.deg2rad(subtab['RA'][center_index])
        c_dec = np.deg2rad(subtab['Dec'][center_index])

        #initialize outputs        
        xy_pos = np.zeros((len(subtab),2))        
        starsxy = []
        
        for i in range(len(subtab)):
            #convert hours/degrees to radians
            ra = np.deg2rad(subtab['RA'][i])
            dec = np.deg2rad(subtab['Dec'][i]) - c_dec
            
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
            
            xy_pos[i,0] = 2*np.sqrt(2)*(ra-c_ra)*np.cos(t_0)/np.pi
            xy_pos[i,1] = np.sqrt(2)*np.sin(t_0)
            
            #create star object with parameters
            #print(subtab[i])
            starsxy.append(Star(pos=xy_pos[i],bright=subtab['Mag'][i],star_index = self.subtable_indices[i]))

        #return a starset of stars in starsxy
        return StarSet(data = starsxy)
        
class Point:
    def __init__(self, pos = None):
        '''random position if not specified'''
        self.pos = np.zeros((2))
        if pos is None:
            self.pos = np.random.rand(2)
        else: self.pos = pos
        
class SetOfPoints:    
    def GetAngles(self, verts = [0,1,2]):
        '''
        get angles between vertices specified by
        verts (defaults to first three points)
        '''
        d = np.zeros((3))
        ab = self.points[verts[0]].pos-self.points[verts[1]].pos
        ac = self.points[verts[0]].pos-self.points[verts[2]].pos
        cb = self.points[verts[1]].pos-self.points[verts[2]].pos
        d[0] = np.linalg.norm(ab)
        d[1] = np.linalg.norm(ac)
        d[2] = np.linalg.norm(cb)
        
        a = np.zeros((3))
        a[0] = np.arccos(np.dot(ab,ac)/(d[0]*d[1]))
        a[1] = np.arccos(np.dot(ab,cb)/(d[0]*d[2]))
        a[2] = np.arccos(np.dot(ac,cb)/(d[1]*d[2]))
        
        return a
        
    def GetMatrix(self):
        '''
        returns position data for all points in set in the form of a
        nx2 matrix (nx3 if matrix of stars, 3rd col being brightness)
        '''
        return np.array([self.points[i].pos for i in range(len(self.points))])
        
    def GetLength(self):
        return len(self.points)
        
        
    def GetSubset(self, subsize = 3, indices = None, greatestMag = None):
        '''
        generates subset of StarSet, either random or specified by indices
        '''
        if greatestMag is not None:
            bright_points = [self.points[i] for i in len(self.points) if self.points[i].bright < greatestMag][0]
        else:
            bright_points = self.points
        
        if indices is None:
            subset = np.random.choice(bright_points, subsize, replace=False)
        else:
            subset = [bright_points[i] for i in indices]

        if type(self) is StarSet:
            return StarSet(data = subset)
        elif type(self) is FeatureSet:
            return FeatureSet(data = subset)
        else:
            return subset
            
    def ClusterSearch(self, starset):
        """
        Does a smarter search than RandomSearch? Finds a good BRIGHT match
        """
        
        # Get angles formed by first 3 points in self
        feat_angles = np.sort(self.GetAngles())
        print('feature:',feat_angles)
        
        # Sort starset by its brightness (add first column to remember index)
        mat = starset.GetMatrix()
        n = starset.GetLength()
        stars = np.hstack((np.arange(n).reshape(n,1),mat))
        s_sorted = stars[stars[:,3].argsort()]
        
        print(s_sorted)

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
        print('mags',mat[v,2])

        print('done')
                
        # Return starset object of matching points.
        return starset.GetSubset(indices = v)
            
    def RandomSearch(self, starset):
        '''
        Randomly searches starset for a subset that matches the self's shape
        Finds angles that match.
        Only works on first three points of self.
        '''
        # Get angles formed by first 3 points in self
        feat_angles = self.GetAngles()
        
        # Set error tolerance, initialize counter
        epsilon = .05
        num_tries = 0
        
        while True:

            #bright_subset = starset.GetSubset(greatestMag = 10)
            #sub_verts = np.random.choice(bright_subset.GetLength(),3,replace=False)
            #sub_angles = bright_subset.GetAngles(verts = sub_verts)
            
            sub_verts = np.random.choice(starset.GetLength(),3,replace=False)
            sub_angles = starset.GetAngles(verts = sub_verts)

            error = np.linalg.norm(sub_angles - feat_angles)
            if error < epsilon: break
            sub_angles = sub_angles[[1,2,0]]
            error = np.linalg.norm(sub_angles - feat_angles)
            if error < epsilon: break
            sub_angles = sub_angles[[1,2,0]]
            error = np.linalg.norm(sub_angles - feat_angles)
            if error < epsilon: break
            num_tries += 1
            if num_tries%3000 == 0: epsilon = 1.5*epsilon        
            #print(num_tries)
    
        # Return starset object of matching points.
        return starset.GetSubset(indices = sub_verts)

            
    def Procrustes(self, target, selfindices = [0,1,2]):
        '''
        Finds the matrix that ideally maps points of self to points of target
        For self, chooses points indexed by "selfindices" (defaults
        to first 3 points)
        '''
        
        A = self.GetMatrix()[selfindices,0:2]
        B = target.GetMatrix()[:,0:2]
        
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
        return scale*np.dot(self.GetMatrix(),R)+T
        
    
class FeatureSet(SetOfPoints):
    def __init__(self, numPoints = 3, data = None):
        '''generates a random set of points or assigns points in data array
        or if data is list of points, makes them a FeatureSet'''
        self.points = []
    
        if data is None:
            for i in range(numPoints):
                self.points.append(Point())
        elif type(data[0] is Point):
            for i in range(len(data)):
                self.points.append(data[i])
        else:
            for i in range(data.shape[0]):
                self.points.append(Point(pos = data[i,:]))
        
class Star(Point):
    def __init__(self, pos = None, bright = None, spread = 1, star_index=None):
        '''generates a random star or a star with given position/brightness'''
        self.index = star_index
        self.pos = np.zeros((2))        
        if pos is None:
            self.pos = np.random.rand(2)*spread
        else: self.pos = pos
        if bright is None:
            self.bright = np.random.rand()
        else: self.bright = bright
    
class StarSet(SetOfPoints):
    def __init__(self, numStars = 100, spread = 10, data = None):
        '''either generates random set of stars or generates a container of stars'''
        self.points = []
        if data is None:
            for i in range(numStars):
                self.points.append(Star(spread = spread))
        else:
            for i in data:
                self.points.append(i)

        
    def GetMatrix(self):
        '''returns position/mag data for all points in set in the form of a nx3 matrix'''
        p = np.array([self.points[i].pos for i in range(len(self.points))])
        m = np.array([self.points[i].bright for i in range(len(self.points))])
        
        m = m.reshape(m.shape[0],1)

        return np.hstack((p,m))
        
    def GetClosestStar(self, point):
        '''
        Returns starset index of closest star to point.
        '''
        tree = spatial.KDTree(self.GetMatrix()[:,0:2])
        return tree.query(point)[1]
        
def main():
    
    plt.close("all")    
    
    star_tab = StarTable()

    featdata = None
    featset = FeatureSet(numPoints=5, data = featdata)

    Search(star_tab,featset)
    
def Search(star_tab, featset):

    # Pick random index of star to search around.
    center = np.random.randint(len(star_tab.tab))
    # Get subtable of stars near center star.
    search_radius = 20
    m = star_tab.ClosestStars(center,search_radius)
    closest_subtable = star_tab.tab[m]

    # Convert from spherical to cartesian using mollweide projection
    star_subset = star_tab.MollProject(closest_subtable)
    
    
    cs = closest_subtable    
    
    csa[:,0] = np.array(cs['RA'])
    csa[:,1] = np.array(cs['Dec'])
    csa[:,2] = np.array(cs['Mag'])
    
    sio.savemat('sub-table.mat',{'RA':cs['RA'], 'Dec':cs['Dec'], 'Mag':cs['Mag']})
    
    
    # Pick 3 random feature points of feature set, get angles
    subindices = np.random.choice(featset.GetLength(),3,replace=False)
    featsub = featset.GetSubset(subsize = 3, indices = subindices)
    
    #match = featsub.RandomSearch(star_subset)
    
    match = featsub.ClusterSearch(star_subset)    
        
    # Get procrustes transformation, apply to all feature points
    [R,T,scale] = featset.Procrustes(match, selfindices = subindices)
    featMprime = featset.Transform(R,T,scale)    
    
    # Find center of transformed feature points
    center = np.mean(featMprime,axis=0)    
    centerstar = star_subset.GetClosestStar(center)
    # Get original star table index of star closest
    centerstar_index = star_subset.points[centerstar].index

    #get new table of stars around center star
    # twice as big as original search radius
    m = star_tab.ClosestStars(centerstar_index,search_radius*2)
    closest_subtable = star_tab.tab[m]

    #convert from spherical to mollweide projection
    search_subset = star_tab.MollProject(closest_subtable)
    searchM = search_subset.GetMatrix()

    # Get closest points to feature points in new search subset
    """MAKE MORE EFFICIENT BY ALLOWING GETCLOSESTSTAR TO DO MULTIPLE POINTS AT ONCE"""
    matchMprime = []
    for i in range(featset.GetLength()):
        closest_star = search_subset.GetClosestStar(featMprime[i,:])
        matchMprime.append(searchM[closest_star])
        
    # Plot everything...

    # Get coordinates in matrix form
    featM = featset.GetMatrix()
    starM = star_subset.GetMatrix()
    matchM = match.GetMatrix()

    lbound = min(min(featM[:,0]),min(featM[:,1]))
    ubound = max(max(featM[:,0]),max(featM[:,1]))   

    plt.figure()
    plt.scatter(featM[:,0],featM[:,1])
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.show()
    
    plt.figure()
    for i in range(searchM.shape[0]):
        plt.scatter(searchM[i,0],searchM[i,1],c=[0,0,0])
        #plt.xlim([lbound,ubound])
        #plt.ylim([lbound,ubound])
    plt.show()

    lbound = min(min(starM[:,0]),min(starM[:,1]))
    ubound = max(max(starM[:,0]),max(starM[:,1]))   

    matchM[:,2] = (matchM[:,2]-min(starM[:,2]))/(max(starM[:,2]+1))
    starM[:,2] = (starM[:,2]-min(starM[:,2]))/(max(starM[:,2]+1))
    searchM[:,2] = (searchM[:,2]-min(searchM[:,2]))/(max(searchM[:,2]+1))
    
    plt.figure()
    for i in range(starM.shape[0]):
        plt.scatter(starM[i,0],starM[i,1],c=[0,0,0], alpha=abs(1-starM[i,2]))
        plt.xlim([lbound,ubound])
        plt.ylim([lbound,ubound])
    plt.show()    

    lbound = min(min(featMprime[:,0]),min(featMprime[:,1]))-.1
    ubound = max(max(featMprime[:,0]),max(featMprime[:,1]))+.1  
    
    
    plt.figure()
    for i in range(searchM.shape[0]):
        plt.scatter(searchM[i,0],searchM[i,1],s=70,c=[0,0,0],alpha=abs(1-searchM[i,2]))
    for i in range(matchM.shape[0]):
        plt.scatter(matchM[i,0],matchM[i,1],s=50,c='r') #alpha=matchM[i,2]
    for i in range(featMprime.shape[0]):
        if any(subindices==i):
            plt.scatter(featMprime[i,0],featMprime[i,1],s=50,c='g')
        else:
            plt.scatter(featMprime[i,0],featMprime[i,1],s=50,c='b')
    for i in range(len(matchMprime)):
        plt.scatter(matchMprime[i][0],matchMprime[i][1],s=50,c='y')
    plt.scatter(center[0],center[1],s=60,c='m')
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.show()
    
    plt.figure()
    for i in range(len(matchMprime)):
        plt.scatter(matchMprime[i][0],matchMprime[i][1],s=50,c='y')
    plt.xlim([lbound,ubound])
    plt.ylim([lbound,ubound])
    plt.show()
    
if __name__ == '__main__': main()