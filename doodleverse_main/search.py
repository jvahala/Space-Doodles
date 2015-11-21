
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
import pydl.pydlutils.spheregroup.spherematch as spherematch

class StarTable:
    def __init__(self, file = 'hyg_catalog.fits'):
        self.tab = table.Table.read(file, format = 'fits')
        
    def ClosestStars(self, center_index, radius):
        """
        Returns subtable of stars that is
        within radius of the center_index star
        """
        
        tRA = self.tab['RA']    
        tDec = self.tab['Dec']
        cRA = np.array([self.tab['RA'][center_index]])
        cDec = np.array([self.tab['Dec'][center_index]])
        
        m = spherematch(tRA, tDec, cRA, cDec, maxmatch = 0, matchlength = radius)
        
        return self.tab[m[0]]
        
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
            ra = np.deg2rad(subtab['RA'][i])
            dec = np.deg2rad(subtab['Dec'][i]) - c_dec
            
            #longitude = RA = lambda ?
            #latitude = dec = phi ?
            
            t_0 = dec
            epsilon = .000001
            
            #iterate to find theta
            error = 1+epsilon        
            while error > epsilon:
                t_1 = t_0 - (2*t_0+np.sin(2*t_0)-np.pi*np.sin(dec))/(2+2*np.cos(2*t_0))
                error = np.abs(t_1 - t_0)
                t_0 = t_1
            
            xy_pos[i,0] = 2*np.sqrt(2)*(ra-c_ra)*np.cos(t_0)/np.pi
            xy_pos[i,1] = np.sqrt(2)*np.sin(t_0)
            
            #create star object with parameters
            starsxy.append(Star(pos=xy_pos[i],bright=subtab['Mag'][i]))

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
        '''get angles between vertices specified by verts (defaults to first three points)'''
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
        '''returns position data for all points in set in the form of a nx2 matrix'''
        return np.array([self.points[i].pos for i in range(len(self.points))])
        
    def GetLength(self):
        return len(self.points)
        
class FeatureSet(SetOfPoints):
    def __init__(self, numPoints = 3, data = None):
        '''generates a random set of points or assigns points in data array'''
        self.points = []
        if data is None:
            for i in range(numPoints):
                self.points.append(Point())
        else:
            for i in range(data.shape[0]):
                self.points.append(Point(pos = data[i,:]))
        
class Star(Point):
    def __init__(self, pos = None, bright = None, spread = 1):
        '''generates a random star or a star with given position/brightness'''
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
        
    def GetSubset(self, subsize = 3, indices = None):
        '''generates subset of StarSet, either random or specified by indices'''
        if indices is None:
            subset = np.random.choice(self.points, subsize, replace=False)
        else:
            subset = [self.points[i] for i in indices]
        
        return StarSet(data = subset)
        
    def GetMatrix(self):
        '''returns position/mag data for all points in set in the form of a nx3 matrix'''
        p = np.array([self.points[i].pos for i in range(len(self.points))])
        m = np.array([self.points[i].bright for i in range(len(self.points))])
        
        m = m.reshape(m.shape[0],1)

        return np.hstack((p,m))

def main():
    
    plt.close("all")    
    
    star_tab = StarTable()
    
    #specify index of star to search around
    center = 0
    
    #pick random index of star to search around
    center = np.random.randint(len(star_tab.tab))
    
    #get subtable of stars near center star
    m = star_tab.ClosestStars(center,10)

    #convert from spherical to mollweide projection
    star_subset = star_tab.MollProject(m)
    
    
    RandSearchTest(star_subset)
    

    
def RandSearchTest(starset):
    
    #specify feature points
    featdata = np.array([[.3,.3],[.6,.2],[.5,.6]]) 
    
    #generate random feature points
    featdata = None

    #generate random starset, featureset, get feature angles...
    featset = FeatureSet(data = featdata)
    feat_angles = featset.GetAngles()
    
    #error tolerance in radians
    epsilon = .005
    num_tries = 0
    
    
    #print(starset.points[2].pos)    
    
    '''picks 3 random points and cycles through all permutations to see if
    there is a good match'''
    while True:
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
        print(num_tries)
    
    print(sub_angles)
    print(feat_angles)    
    
    #generate starset object
    match = starset.GetSubset(indices = sub_verts)

    #get coordinates in matrix form
    featM = featset.GetMatrix()
    starM = starset.GetMatrix()
    matchM = match.GetMatrix()

    #plot everything...

    plt.figure()
    plt.scatter(featM[:,0],featM[:,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    
    lbound = min(min(starM[:,0]),min(starM[:,1]))
    ubound = max(max(starM[:,0]),max(starM[:,1]))   

    matchM[:,2] = (matchM[:,2]-min(starM[:,2]))/(max(starM[:,2]+1))
    starM[:,2] = (starM[:,2]-min(starM[:,2]))/(max(starM[:,2]+1))
    
    plt.figure()
    for i in range(starM.shape[0]):
        plt.scatter(starM[i,0],starM[i,1],c=[0,0,0],alpha=starM[i,2])
        plt.xlim([lbound,ubound])
        plt.ylim([lbound,ubound])
    plt.show()    
    
    
    plt.figure()
    for i in range(matchM.shape[0]):
        plt.scatter(matchM[i,0],matchM[i,1],c=[0,0,0],alpha=matchM[i,2])
        plt.xlim([lbound,ubound])
        plt.ylim([lbound,ubound])
    plt.show()    

if __name__ == '__main__': main()