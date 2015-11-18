
import numpy as np
import matplotlib.pyplot as plt

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
        

def main():
    
    plt.close("all")    
    
    numStars = 2000
    
    #specify feature points
    featdata = np.array([[.3,.3],[.6,.2],[.5,.6]]) 
    
    #uncomment to generate random feature points
    #featdata = None

    #generate random starset, featureset, get feature angles...
    starset = StarSet(numStars = numStars, spread=100)
    featset = FeatureSet(data = featdata)
    feat_angles = featset.GetAngles()
    
    #error tolerance in radians... around .05 is good. .01 can take too long.
    epsilon = .05
    num_tries = 0
    
    '''picks 3 random points and cycles through all permutations to see if
    there is a good match'''
    while True:
        sub_verts = np.random.choice(numStars,3,replace=False)
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
    
    plt.figure()
    plt.scatter(starM[:,0],starM[:,1])
    plt.xlim([0,max(starM[:,0])])
    plt.ylim([0,max(starM[:,1])])
    plt.show()    
    
    plt.figure()
    plt.scatter(matchM[:,0],matchM[:,1])
    plt.xlim([0,max(starM[:,0])])
    plt.ylim([0,max(starM[:,1])])
    plt.show()

if __name__ == '__main__': main()