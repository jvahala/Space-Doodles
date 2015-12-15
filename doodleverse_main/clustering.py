import numpy as np
import search
from sklearn.cluster import KMeans
import scipy.io as sio

def main():
    
    star_tab = search.StarTable()
    
    print('Reduced star table has',star_tab.num_stars,'stars.')    
    
    n = len(star_tab.tab)
    
    angles_computed = set()
    startrios = []
    angles = []
    
    print('Computing angles',end='')    
    
    num = 0
    
    for i in range(n):
        
        print('.',end='')        
        
        local_star_tab = star_tab.ClosestStars(i,25)
        
        local_stars = local_star_tab.MollProject()
        
        n = local_stars.length
        
        mag_tol = .8
        
        for i in range(2,n):
            for j in range(1,i):
                for k in range(j):
                    trio = (local_stars.indices[i],local_stars.indices[j],local_stars.indices[k])
                    trio = tuple(sorted(trio))
                    if trio in angles_computed:
                       # print('Repeat found!')
                        continue
                    if any(np.array([local_stars.points[i].mag, local_stars.points[j].mag, local_stars.points[k].mag])> mag_tol):
                        #print('Dim star found!')
                        continue
                    
                    else:
                        a = list(local_stars.GetAngles([i,j,k]))
                        if any(np.isnan(a)):
                            continue
                        else:
                            angles.append(sorted(a))
                            startrios.append(trio)
                            angles_computed.add(trio)
                            num += 1
                            
    
    print()
    
    print('Found',len(startrios),'star trios')
    
    angles = np.array(angles)
    startrios = np.array(startrios)
    
    print('Computing clusters...')
    
    numclusters = len(startrios)//30
    
    if numclusters < 2: numclusters = 2
    
    clustering = KMeans(n_clusters = numclusters, n_init=10)

    clustering.fit(angles)
    
    centers = clustering.cluster_centers_
    labels = clustering.labels_

    output1 = dict()
    output2 = dict()

    output1['startrio'] = startrios
    output1['label'] = np.array(labels).reshape(len(labels),1)

    output2['clustercenter'] = np.array(centers)
    
    sio.savemat('starlabels.mat',output1)
    sio.savemat('clustercenters.mat',output2)


if __name__ == '__main__': main()