import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def project(ra, dec, c_ra, c_dec):
    '''
    Finds the Mollweide projection coordinates (x,y) for the point (ra,dec) around
    point (c_ra,c_dec).
    '''

    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    c_ra = np.deg2rad(c_ra)
    c_dec = np.deg2rad(c_dec)
    
    # Find theta
    theta_0 = dec - c_dec
    epsilon = 10**-6
    error = 1+epsilon
    while error > epsilon:
        m = (2 * theta_0+np.sin(2 * theta_0) - np.pi * np.sin(dec - c_dec))/(2+2 * np.cos(2 * theta_0))
        theta_1 = theta_0 - m
        error = np.abs(theta_1 - theta_0)
        theta_0 = theta_1
    # Compute (x,y) coordinates
    x = 2 * np.sqrt(2) * (ra - c_ra) * np.cos(theta_0)/np.pi
    y = np.sqrt(2) * np.sin(theta_0)
    return [x,y]
    
def main():
    
    stars = sio.loadmat('sub-table.mat')
    
    starsmat = np.array([stars['RA'],stars['Dec']])
    
        
    
    n = starsmat.shape[2]    
    
    XYmat = np.zeros((n,2))

    plt.figure()    
    
    for i in range(n):
        
        [XYmat[i,0],XYmat[i,1]] = project(starsmat[0][0][i],starsmat[1][0][i],293,-2.8)
        plt.scatter(XYmat[i,0],XYmat[i,1])
    
    print(XYmat)
    plt.show()
if __name__=='__main__': main()