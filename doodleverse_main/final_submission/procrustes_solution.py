import numpy as np
import scipy.io as sio

def main():
    
    data = sio.loadmat('procrustes.mat')
    
    A, B = data['A'], data['B']
    
    T = GetProcrustes(A.T,B.T)
    
    TA = np.dot(A.T,T)
    
    error = np.linalg.norm(TA-B.T)
    
    print(error) # ~0

def GetProcrustes(A,B):
    '''
    This is transpose of version proposed in lab, because
    more compatible with how python handles lists
    '''
        
    # Get Procrustes rotation matrix.
    M = np.dot(B.T,A)
    [U,s,VT] = np.linalg.svd(M)
    R = np.dot(VT.T,U.T)

    return R

if __name__ == '__main__': main()