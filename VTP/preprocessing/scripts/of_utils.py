import numpy
import cvapi.cv2api as cv2
import scipy.signal

def deriv(u, v):
    '''
    u: npy array
        X-components of optical flow vector field
    v: npy array
        Y-components of optical flow vector field

    Computes the image derivatives of the optical flow vectors so that
    we can compute some interesting differential image invariants. 

    uu: X-derivative of the X-component.
    vu: X-derivative of the Y-component.
    uv: Y-derivative of the X-component.
    vv: Y-derivative of the Y-component.
    
    '''
    
    # filter settings, can be changed 
    sigmaBlur = 1
    gBlurSize = 2 * np.around(2.5 * sigmaBlur) + 1
    grid = np.mgrid[1:gBlurSize + 1] - np.around((gBlurSize + 1) / 2)
    g_filt = np.exp(-(grid ** 2) / (2 * (sigmaBlur ** 2)))
    g_filt /= np.sum(g_filt)
    dxg_filt = (-grid / (sigmaBlur ** 2)) * g_filt
    
    # derivative
    xu = scipy.signal.sepfir2d(u, dxg_filt, g_filt)
    yu = scipy.signal.sepfir2d(u, g_filt, dxg_filt)
    xv = scipy.signal.sepfir2d(v, dxg_filt, g_filt)
    yv = scipy.signal.sepfir2d(v, g_filt, dxg_filt)

    return [uu, vu, uv, vv]

def curl(u, v):
    '''
    u: npy array
        X-components of optical flow vector field
    v: npy array
        Y-components of optical flow vector field

    computes curl, important for ciliary detection
    '''

    uu, vu, uv, vv = deriv(u, v)
    return uv - vu

def deformation(u, v):
    '''
    u: npy array
        X-components of optical flow vector field
    v: npy array
        Y-components of optical flow vector field

    computes biaxial shear
    '''

    uu, vu, uv, vv = deriv(u, v)
    return [uu - vv, uv + vu]

def main():
    # no driver
