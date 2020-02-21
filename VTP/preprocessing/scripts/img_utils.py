from scipy import ndimage
import numpy as np
from sklearn.feature_extraction import image
from skimage import filters

def iou(ypred, ytrue):
    '''                                                                                                                                         
    ypred: numpy array with shape (n, m) 
        predicted mask with binary values for each pixel (1 = cilia, 0 = background)                                              
    ytrue: numpy array with shape (n, m)
        ground truth seg mask
    
    IOU: float n 
        ratio of intersecting pixels to union pixels    
    '''
    mask_pred = (ypred == 1)
    mask_true = (ytrue == 1)
    inter = (mask_true & mask_pred).sum()
    union = (mask_true | mask_pred).sum()
    iou = float(inter) / float(union)
    return iou
    
def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    helper for rolling window var, padding 0 if mean not specified
    '''
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def rolling_window_var(imarr,kernel):
    '''
    imarr: numpy array with shape (n, m) 
        one frame of video with n rows and m cols
    kernel: integer n
        kernel size, assumes kernel rows = cols

    var: numpy array with shape (n, m)
        local variance at each pixel within kernel neighborhood (mean or constant padding)
    '''
    imrow, imcol = imarr.shape
    imarr = np.pad(imarr,kernel//2,pad_with,padder=np.mean(imarr))
    patches = image.extract_patches_2d(imarr, (kernel, kernel))
    var = np.array([ndimage.variance(patch) for patch in patches]).reshape((imrow,imcol))
    return var
