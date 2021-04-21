import numpy as np
from scipy import sparse
import numpy.ma as ma


BW1 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
BW2 = 0o100000 # 32768 or 2<<14 or 1<<15
BW3 = 0o140000 # 49152 or 3<<14
MSK = 0x3fff # 16383 or (1<<14)-1 - 14-bit mask


def divide_protected(num, den, vsub_zero=0) :
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den


def image_from_dxy(d,x,y, **kwargs):
    """ Convert tiled detector arrays in a single detector image.
    Function taken from smalldata_tools
    
    Parameters
        ----------
        d: np.ndarray
            Tiled detector data
        x: np.ndarray
            x image coordinates
        y: np.ndarray
            y image coordinates
    """
    
    if np.array(x).shape!=np.array(y).shape or  np.array(d).shape!=np.array(y).shape:
        print('shapes of data or x/y do not match ',np.array(d).shape, np.array(x).shape, np.array(y).shape)
        return None

    outShape = kwargs.pop("outShape", None)
    #check if inpu arrays are already indices. 
    if np.abs(x.flatten()[0]-int(x.flatten()[0]))<1e-12: #allow for floating point errors.
        ix = x.astype(int)
        iy = y.astype(int)
        ix = ix - np.min(ix)
        iy = iy - np.min(iy)
    else:
        #cspad
        if x.shape==(32,185,388): imgShape=[1689,1689]
        #cs140k
        elif x.shape==(2,185,388): imgShape=[391,371] #at least for one geometry
        #epix100a
        elif x.shape==(704,768): imgShape=[709,773]
        #jungfrau512k
        elif x.shape==(1,512,1024): imgShape=[514,1030]
        elif x.shape==(512,1024): imgShape=[514,1030]
        #jungfrau1M
        elif x.shape==(2,512,1024): imgShape=[1064,1030]
        elif len(x.shape)==2:#is already image (and not special detector)
            return d
        else:
            if outShape is None:
                print('do not know which detector in need of a special geometry this is, cannot determine shape of image')
            else:
                imgShape=outShape
            return
        ix = x.copy()
        ix = ix - np.min(ix)
        ix = (ix/np.max(ix)*imgShape[0]).astype(int)
        iy = y.copy()
        iy = iy - np.min(iy)
        iy = (iy/np.max(iy)*imgShape[1]).astype(int)

    if outShape is None:
        outShape = (np.max(ix)+1, np.max(iy)+1)
        
    img = sparse.coo_matrix((d.flatten(), (ix.flatten(), iy.flatten())), shape=outShape).todense()
    return img


class Correct_JF_img(object):
    def __init__(self, ped=None, gain=None, offset=None, ix=None, iy=None, mask=None):
        """ Correct Jungfrau1M detector images. Applies pedestal and gain maps.
        Inputs to be taken from smalldata h5 files under /userDataCfg/jungfrau1M/
        
        Parameters
        ----------
        ped, gain, offset: np.ndarray
            JF calibration
        ix, iy: np.ndarray
            mapping from tiles to image
        mask: np.ndarray
            pixel mask
        
        mask: /userDataCfg/jungfrau1M/cmask
        
        SOURCE: https://github.com/lcls-psana/Detector/blob/master/src/UtilsJungfrau.py
        """
        # make the tiled correction array into image format
        ped = np.asarray([image_from_dxy(p, ix, iy) for p in ped])
        gain = np.asarray([image_from_dxy(g, ix, iy) for g in gain])
        offset = np.asarray([image_from_dxy(o, ix, iy) for o in offset])
        if mask is None:
            mask = gain==0
        else:
            mask = np.logical_not(image_from_dxy(mask,ix,iy))
            mask = np.repeat(np.asarray(mask[np.newaxis,:,:]), 3, axis=0)
        
        self.mask = mask
        self.ped = ma.masked_array(ped, mask)
        self.gain = ma.masked_array(gain, mask)
        self.offset = ma.masked_array(offset, mask)
        
        self.gfac = divide_protected(np.ones_like(self.ped), self.gain)
        return
    
    def correct_img(self, img):
        arr = ma.masked_array(img, self.mask[0])
        peds = self.ped
        gains = self.gain
        offs = self.offset
        gfac = self.gfac
        
        # Define bool arrays of ranges
        # faster than bit operations
        gr0 = arr <  BW1              # 490 us
        gr1 =(arr >= BW1) & (arr<BW2) # 714 us
        gr2 = arr >= BW3              # 400 us
        
        # Subtract pedestals
#         arrf = np.array(arr & MSK, dtype=np.float32)
        arrf = ma.masked_array(arr & MSK, dtype=np.float32)
        arrf[gr0] -= peds[0,gr0]
        arrf[gr1] -= peds[1,gr1] #- arrf[gr1]
        arrf[gr2] -= peds[2,gr2] #- arrf[gr2]
        
        factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=1) # 2msec
        offset = np.select((gr0, gr1, gr2), (offs[0,:], offs[1,:], offs[2,:]), default=0)
        
        arrf -= offset # Apply offset correction
        arrf *= factor # Gain correction
        return arrf