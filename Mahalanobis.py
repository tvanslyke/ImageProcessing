""" 
    For MCC UAV club.

    Contains functionality for finding mahalanobis distance of all pixels in an image 
    to the mean RGB-values of all the pixels. 
    
    Works very, VERY well as long as targets are a different color than other major
    colors in the image.  Otherwise it's kind of meh.  
    
    This should serve as an example in how using Python's looping constructs can seriously
    slow down an image processing algorithm.  Python has SOOOOO many C-compiled, over-optimized
    libraries like numpy and pandas.  It's rare that you actually can't replace a loop with a
    faster (usually more readable) library function.   Use mahal_slow() to see how this can be true.
    
    Author:              Timothy Van Slyke  <--- I made it more better
    Original smart-guy:  Brandon Doyle      <--- His idea
    
    STRONGLY SUGGESTED READINGS:  
    WTF is einsum?:       http://stackoverflow.com/questions/26089893/understanding-numpys-einsum
                          https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
                          https://en.wikipedia.org/wiki/Einstein_notation
                          
    WTF is Mahalanobis?:  https://en.wikipedia.org/wiki/Mahalanobis_distance#Intuitive_explanation
"""

import numpy as np
from numpy.linalg import inv
#from scipy.spatial.distance import mahalanobis  <---- don't use this slow fuck
import cv2      

import tqdm # "sudo pip install tqdm" or "sudo apt-get install python-tqdm" for 
            # progress bars on for loops and real-time looping statistics.
            # If on Windows, just do "pip install tqdm".


# run this to see just how much slower one python for loop can be
def mahal_slow(img, select = 10000):
    """ Shitty version of Mahalanobis that is one million times slower left here 
        for comparison. This is NOT the original implementation, but rather one made
        to closely resemble the one below, but uses a Python for-loop instead of 
        numpy.einsum().  Run them side by side to see how important it is to avoid
        using native Python functionality as much as possible for this application."""
        
    arr = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    if isinstance(select,int):
        select = np.random.choice(arr.shape[0], select)
    select = arr[select]
    invcovar = inv(np.cov(np.transpose(select)))
    meanval =np.mean(select, axis = 0)
    output = np.zeros((img.shape[0] * img.shape[1], 1))
    mahalanobis = lambda x,y: np.dot(np.dot(np.array([x]), y),np.transpose([x]))
    for index in tqdm.tqdm(xrange(len(output))):
        output[index] = mahalanobis(arr[index]-meanval, invcovar)
    ret = np.sqrt(output)
    return ret.reshape(img.shape[:-1]), ret
    
def mahal(img, select = 10000):
    """ Improved Mahalanobis distance algorithm originally written 
        by Brandon Doyle.  This is written with care taken to NOT 
        use Python loops.  The key breakthrough here is numpy.einsum().
        
        IT IS STRONGLY SUGGESTED THAT FUTURE STUDENTS AVOID PYTHON LOOPS 
        AND *** LEARN EINSUM *** !!!!!!!  numpy.einsum is basically black
        magic until you understand it but ONCE YOU DO you can make very, very
        efficient operations out of previous slow ones.  
        Seriously, learn einsum.
        
        Args:
        img:     Input image to compute mahalanobis distance on.
        select:  Number of pixels to randomly select when computing the
                 covariance matrix OR a specified list of indices in the 
                 image.  Specifying the indices outright would be faster
                 if you have a bunch of images that are the same size.  
                 The indices could be reused each time rather than recalculating
                 every time you call the function.
        """
    # Flatten image to just one long array of RGB-valued pixels 
    arr = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    
    # if 'select' is a number, generate a list of size 'select' containing
    # random indices in 'arr'
    if isinstance(select,int):
        select = np.random.choice(arr.shape[0], select)
    # choose pixels from 'arr' at positions specified in 'select' 
    select = arr[select]
    
    # calculate the covariance matrix and immediately calculate its inverse
    invcovar = inv(np.cov(np.transpose(select)))
    
    # calculate the mean value of pixels in 'select'. We could get the actual mean from
    # arr', but it's faster to do it from the sub-sample we already have in 'select'.
    meandiff = arr - np.mean(select, axis = 0)
    
    '''
        Formula:
            pp = particular pixel
            mp = mean pixel value
            MD = sqrt( transpose(pp - mp) * (covariance_mat^-1) * (pp - mp) )
        
        You'll notice that I've reversed which side gets transposed.  It's just because the
        pixels are stored as rows at this point (above assumes column vectors) and I've set up
        numpy.einsum() to handle the multiplication properly.
    '''
    
    # calculate the first multiplication
    output = np.dot(meandiff, invcovar)
    # do literally everything else all in this step, then reshape back to image dimensions and return
    output = np.sqrt(np.einsum('ij,ij->i', output, meandiff))
    return output.reshape(img.shape[:-1])

def amplify(image, mask, cutoff = 0):
    """ Applies 'mask' to 'image' while cutting off values below a percent
        brightness specified in cuttoff.
        
        Args:
        image:   RGB/BGR image to amplify/apply mask to.
        mask:    Grayscale/monocolor (one number per pixel) "image" to act as a mask.
        cuttoff: Percent brightness below which pixels should be cut off.  This is 
                 referring to the brightness of pixels in 'mask' and the 100% setpoint 
                 is the brighted pixel in 'mask'.
        
        Returns the 'amplified' image."""
    
    # cut off values below specified threshold
    mask_modf = np.greater(mask*100.0/np.amax(mask), cutoff) * mask
    
    # apply mask
    output = np.einsum('ij,ijk->ijk',mask_modf, img)/np.amax(filt)
    
    # scale to 255 (2^8 - 1 = unsigned 8-bit max)
    output *= (255.0/np.amax(image))
    
    return np.uint8(output)
    

# 'int main()' for you C people
if __name__ == '__main__':
    # show images with matplotlib because cv2.imshow() sucks a lot
    import matplotlib.pyplot as plt
    from time import time
    
    # we're gonna plot these
    pixelcounts = []
    times = []
    
    # do stuff to 8 images
    for num in range(8):
        # select image
        img = cv2.imread("testimg" + str(num) + ".jpg")
        # turn it to RGB because nobody uses BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # stuff for plots
        pixelcounts.append(img.shape[0] * img.shape[1])
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        

        # do fast mahalanobis distance calculation
        t = time()
        filt = mahal(img)
        times.append(time() - t)
        
        # put images in the 'plot buffer' if you will
        ax1.imshow(img, interpolation='nearest')
        ax2.imshow(filt, interpolation='nearest', cmap='Greys_r' )
        ax3.imshow(amplify(img, filt, cutoff = 0), interpolation='nearest' )
        ax4.imshow(amplify(img, filt, cutoff = 25), interpolation='nearest' )
        
    # plot the other stuff and show
    plt.figure()
    plt.plot(pixelcounts)
    plt.title("Number of pixels")
    plt.figure()
    plt.plot(times)
    plt.title("Mahalanobis distance calculation time")
    plt.show()
    
    
    