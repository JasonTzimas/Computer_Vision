# Imports
import cv2
import numpy as np
from scipy.signal import convolve2d

# The derivative of a Gaussian is now constructed

def Sobel(dim):
    '''
    Return 3x3 Sobel filter of either X or Y direction depending on input dim
    '''
    if dim not in [0, 1]:
        raise Exception("dim must be either 0 or 1")
    if dim==0:
        return np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        return np.array([[1, 2 , 1], [0, 0, 0], [-1, -2, -1]])
    

def gaussian_filter(sigma):
    '''
    Generate a gaussian filter of kernel size dynamically calculated based on sigma
    Returns: filter, kernel size
    '''
    dim = int(np.ceil(3 * sigma))
    if dim % 2 == 0:
        dim += 1
    X, Y = np.mgrid[-dim//2+1:dim//2+1, -dim//2+1:dim//2+1]
    gf = np.exp(-((X**2 + Y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)
    
    return gf, dim


def DG(sigma):
    '''
    Generate the Derivative of a Gaussian based on the given Sigma
    Returns: Both X and Y derivative filters
    '''
    gf, dim = gaussian_filter(sigma)
    if dim <= 3:
        DGX = Sobel(dim=1)
        DGY = Sobel(dim=0)
    else:
        DGX = convolve2d(gf, Sobel(dim=1),mode="valid", boundary="symm")
        DGY = convolve2d(gf, Sobel(dim=0),mode="valid", boundary="symm")
    
    return DGX, DGY

def DG_filtering(im, sigma):
    '''
    Filter input sigma with Derivative of Gaussian filters on directions X and Y. Performs appropriate replicate padding to accomodate conservation of dimensions
    Returns: Filters, Derivatives, Gradient Magnitudes, Gradient Angles in (-pi/2, pi/2)
    '''
    DGX, DGY = DG(sigma=sigma)
    padding = int(DGX.shape[0] // 2)
    new_im = np.ndarray((im.shape[0] + 2 * padding, im.shape[1] + 2 * padding))
    new_im[padding:-padding, padding:-padding] = im
    new_im[padding:-padding, :padding] = np.repeat(im[:, 0].reshape(-1, 1), padding, axis=1)
    new_im[padding:-padding, -padding:] = np.repeat(im[:, 0].reshape(-1, 1), padding, axis=1)
    new_im[:padding, padding:-padding] = np.repeat(im[0, :].reshape(1, -1), padding, axis=0)
    new_im[-padding:, padding:-padding] = np.repeat(im[0, :].reshape(1, -1), padding, axis=0)
    new_im[:padding, :padding] = im[0, 0] * np.ones((padding, padding))
    new_im[:padding, -padding:] = im[0, -1] * np.ones((padding, padding))
    new_im[-padding:, :padding] = im[-1, 0] * np.ones((padding, padding))
    new_im[-padding:, -padding:] = im[-1, -1] * np.ones((padding, padding))

    dX = convolve2d(new_im, DGX, mode="valid")
    dY = convolve2d(new_im, DGY, mode="valid")
    magn = np.sqrt(dX**2 + dY**2)
    angle = np.arctan(dX / dY) 
    return dX, dY, DGX, DGY, magn, angle

# Non-Maxima Suppresion
# Way 1: Based on 3x3 neighborhood regardless of angle

def non_maxima_suppresion_a(gradient_magnitudes):
    '''
    Perform Non-Maximaum-Suppression using Strategy A: (Suppress non maximum pixels regardless of the gradient angle)
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0] + 2, gradient_magnitudes.shape[1] + 2))
    output[1:-1, 1:-1] = gradient_magnitudes
    output[:, [0, -1]] = 0
    output[[0, -1], -1] = 0
    out_list = [0 if output[i, j] < np.max(output[i-1:i+2, j-1:j+2]) else output[i, j] for i in range(1, output.shape[0]-1) for j in range(1, output.shape[1] - 1)]
    out_array = np.array(out_list).reshape(gradient_magnitudes.shape[0], gradient_magnitudes.shape[1])
    output[1:-1, 1:-1] = out_array
    return output

# Way 2: Discretizing the angles in 4 bins, namely (0, 45, 90, 135) and performing non-maxima suppresion at these angles
def non_maxima_suppresion_b(gradient_magnitudes, gradient_angles):
    '''
    Perform Non-Maximaum-Suppression using Strategy B: (Suppress non maximum pixels in the direction of the quantized angle in bins of angle (0, 45, 90, 135))
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0], gradient_magnitudes.shape[1]))
    output[:, [0, -1]] = 0
    output[[0, -1], :] = 0
    output[1:-1, 1:-1] = gradient_magnitudes[1:-1, 1:-1]

    # Map angles
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            ang = gradient_angles[i-1, j-1]
            if np.abs(ang) < np.pi / 8:
                if gradient_magnitudes[i, j] < np.max(gradient_magnitudes[i, j-1:j+2]):
                    output[i, j] = 0
            elif ang >= np.pi / 8 and ang < 3 * np.pi / 8:
                if gradient_magnitudes[i, j] < gradient_magnitudes[i-1, j+1] or gradient_magnitudes[i, j] < gradient_magnitudes[i+1, j-1]:
                    output[i, j] = 0
            elif ang <= - np.pi / 8 and ang > - 3 * np.pi / 8:
                if gradient_magnitudes[i, j] < gradient_magnitudes[i+1, j+1] or gradient_magnitudes[i, j] < gradient_magnitudes[i-1, j-1]:
                    output[i, j] = 0
            else:
                if gradient_magnitudes[i, j] < np.max(gradient_magnitudes[i-1:i+2, j]):
                    output[i, j] = 0
    
    return output


def non_maxima_suppresion_c(gradient_magnitudes, gradient_angles):
    '''
    Perform Non-Maximaum-Suppression using Strategy C: (Suppress non maximum pixels in the direction of the actual gradient angle using interpolated pixel intensities
    Returns: Suppressed filtered Image
    '''
    output = np.ndarray((gradient_magnitudes.shape[0], gradient_magnitudes.shape[1]))
    output[:, [0, -1]] = 0
    output[[0, -1], :] = 0
    output[1:-1, 1:-1] = gradient_magnitudes[1:-1, 1:-1]
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            ang = gradient_angles[i-1, j-1]
            if ang > 0 and ang <= np.pi / 4:
                coef = np.tan(ang)
                p1 = (1 - coef) * gradient_magnitudes[i, j+1] + coef * gradient_magnitudes[i-1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i, j-1] + coef * gradient_magnitudes[i+1, j-1] 
                #print(coef, ang, "Case = 1")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            elif ang > np.pi / 4 and ang <= np.pi / 2:
                coef = np.tan(np.pi / 2 - ang)
                p1 = (1 - coef) * gradient_magnitudes[i-1, j] + coef * gradient_magnitudes[i-1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i+1, j] + coef * gradient_magnitudes[i+1, j-1] 
                #print(coef, ang, "Case = 2")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            elif ang <= 0 and ang > - np.pi / 4:
                coef = np.tan(np.abs(ang))
                p1 = (1 - coef) * gradient_magnitudes[i, j+1] + coef * gradient_magnitudes[i+1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i, j-1] + coef * gradient_magnitudes[i-1, j-1] 
                #print(coef, ang, "Case = 3")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            else:
                coef = np.tan(np.pi / 2 - np.abs(ang))
                p1 = (1 - coef) * gradient_magnitudes[i+1, j] + coef * gradient_magnitudes[i+1, j+1]
                p2 = (1 - coef) * gradient_magnitudes[i-1, j] + coef * gradient_magnitudes[i-1, j-1] 
                #print(coef, ang, "Case = 4")
                if gradient_magnitudes[i, j] < p1 or gradient_magnitudes[i, j] < p2:
                    output[i, j] = 0
            
           
    
    return output


def connectivity_labeling(input_image):
    '''
    Adapt connectivity labeling algorithm for Canny Edge detection Hysterisis Thresholding
    Returns: Canny Edge Detection output
    '''
    label = 1
    flag_image = np.zeros_like(input_image)
    output_image = np.zeros_like(input_image)
    edge_pixels = []
    for i in range(flag_image.shape[0]):
        for j in range(flag_image.shape[1]):
            if input_image[i, j] == 2:
                if flag_image[i, j] == 0: # Pixel not yet labeled
                    # Label all component-connected pixels
                    flag_image[i, j] = label
                    queue = [(i, j)]
                    while queue:
                        tail = queue[-1]
                        queue.pop()
                        new_queue = [(k, n) for k in range(tail[0]-1, tail[0]+2) for n in range(tail[1]-1, tail[1]+2) if (k != tail[0] or n != tail[1])
                                        and k >= 0 and k <= flag_image.shape[0] - 1
                                        and n >= 0 and n <= flag_image.shape[1] - 1 and (input_image[k, n] == 1 or input_image[k, n] == 2) and flag_image[k, n] == 0]
                        for pix in new_queue: 
                            flag_image[pix] = label
                            output_image[pix] = 2
                            edge_pixels.append(pix)
                        queue = new_queue + queue

    return output_image, edge_pixels



def Canny(img, thresh1, thresh2, sigma):
    '''
    Canny Edge Detection from scratch
    Inputs: threshold1, threshold2 (t1 < t2), sigma for Bluring
    Returns: Canny Edge Detection Output
    '''
    dX, dY, DGX, DGY, magn, angle = DG_filtering(img, sigma=sigma)
    thresh1 = np.std(img) / 255 * thresh1
    thresh2 = np.std(img) / 255 * thresh2
    out = non_maxima_suppresion_c(magn, angle)
    out_thresh = np.zeros_like(out)
    out_thresh[np.logical_and(out >= thresh1, out <= thresh2)] = 1
    out_thresh[out > thresh2] = 2
    out_final, edge_pixels = connectivity_labeling(out_thresh)
    return out_final, out_thresh, angle, edge_pixels
