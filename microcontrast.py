
import numpy as np
from typing import Union, Tuple
import cv2
import os
from scipy.ndimage import convolve

def calculate_microcontrast(
                            image: np.ndarray,
                            window_size: int = 3,
                            sensitivity: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Calculate microcontrast in the image using vectorized operations
    Supports both 2D and 3D (color) images
    
    Args:
        image: Input image as float32 numpy array (HxW or HxWxC)
        window_size: Size of the sliding window (odd number)
        sensitivity: Contrast sensitivity parameter
        
    Returns:
        Tuple of (processed image, contrast metric)
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError("Image must be a float32 numpy array")
        
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
        
   # Handle both 2D and 3D images
    is_3d = len(image.shape) == 3
    
    if is_3d:
        # Process each channel separately
        channels = []
        metrics = []
        
        for channel in range(image.shape[2]):
            processed_channel, channel_metric = _process_single_channel(
                image[:,:,channel],
                window_size,
                sensitivity
            )
            channels.append(processed_channel)
            metrics.append(channel_metric)
        
        # Combine processed channels
        output = np.stack(channels, axis=2)
        contrast_metric = np.mean(metrics)
    else:
        output, contrast_metric = _process_single_channel(
            image,
            window_size,
            sensitivity
        )
    
    return output, contrast_metric

def _process_single_channel(
                            channel: np.ndarray,
                            window_size: int,
                            sensitivity: float) -> Tuple[np.ndarray, float]:
    """
    Process a single channel of the image
    
    Args:
        channel: 2D array representing a single channel
        window_size: Size of the sliding window
        sensitivity: Contrast sensitivity parameter
        
    Returns:
        Tuple of (processed channel, contrast metric)
    """
    # Create padded image for sliding window
    pad_size = window_size // 2
    padded = np.pad(channel, pad_size, mode='reflect')
    
    # Create kernel for average calculation (excluding center)
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel[pad_size, pad_size] = 0  # Exclude center pixel
    kernel = kernel / (np.sum(kernel))  # Normalize weights
    
    # Calculate local mean using convolution
    local_mean = convolve(padded, kernel, mode='constant', cval=0.0)
    
    # Extract the valid part (removing padding)
    local_mean = local_mean[pad_size:-pad_size, pad_size:-pad_size]
    
    # Calculate microcontrast
    contrast = (channel - local_mean) * (sensitivity / 100.0)
    output = channel + contrast
    
    # Calculate contrast metric
    contrast_metric = np.mean(np.abs(output - channel))
    
    return output, contrast_metric

def analyze_frequency_components(image: np.ndarray) -> dict:
    """
    Analyze frequency components of the image
    
    Args:
        image: Input image as float32 numpy array
        
    Returns:
        Dictionary containing frequency analysis results
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    
    # Calculate magnitude spectrum
    magnitude = np.abs(fft_shift)
    
    # Analyze different frequency bands
    high_freq = np.mean(magnitude[magnitude > np.percentile(magnitude, 90)])
    mid_freq = np.mean(magnitude[magnitude > np.percentile(magnitude, 50)])
    low_freq = np.mean(magnitude[magnitude <= np.percentile(magnitude, 50)])
    
    return {
        'high_frequency_content': float(high_freq),
        'mid_frequency_content': float(mid_freq),
        'low_frequency_content': float(low_freq),
        'frequency_ratio': float(high_freq / low_freq if low_freq > 0 else 0)
    }


if __name__ == '__main__':
    img = cv2.imread(os.getcwd() + "/picture/DSCF0002.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255

    img, _ = calculate_microcontrast(img, 7, 100)

    img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR)*255).astype(np.uint8)
    cv2.imshow('microcontrast', img)
    cv2.waitKey(0)
