`import cv2
import numpy as np

def solution(image_path):
    image = cv2.imread(image_path)

    mask = np.all(image > 89, axis=-1)    #pixels where all color channels>89 to be set as black(0)
    image[mask] = 0

    gamma_corrected = np.power(image / 255.0, 0.8) * 255.0    #gamma corrction to improve contrast and darken it
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    blue, green, red = cv2.split(gamma_corrected)
    red_channel_only = cv2.merge([np.zeros_like(blue), np.zeros_like(green), red]) #bcs red channel is useful

    blurred_image = cv2.GaussianBlur(red_channel_only, (9, 9), 0)    #blurr image for better circle detection
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)    

    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=20, maxRadius=70)

    if circles is not None:    
        black_image = np.zeros_like(image)
        return black_image    #if detected, return black image

    gray_image = cv2.cvtColor(red_channel_only, cv2.COLOR_BGR2GRAY)
    _, red_thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#Thresholding highlights areas of interest (bright regions) in the red channel, and median filtering helps to clean up the binary image.
    median_filtered = cv2.medianBlur(red_thresholded, 25)

    for row in range(median_filtered.shape[0]):    #For each row, find the leftmost and rightmost white pixels and fill in between them
        white_pixels = np.where(median_filtered[row] == 255)[0]    #ensures continuous
        if len(white_pixels) > 0:
            leftmost_pixel = white_pixels[0]
            rightmost_pixel = white_pixels[-1]
            median_filtered[row, leftmost_pixel:rightmost_pixel] = 255
            
    bgr_image = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)

    return bgr_image
