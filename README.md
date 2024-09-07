# ImageProcessingUsingML

This repository features solutions for enhancing image quality and detecting specific features using OpenCV and NumPy. The project includes two main functionalities:

### Lava Flow Detection
### Flash and No-Flash Image Merging

## Lava Flow Detection

The lava flow detection solution focuses on identifying specific features in lava flow dynamics. It utilizes advanced image processing techniques to analyze and detect key features:

### Feature Extraction:
Masking and Gamma Correction: Applies masking to remove bright areas and gamma correction to adjust image brightness.
Circle Detection: Uses the Hough Circle Transform to detect circular features in the processed image.
Thresholding and Filtering: Highlights features and removes noise for clearer detection.
Application: Ideal for analyzing lava flow dynamics and other scenarios where detecting specific features is crucial.

## Flash and No-Flash Image Merging

This solution enhances low-light images by merging flash and no-flash photos. It improves image clarity and visibility in challenging lighting conditions:

### Image Enhancement:
Image Processing: Combines details from both flash and no-flash images to create a clearer final image.
Bilateral Filtering: Preserves fine details while smoothing the image to reduce noise.
Application: Useful for improving clarity in low-light conditions and achieving better overall image quality.

## Usage

Both functionalities are designed to process images effectively. Use the respective methods to either detect specific features or enhance image quality based on your needs.
