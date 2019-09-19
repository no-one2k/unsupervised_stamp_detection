# unsupervised_stamp_detection

## Task

Detect stamps and signs on image, plot and save bounding boxes

## Solution

1. Preprocess image:
    - resize
    - binarize (RGB-image -> greyscale -> otsu threshold -> BW-image)
    - blur
2. Find clusters
3. For each cluster:
    - detect edges (cv2.Canny)
    - close figures (dilate+morph.closing)
    - find contours
4. Select largest contours

Files:
- research.ipynb - sandbox to find possible solutions
- processphoto.py - final script

