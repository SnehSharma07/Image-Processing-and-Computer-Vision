import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Update this to your image file name/path as needed.
IMAGE_PATH = Path(__file__).with_name("image.cv.webp")
img = cv2.imread(str(IMAGE_PATH), 0)

if img is None:
    raise FileNotFoundError(f"Could not load image from: {IMAGE_PATH}")

#contrast stretching

min_val, max_val = np.min(img), np.max(img)
if max_val == min_val:
    # Avoid division by zero for constant-intensity images.
    cs_image = np.zeros_like(img, dtype=np.uint8)
else:
    cs_image = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

#histogram equalization
he_img = cv2.equalizeHist(img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1);plt.imshow(img, cmap='gray');plt.title('Original Image');plt.axis('off')
plt.subplot(1, 3, 2);plt.imshow(cs_image, cmap='gray');plt.title('Contrast Stretched Image');plt.axis('off')
plt.subplot(1, 3, 3);plt.imshow(he_img, cmap='gray');plt.title('Histogram Equalized Image');plt.axis('off')
plt.show() 
