import cv2
import numpy as np
import os

# 1. Load the image in grayscale
image_path = 'rtest8boom.jpeg'

# Safety check: Does the file even exist?
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found in this directory!")
else:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Safety check: Did OpenCV successfully read it?
    if img is None:
        print("Error: OpenCV couldn't read the image data. The file might be corrupted.")
    else:
        # 2. Apply a Gaussian blur
        blurred = cv2.GaussianBlur(img, (9, 9), 0)

        # 3. Use thresholding to bring back crisp contrast
        _, smoothed_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # 4. Save the clean result
        cv2.imwrite('smoothed_lines.png', smoothed_img)
        print("Success! 'smoothed_lines.png' has been created.")