import cv2
import numpy as np

image = cv2.imread("input_images/rubin_img_sample.webp")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold: anything brighter than ~20 brightness is "not black"
_, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    
    if area < 5:  # skip tiny noise pixels
        continue
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Object {i}: x={x}, y={y}, w={w}, h={h}, area={area:.1f}px")

cv2.imwrite("output_images/rubin_img_sample.webp", image)
print("Saved to rubin_img_sample.webp")
