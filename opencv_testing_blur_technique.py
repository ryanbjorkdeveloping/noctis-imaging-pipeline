import cv2
import numpy as np

image = cv2.imread("input_images/rubin_img_sample.webp")
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Galaxy detection ---
# Blur heavily to suppress stars, leaving only large diffuse structures
galaxy_blur = cv2.GaussianBlur(gray, (51, 51), 0)
_, galaxy_thresh = cv2.threshold(galaxy_blur, 30, 255, cv2.THRESH_BINARY)

galaxy_contours, _ = cv2.findContours(galaxy_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

galaxy_mask = np.zeros_like(gray)
galaxies = []

for contour in galaxy_contours:
    area = cv2.contourArea(contour)
    if area < 500:  # galaxies are large blobs
        continue

    # Circularity filter: galaxies are roundish/elliptical blobs
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < 0.1:  # skip thin streaks/noise
        continue

    galaxies.append(contour)
    cv2.drawContours(galaxy_mask, [contour], -1, 255, -1)

    # Draw galaxy highlight (blue)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.putText(output, "Galaxy", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    print(f"Galaxy: x={x}, y={y}, w={w}, h={h}, area={area:.1f}px")

# --- Star detection (background regions only) ---
# Remove galaxy regions from consideration for global star pass
non_galaxy_mask = cv2.bitwise_not(galaxy_mask)

star_blur = cv2.GaussianBlur(gray, (3, 3), 0)
_, star_thresh_global = cv2.threshold(star_blur, 30, 255, cv2.THRESH_BINARY)
star_thresh_bg = cv2.bitwise_and(star_thresh_global, non_galaxy_mask)

bg_contours, _ = cv2.findContours(star_thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
star_count = 0

for contour in bg_contours:
    area = cv2.contourArea(contour)
    if area < 2 or area > 300:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    star_count += 1

print(f"Background stars detected: {star_count}")

# --- Star detection inside galaxies (local normalization) ---
for i, contour in enumerate(galaxies):
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the galaxy region
    roi_gray = gray[y:y+h, x:x+w]
    roi_mask = galaxy_mask[y:y+h, x:x+w]

    # Estimate galaxy glow by blurring heavily, then subtract it
    galaxy_glow = cv2.GaussianBlur(roi_gray, (51, 51), 0)
    # Avoid underflow when subtracting
    local_norm = cv2.subtract(roi_gray, galaxy_glow)

    # Threshold on the residual (stars are bright relative to local glow)
    _, local_thresh = cv2.threshold(local_norm, 15, 255, cv2.THRESH_BINARY)
    # Only look inside the galaxy contour
    local_thresh = cv2.bitwise_and(local_thresh, roi_mask)

    inner_contours, _ = cv2.findContours(local_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_star_count = 0

    for contour_inner in inner_contours:
        area = cv2.contourArea(contour_inner)
        if area < 2 or area > 200:
            continue
        ix, iy, iw, ih = cv2.boundingRect(contour_inner)
        # Translate back to full image coordinates
        cv2.rectangle(output, (x + ix, y + iy), (x + ix + iw, y + iy + ih), (0, 255, 255), 1)
        inner_star_count += 1

    print(f"Galaxy {i} interior stars detected: {inner_star_count}")

cv2.imwrite("output_images/rubin_img_sample_detected.jpg", output)
print("Saved to rubin_img_sample_detected.jpg")
