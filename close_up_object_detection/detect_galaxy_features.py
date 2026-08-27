#Brainstorming different features of a galaxy:
### bright center, fading on the outskirts (soft, gradual falloff. radial profile)
### large angular size of a galaxy and diffuseness. large ratio. 
### elliptical, not circular, shape
### no hard limb nor sharp edge

# WEIGHTS from strongest to weakest:
### 1) detect_large_angular_size
### 2) detect_bright_center
### 3) detect_soft_edge
### 4) detect_elliptical_shape


import os
import sys

import cv2
import numpy as np

class DetectGalaxyFeatures:

    #global variables
    BLUR_AMOUNT = 51
    MIN_AREA = 500
    THRESHOLD = 30
    SIZE_FULL = 0.15
    MIN_FRAME_FRAC = 0.15
    MAX_FRAME_FRAC = 0.7

    #global variables that detect_ratio_diffusion method is working with
    LOW_THRESH = 30
    HIGH_THRESH = 120
    DIFFUSE_FULL = 4.0

    #global variables that detect_bright_center method is working with
    N_RINGS = 8

    #global variables that detect_elliptical_shape is working with
    ELLIPSE_FULL = 2.0

    #global variabels that detect_soft_edge is using
    DROP_FULL = 80
    EDGE_KERNEL = 9

    #global variables used to help with scoring properly. Different weights used in scoring
    W_SIZE = 0.15
    W_CENTER = 0.65 
    W_ELLIPSE = 0.05
    W_SOFT = 0.15
    GALAXY_THRESHOLD = 0.80

    # Defining everything this function will be testing as well as transporting it to other parts of program.
    def __init__ (self, image_path) :
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None :
            raise FileNotFoundError(f"Could not read {image_path}")
        
        #Grayscale copy of my image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        #parameters detection functions will be testing out
        self.current_mask = None
        self.bbox = None
        self.center = None
        self.major = 0.0
        self.minor = 0.0
        self.angle = 0.0

        #set up candidates
        self.candidates = self._detect_candidate_disks()

    def _detect_candidate_disks (self):
        #use the blur technique in order to find and detect what could be considered as a galaxy or not
        BLUR_AMOUNT = self.BLUR_AMOUNT
        THRESHOLD = self.THRESHOLD
        MIN_AREA = self.MIN_AREA

        blurred = cv2.GaussianBlur(self.gray, (BLUR_AMOUNT, BLUR_AMOUNT), 0)

        _, thresh = cv2.threshold(blurred, THRESHOLD, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            if len(c) < 5:
                continue
            ellipse = cv2.fitEllipse(c)
            (cx, cy), (maj, minr), ang = ellipse
            ell_area = np.pi * (maj / 2) * (minr / 2)
            frac = ell_area / (self.gray.shape[0] * self.gray.shape[1])
            if frac < self.MIN_FRAME_FRAC or frac > self.MAX_FRAME_FRAC:
                continue
            candidates.append(ellipse)

        return candidates

    def _set_current_object (self, ellipse):
        # give values to everything all the other detections are going to use to detect whether a candidate is a galaxy or not
        
        #unpack ellipse
        (cx, cy), (major, minor), angle = ellipse

        #create the mask
        mask = np.zeros_like(self.gray)
        cv2.ellipse(mask, ellipse, 255, -1)
        self.current_mask = mask

        #create different ellipses that were unpacked
        self.center = (cx, cy)
        self.major = major
        self.minor = minor
        self.angle = angle

        #create bounding box around image
        self.bbox = (int(cx - major/2), int(cy-minor/2), int(major), int(minor))

    #how big and spread out light is from its core
    def detect_ratio_diffusion (self):

        LOW_THRESH = self.LOW_THRESH
        HIGH_THRESH = self.HIGH_THRESH
        DIFFUSE_FULL = self.DIFFUSE_FULL

        if self.current_mask is None:
            return 0.0


        #booleans for what can be classified as a lower_area or high_area for pixels
        low_area = ((self.gray > LOW_THRESH) & (self.current_mask == 255)).sum()
        high_area = ((self.gray > HIGH_THRESH) & (self.current_mask == 255)).sum()

        # guard ratio if high_area == 0. else, continue with ratio calculation of said image
        if high_area == 0:
            ratio = 0.0
        else:
            ratio = low_area / high_area

        #map ratio
        ratio_score = (ratio - 1.0) / (self.DIFFUSE_FULL - 1.0)
        return float(np.clip(ratio_score,0.0, 1.0))


    #measuring the radial falloff from the center that's supposed to be super bright when detecting whether something is a galaxy or not
    def detect_bright_center (self):
        N_RINGS = self.N_RINGS

        #guard check
        if self.current_mask is None or self.center is None:
            return 0.0
        
        #building a distance map of how far each pixel is from center
        cx, cy = self.center
        ys, xs = np.indices(self.gray.shape)
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)

        #figuring out object's radius
        inside = self.current_mask == 255
        max_r = dist[inside].max()
        if max_r == 0:
            return 0.0
        
        #average brightness in each ring
        ring_means = []
        for k in range(self.N_RINGS):
            lo = max_r * k / self.N_RINGS
            hi = max_r * (k+1) / self.N_RINGS
            band = inside & (dist >= lo) & (dist < hi)
            if band.sum() == 0:
                continue
            ring_means.append(self.gray[band].mean())

        # compare brightness in each ring and calculate monotonic falloff to determine whether candidate is galaxy or not
        if len(ring_means) < 2:
            return 0.0
        drops = sum(
            1 for i in range(len(ring_means) - 1)
            if ring_means[i+1] <= ring_means[i]
        )
        score = drops / (len(ring_means) - 1)
        return float(score)

    #finding if shape is eliptical or not. NOTE: using max/min in order to find it, since major or minor could be bigger than the other
    def detect_elliptical_shape (self):
        
        ELLIPSE_FULL = self.ELLIPSE_FULL

        #safety guard
        if self.major == 0 or self.minor == 0:
            return 0.0

        longer = max(self.major, self.minor)
        shorter = min(self.major, self.minor)
        ratio = longer / shorter

        # ratio 1.0 = perfect circle (score 0), bigger = more elliptical (score up)
        score = (ratio - 1.0) / (ELLIPSE_FULL - 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def detect_soft_edge (self):

        #guard
        if self.current_mask is None:
            return 0.0

        #set kernel up (defines length of pixels object grows/shrinks by)
        kernel = np.ones((self.EDGE_KERNEL, self.EDGE_KERNEL), np.uint8)

        # getting inner and outer masks
        inside_mask = self.current_mask
        outer = cv2.dilate(inside_mask, kernel)
        inner = cv2.erode(inside_mask, kernel)

        # create two edge rings using boolean logic
        orig = self.current_mask == 255
        inner_b = inner == 255
        outer_b = outer == 255

        inner_ring = orig & ~inner_b
        outer_ring = outer_b & ~orig

        #getting average brightness of each ring
        if inner_ring.sum() == 0 or outer_ring.sum() == 0:
            return 0.0
        inner_bright = self.gray[inner_ring].mean()
        outer_bright = self.gray[outer_ring].mean()

        #drop in brightness from inner ring to outer ring
        drop = inner_bright - outer_bright
        soft = 1.0 - np.clip(drop / self.DROP_FULL, 0.0, 1.0)
        return float(soft)


    def galaxy_scoring (self):
        W_SIZE = self.W_SIZE
        W_CENTER = self.W_CENTER
        W_ELLIPSE = self.W_ELLIPSE
        W_SOFT = self.W_SOFT
        GALAXY_THRESHOLD = self.GALAXY_THRESHOLD

        #putting each feature into a dictionary
        features = {
            "ratio_diffusion": self.detect_ratio_diffusion(),
            "bright_center": self.detect_bright_center(),
            "elliptical_shape": self.detect_elliptical_shape(),
            "soft_edge": self.detect_soft_edge()
        }

        #combining features using a weighted sum
        galaxy_score = (
            self.W_SIZE * features["ratio_diffusion"]
            + self.W_CENTER * features["bright_center"]
            + self.W_ELLIPSE * features["elliptical_shape"]
            + self.W_SOFT * features["soft_edge"]
        )
        galaxy_score = float(np.clip(galaxy_score, 0.0, 1.0))

        #label and returning a dictionary of result
        is_galaxy = galaxy_score >= self.GALAXY_THRESHOLD
        return {
            "features": features,
            "galaxy_score": galaxy_score,
            "label": "Galaxy" if is_galaxy else None
        }

    #produces results with bbox and label and scoring and all that stuff
    def scoring_all(self):
        
        results = []

        for ellipse in self.candidates:
            self._set_current_object(ellipse)
            result = self.galaxy_scoring()
            result["bbox"] = self.bbox
            results.append(result)
        return results

    #physically draws the result and everything in the actual image that the user sees. what makes the code results user friendly hehe
    def draw_results(self, results, output_dir="output_images"):
        
        #draw copy of the output
        output = self.image.copy()

        #looping results, skipping non-galaxies with bounding boxes
        for result in results:
            if result["label"] != "Galaxy":
                continue

            #finding and labeling where to draw the actual box
            x, y, w, h = result["bbox"]
            cv2.rectangle(
                output, (x,y), (x+w, y+h),
                (255, 255, 0), 3
            )
            text = f"Galaxy {result['galaxy_score']:.2f}"

            #labeling above the box
            cv2.putText(
                output, text, (x, max(y-10, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

        #make sure folder exists
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_galaxy_detected.jpg")
        cv2.imwrite(output_path, output)
        return output_path

        

if __name__ == "__main__":
    d = DetectGalaxyFeatures("input_images/single_isolated_galaxy.jpeg")
    print(f"Found {len(d.candidates)} candidates(s)")