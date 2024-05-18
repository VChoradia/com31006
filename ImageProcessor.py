# ImageProcessor.py
# Author: Vivek Choradia

import cv2
import numpy as np
import time
from PIL import Image


class ImageProcessor:
    def __init__(self, image_path, scale_factor=0.4):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = self.load_image()
        self.gray_image = self.convert_to_gray()
        self.keypoints = None
        self.descriptors = None

    def get_pil_image(self):
        # Convert the OpenCV image to a PIL image
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return pil_image

    def load_image(self):
        image = cv2.imread(self.image_path)
        if self.scale_factor != 1:
            image = self.scale_image(image)
        return image

    def convert_to_gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def scale_image(self, image):
        # Resize the image according to the scale factor
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dimensions = (width, height)
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    def apply_harris_corners(self):
        start_time = time.time()
        gray = np.float32(self.gray_image)
        dst = cv2.cornerHarris(gray, 2, 5, 0.07)
        dst = cv2.dilate(dst, None)
        self.image[dst > 0.01 * dst.max()] = [0, 0, 255]
        end_time = time.time()
        print(f"Harris Corner Detection took {end_time - start_time:.2f} seconds")

    def apply_sift_features(self):
        start_time = time.time()
        sift = cv2.SIFT_create()
        self.keypoints, self.descriptors = sift.detectAndCompute(self.gray_image, None)
        if self.keypoints is None or self.descriptors is None:
            raise ValueError("SIFT feature detection failed.")
        self.image = cv2.drawKeypoints(self.image,
                                       self.keypoints,
                                       None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        end_time = time.time()
        print(f"SIFT Detection took {end_time - start_time:.2f} seconds")
        return self.keypoints, self.descriptors

    def apply_orb_features(self):
        start_time = time.time()
        orb = cv2.ORB_create()
        # Detect keypoints and compute the ORB descriptors
        self.keypoints, self.descriptors = orb.detectAndCompute(self.gray_image, None)
        if self.keypoints is None or self.descriptors is None:
            raise ValueError("ORB feature detection failed.")
        self.image = cv2.drawKeypoints(self.image,
                                       self.keypoints,
                                       None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        end_time = time.time()
        print(f"ORB Detection took {end_time - start_time:.2f} seconds")
        return self.keypoints, self.descriptors

    def match_features_ssd(self, other):
        """Match features using Sum of Squared Differences (SSD), calculated manually."""
        if self.keypoints is None or self.descriptors is None or other.keypoints is None or other.descriptors is None:
            raise ValueError("Keypoints and descriptors must be detected before matching.")
        start_time = time.time()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.descriptors, other.descriptors)
        print(f"Initial matches found: {len(matches)}")  # Debug: print initial match count

        # Manually squaring the L2 distances to get SSD
        for match in matches:
            match.distance **= 2  # Squaring the distance to conform to SSD

        matches = sorted(matches, key=lambda x: x.distance)

        print(f"Matches after filtering: {len(matches)}")  # Debug: print filtered match count
        if not matches:
            print("No good matches found. Adjusting parameters or retrying may be necessary.")
            return None

        end_time = time.time()

        matched_image = cv2.drawMatches(self.image, self.keypoints, other.image, other.keypoints, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        num_matches = len(matches)
        matching_accuracy = num_matches / min(len(self.keypoints), len(other.keypoints))
        computational_time = end_time - start_time

        return matched_image, num_matches, matching_accuracy, computational_time

    def match_features_ratio(self, other):
        """Match features using the Ratio Test."""
        if self.keypoints is None or self.descriptors is None or other.keypoints is None or other.descriptors is None:
            raise ValueError("Keypoints and descriptors must be detected before matching.")
        start_time = time.time()
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.descriptors, other.descriptors, k=2)

        # Apply ratio test
        good_matches = []
        ratio_threshold = 0.75  # Typical threshold value
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        end_time = time.time()

        # Draw matches on the image
        matched_image = cv2.drawMatches(self.image, self.keypoints, other.image, other.keypoints, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Calculate performance metrics
        num_matches = len(good_matches)
        matching_accuracy = num_matches / min(len(self.keypoints), len(other.keypoints))
        computational_time = end_time - start_time

        return matched_image, num_matches, matching_accuracy, computational_time

    def display_features(self, method_name):
        cv2.imshow(method_name + ' Features', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
