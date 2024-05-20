# ImageProcessor.py
# Author: Vivek Choradia

import cv2
import numpy as np
import time
from PIL import Image


class ImageProcessor:
    def __init__(self, image_path, scale_factor=1.0):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = self.load_image()
        self.gray_image = self.convert_to_gray_and_enhance()
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

    def convert_to_gray_and_enhance(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        enhanced_image = cv2.equalizeHist(gray_image)
        return enhanced_image

    def scale_image(self, image):
        # Resize the image according to the scale factor
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dimensions = (width, height)
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    def apply_harris_corners(self):

        threshold = 0.01

        start_time = time.time()

        gray = np.float32(self.gray_image)
        dst = cv2.cornerHarris(gray, 2, 5, 0.06)
        dst = cv2.dilate(dst, None)
        corners = dst > threshold * dst.max()
        self.image[corners] = [0, 0, 255]

        # Counting the number of corners detected
        corner_count = np.sum(corners)
        print(f"Number of Harris corners detected: {corner_count}")

        end_time = time.time()
        print(f"Harris Corner Detection took {end_time - start_time:.2f} seconds")

    def apply_sift_features(self, match=False):
        start_time = time.time()

        sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.08, edgeThreshold=10)
        self.keypoints, self.descriptors = sift.detectAndCompute(self.gray_image, None)
        if not self.keypoints:
            print("No keypoints detected.")
            return []

        sift_time = time.time() - start_time

        print(f"SIFT Detection took {sift_time:.4f} seconds")
        print(f"Number of SIFT keypoints detected: {len(self.keypoints)}")
        print(f"SIFT Descriptor Size: {self.descriptors.shape[1] if self.descriptors is not None else 'N/A'}")

        if not match:
            self.image = cv2.drawKeypoints(self.image, self.keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return self.keypoints, self.descriptors, sift_time

    def apply_orb_features(self, match=False):
        start_time = time.time()

        orb = cv2.ORB_create()
        self.keypoints, self.descriptors = orb.detectAndCompute(self.gray_image, None)
        if self.keypoints is None or self.descriptors is None:
            raise ValueError("ORB feature detection failed.")

        orb_time = time.time() - start_time
        print(f"ORB Detection took {orb_time:.4f} seconds")
        print(f"Number of ORB keypoints detected: {len(self.keypoints)}")
        print(f"ORB Descriptor Size: {self.descriptors.shape[1] if self.descriptors is not None else 'N/A'}")

        if not match:
            self.image = cv2.drawKeypoints(self.image, self.keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return self.keypoints, self.descriptors, orb_time

    def match_features(self, other, ratio_threshold=0.75):
        if self.keypoints is None or self.descriptors is None or other.keypoints is None or other.descriptors is None:
            raise ValueError("Keypoints and descriptors must be detected before matching.")

        start_time = time.time()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Match descriptors using knn method
        matches = bf.knnMatch(self.descriptors, other.descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        match_time = time.time() - start_time

        matched_image = cv2.drawMatches(self.image, self.keypoints, other.image, other.keypoints, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_DEFAULT)

        num_matches = len(good_matches)
        matching_accuracy = num_matches / min(len(self.keypoints), len(other.keypoints))

        return matched_image, good_matches, matching_accuracy, match_time

    def match_features_ssd(self, other):
        """Match features using Sum of Squared Differences (SSD), calculated manually."""

        if self.keypoints is None or self.descriptors is None or other.keypoints is None or other.descriptors is None:
            raise ValueError("Keypoints and descriptors must be detected before matching.")
        start_time = time.time()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.descriptors, other.descriptors)
        print(f"Initial matches found: {len(matches)}")

        for match in matches:
            match.distance **= 2

        matches = sorted(matches, key=lambda x: x.distance)

        print(f"Matches after filtering: {len(matches)}")
        if not matches:
            print("No good matches found. Adjusting parameters or retrying may be necessary.")
            return None

        end_time = time.time()

        matched_image = cv2.drawMatches(self.image, self.keypoints, other.image, other.keypoints, matches, None,
                                        flags=cv2.DrawMatchesFlags_DEFAULT)

        num_matches = len(matches)
        matching_accuracy = num_matches / min(len(self.keypoints), len(other.keypoints))
        computational_time = end_time - start_time

        return matched_image, matches, matching_accuracy, computational_time

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
                                        flags=cv2.DrawMatchesFlags_DEFAULT)

        # Calculate performance metrics
        num_matches = len(good_matches)
        matching_accuracy = num_matches / min(len(self.keypoints), len(other.keypoints))
        computational_time = end_time - start_time

        return matched_image, matches, matching_accuracy, computational_time

    def display_features(self, method_name):
        cv2.imshow(method_name + ' Features', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
