# ImageProcessor.py
# Author: Vivek Choradia

import cv2
import numpy as np
import time
from PIL import Image


class ImageProcessor:
    """
    Processes images for feature detection, description, and matching.

    Attributes:
        image_path (str): Path to the image file.
        scale_factor (float): Factor to scale the image.
        image (np.array): Loaded and possibly resized image.
        gray_image (np.array): Grayscale version of the image.
        keypoints (list): Detected keypoints in the image.
        descriptors (np.array): Descriptors of the keypoints.
    """

    def __init__(self, image_path, scale_factor=1.0):
        """
        Initializes the ImageProcessor with a path to the image and a scaling factor.

        Parameters:
            image_path (str): Path to the image file.
            scale_factor (float, optional): Factor by which to scale the image. Default is 1.0 (no scaling).
        """

        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = self.load_image()
        self.gray_image = self.convert_to_gray_and_enhance()
        self.keypoints = None
        self.descriptors = None

    def get_pil_image(self):
        """
        Converts the OpenCV image to a PIL image for easier manipulation and display.

        Returns:
            Image: The converted PIL image.
        """

        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return pil_image

    def load_image(self):
        """
        Loads an image from the specified path and scales it according to the scale factor.

        Returns:
            np.array: The loaded and scaled image.
        """

        image = cv2.imread(self.image_path)
        if self.scale_factor != 1:
            image = self.scale_image(image)
        return image

    def convert_to_gray_and_enhance(self):
        """
        Converts the image to grayscale and applies histogram equalization to enhance it.

        Returns:
            np.array: The enhanced grayscale image.
        """

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.equalizeHist(gray_image)
        return enhanced_image

    def scale_image(self, image):
        """
        Scales the image using the specified scale factor.

        Parameters:
            image (np.array): The original image to scale.

        Returns:
            np.array: The scaled image.
        """

        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dimensions = (width, height)
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    def apply_harris_corners(self):
        """
        Applies Harris corner detection to the image and annotates the corners.

        Outputs the number of corners detected and the time taken for the detection.
        """

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
        """
        Applies SIFT to detect and compute keypoints and descriptors in the image.

        Parameters:
            match (bool): If True, does not draw keypoints on the image.

        Returns:
            tuple: A tuple containing keypoints, descriptors, and the time taken for detection.
        """

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
        """
        Applies ORB to detect and compute keypoints and descriptors in the image.

        Parameters:
            match (bool): If True, does not draw keypoints on the image.

        Returns:
            tuple: A tuple containing keypoints, descriptors, and the time taken for detection.
        """

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
        """
        Matches features between this image and another using the ratio test for robust matching.

        Parameters:
            other (ImageProcessor): The other image processor instance to match features with.
            ratio_threshold (float): The threshold to apply in the ratio test to determine good matches.

        Returns:
            tuple: Returns a tuple containing the matched image, good matches, matching accuracy, and computation time.
        """

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

def stitch_images(processor1, processor2, matches):

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = processor1.keypoints[match.queryIdx].pt
        points2[i, :] = processor2.keypoints[match.trainIdx].pt

    # Compute homography matrix using RANSAC
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    if H is None:
        print("Homography could not be computed.")
        return

    # Warp and stitch images
    height, width, channels = processor1.image.shape
    warped_image = cv2.warpPerspective(processor2.image, H, (width + processor2.image.shape[1], height))
    stitched_image = np.copy(warped_image)
    stitched_image[0:height, 0:width] = processor1.image

    # Resize stitched image
    display_width = 600
    aspect_ratio = display_width / float(stitched_image.shape[1])
    display_height = int(stitched_image.shape[0] * aspect_ratio)
    resized_image = cv2.resize(stitched_image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    return resized_image

