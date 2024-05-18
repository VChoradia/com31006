# FeatureDetectionUI.py
# Author: Vivek Choradia

import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from utils import image_pairs
from ImageProcessor import ImageProcessor
import time


class FeatureDetectionUI:
    def __init__(self, root):
        self.root = root
        self.image_processors = []  # Store ImageProcessor instances for each image
        self.selected_option = tk.StringVar()  # To display the currently selected option
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Feature Detection and Image Stitching")

        # Maximize the window on startup
        self.root.state('zoomed')

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        # Sidebar frame for options
        sidebar_frame = tk.Frame(main_frame, width=200, bg='lightgrey')
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Main display area for images
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # Dropdown for image pair selection
        self.selected_pair = tk.StringVar(value=list(image_pairs.keys())[0])
        dropdown = ttk.Combobox(sidebar_frame, textvariable=self.selected_pair, values=list(image_pairs.keys()))
        dropdown.pack(pady=10)
        dropdown.bind("<<ComboboxSelected>>", self.load_selected_pair)

        # Label to display the currently selected option
        self.option_label = tk.Label(sidebar_frame, textvariable=self.selected_option, bg='lightgrey',
                                     font=("Helvetica", 14))
        self.option_label.pack(pady=10)

        # Add navigation options in the sidebar
        tk.Label(sidebar_frame, text="Options", bg='lightgrey', font=("Helvetica", 14)).pack(pady=10)

        self.option_frame = tk.Frame(sidebar_frame, bg='lightgrey')
        self.option_frame.pack(fill=tk.Y, expand=1)

        # Placeholder label
        self.placeholder_label = tk.Label(display_frame, text="Select an option from the sidebar",
                                          font=("Helvetica", 14))
        self.placeholder_label.pack(pady=20)

        # Image display area
        self.image_label = tk.Label(display_frame)
        self.image_label.pack()

    def load_selected_pair(self, event=None):
        # Clear the previous ImageProcessor instances
        self.image_processors = []

        # Load the new image pair based on the selection
        pair_name = self.selected_pair.get()
        image_paths = image_pairs[pair_name]

        for path in image_paths:
            # Create ImageProcessor instances for each image in the pair
            processor = ImageProcessor(path, scale_factor=0.5)  # Adjust scale_factor as needed
            self.image_processors.append(processor)

        # Display the loaded images side by side
        self.display_images()

        # Set up the options in the sidebar once images are loaded
        self.setup_options()

    def setup_options(self):
        for widget in self.option_frame.winfo_children():
            widget.destroy()

        button_width = 20

        # Feature Detection Options
        tk.Label(self.option_frame, text="1. Feature Detection", bg='lightgrey').pack(anchor=tk.W, padx=10)
        tk.Button(self.option_frame, text='Harris Corner', width=button_width,
                  command=lambda: self.select_option('Harris Corner Detection')).pack(anchor=tk.W, padx=20, pady=5)
        tk.Button(self.option_frame, text='SIFT', width=button_width,
                  command=lambda: self.select_option('SIFT Detection')).pack(anchor=tk.W, padx=20, pady=5)

        # Feature Description Options
        tk.Label(self.option_frame, text="2. Feature Description", bg='lightgrey').pack(anchor=tk.W, padx=10)
        tk.Button(self.option_frame, text='SIFT', width=button_width,
                  command=lambda: self.select_option('SIFT Description')).pack(anchor=tk.W, padx=20, pady=5)
        tk.Button(self.option_frame, text='ORB', width=button_width,
                  command=lambda: self.select_option('ORB Description')).pack(anchor=tk.W, padx=20, pady=5)

        # Feature Matching Option
        tk.Label(self.option_frame, text="3. Feature Matching", bg='lightgrey').pack(anchor=tk.W, padx=10)
        tk.Button(self.option_frame, text='Match Features', width=button_width,
                  command=lambda: self.select_option('Feature Matching')).pack(anchor=tk.W, padx=20, pady=5)

        # Image Stitching Option
        tk.Label(self.option_frame, text="4. Image Stitching", bg='lightgrey').pack(anchor=tk.W, padx=10)
        tk.Button(self.option_frame, text='Stitch Images', width=button_width,
                  command=lambda: self.select_option('Image Stitching')).pack(anchor=tk.W, padx=20, pady=5)

    def display_images(self):
        # Combine images side by side with scaling
        scale_factor = 0.7  # Adjust scale factor as needed
        images = [processor.get_pil_image() for processor in self.image_processors]
        images = [i.resize((int(i.width * scale_factor), int(i.height * scale_factor)), Image.Resampling.LANCZOS) for i
                  in images]
        total_width = sum(i.width for i in images)
        max_height = max(i.height for i in images)
        combined_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            combined_image.paste(im, (x_offset, 0))
            x_offset += im.width
        self.display_image(combined_image)

    def display_image(self, image):
        # Convert PIL Image to ImageTk and display it
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def select_option(self, option):
        try:
            self.selected_option.set(f"Selected Option: \n {option}")
            if 'Detection' in option:
                method = option.split()[0]
                self.detect_features(method)
            elif 'Description' in option:
                method = option.split()[0]
                self.describe_features(method)
            elif option == 'Feature Matching':
                self.match_features()
            elif option == 'Image Stitching':
                self.stitch_images()
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))

    def detect_features(self, method):
        self.load_selected_pair()
        for processor in self.image_processors:
            if method == 'Harris':
                processor.apply_harris_corners()
            elif method == 'SIFT':
                processor.apply_sift_features()
        self.display_images()

    def describe_features(self, method):
        self.load_selected_pair()
        for processor in self.image_processors:
            if method == 'SIFT':
                keypoints, descriptors = processor.apply_sift_features()
            elif method == 'ORB':
                keypoints, descriptors = processor.apply_orb_features()
        self.display_images()

    def match_features(self):
        self.load_selected_pair()  # Ensure images are loaded

        if len(self.image_processors) != 2:
            print("A valid image pair must contain exactly two images.")
            return

        processor1, processor2 = self.image_processors

        # Ensure keypoints and descriptors are computed for both images
        if not processor1.keypoints or not processor1.descriptors:
            processor1.apply_sift_features()  # or apply_harris_corners() or another method based on your setup

        if not processor2.keypoints or not processor2.descriptors:
            processor2.apply_sift_features()  # or apply_harris_corners() or another method based on your setup

        # Now proceed with matching
        matches_img_ssd, num_matches_ssd, matching_accuracy_ssd, computational_time_ssd = processor1.match_features_ssd(
            processor2)

        def resize_image(image, width):
            aspect_ratio = width / float(image.shape[1])
            dim = (width, int(image.shape[0] * aspect_ratio))
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        processor1.image = resize_image(processor1.image, 600)
        processor2.image = resize_image(processor2.image, 600)

        # Match features using SSD
        matches_img_ssd, num_matches_ssd, matching_accuracy_ssd, computational_time_ssd = processor1.match_features_ssd(
            processor2)

        # Match features using Ratio Test
        matches_img_ratio, num_matches_ratio, matching_accuracy_ratio, computational_time_ratio = processor1.match_features_ratio(
            processor2)

        # Display SSD matches
        matches_img_ssd_pil = Image.fromarray(cv2.cvtColor(matches_img_ssd, cv2.COLOR_BGR2RGB))
        self.display_image(matches_img_ssd_pil)

        # Save results for report
        matches_img_ratio_pil = Image.fromarray(cv2.cvtColor(matches_img_ratio, cv2.COLOR_BGR2RGB))
        matches_img_ssd_pil.save("feature_matching_sift.png")
        matches_img_ratio_pil.save("feature_matching_orb.png")

        # Print performance metrics for report
        print(
            f"SSD Matching: Number of Matches = {num_matches_ssd}, Matching Accuracy = {matching_accuracy_ssd:.2f}, Computational Time = {computational_time_ssd:.2f} s")
        print(
            f"Ratio Test Matching: Number of Matches = {num_matches_ratio}, Matching Accuracy = {matching_accuracy_ratio:.2f}, Computational Time = {computational_time_ratio:.2f} s")

    def stitch_images(self):
        self.load_selected_pair()
        if len(self.image_processors) != 2:
            print("A valid image pair must contain exactly two images.")
            return

        processor1, processor2 = self.image_processors

        # Resize images for processing speed and consistency
        def resize_image(image, width):
            aspect_ratio = width / float(image.shape[1])
            dim = (width, int(image.shape[0] * aspect_ratio))
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        processor1.image = resize_image(processor1.image, 600)
        processor2.image = resize_image(processor2.image, 600)

        # Apply SIFT and match features using SSD
        keypoints1, descriptors1 = processor1.apply_sift_features()
        keypoints2, descriptors2 = processor2.apply_sift_features()

        # Use SSD or Ratio Test for matching as per your choice here
        match_result = processor1.match_features_ssd(processor2)
        if not match_result:
            print("Feature matching did not return valid matches.")
            return

        matched_image, matches, matching_accuracy, computational_time = match_result

        # Prepare for homography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("Homography could not be computed.")
            return

        # Warp the second image
        height1, width1 = processor1.image.shape[:2]
        warped_image = cv2.warpPerspective(processor2.image, H, (width1 + processor2.image.shape[1], height1))

        # Create the stitched output
        stitched_image = np.zeros((max(height1, warped_image.shape[0]), width1 + warped_image.shape[1], 3),
                                  dtype=np.uint8)
        stitched_image[:height1, :width1] = processor1.image
        for y in range(warped_image.shape[0]):
            for x in range(warped_image.shape[1]):
                if np.all(stitched_image[y, width1 + x] == 0):
                    stitched_image[y, width1 + x] = warped_image[y, x]
                else:
                    stitched_image[y, width1 + x] = cv2.addWeighted(stitched_image[y, width1 + x], 0.5,
                                                                    warped_image[y, x], 0.5, 0)

        # Convert to PIL Image for display
        stitched_image_pil = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
        self.display_image(stitched_image_pil)

