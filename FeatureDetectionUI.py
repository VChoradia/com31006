# FeatureDetectionUI.py
# Author: Vivek Choradia

import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from utils import image_pairs
from ImageProcessor import ImageProcessor, stitch_images
import time


class FeatureDetectionUI:
    """
        A GUI application for feature detection and image stitching.

        Attributes:
            root (tk.Tk): The main window of the application.
            image_processors (list): A list of ImageProcessor instances for image operations.
            selected_option (tk.StringVar): String variable to display the selected option in the GUI.
        """

    def __init__(self, root):
        """
        Initializes the FeatureDetectionUI class with the main Tkinter root window.

        Parameters:
            root (tk.Tk): The main window of the application.
        """

        self.root = root
        self.image_processors = []  # Store ImageProcessor instances for each image
        self.selected_option = tk.StringVar()  # To display the currently selected option
        self.setup_ui()

    def setup_ui(self):
        """Sets up the user interface for the application."""

        self.root.title("Feature Detection and Image Stitching")
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
        """
        Loads the selected image pair for processing.

        Parameters:
            event (Event, optional): The event triggered by selecting an image pair from the dropdown.
        """

        self.image_processors = []  # Clear the previous ImageProcessor instances

        pair_name = self.selected_pair.get()
        image_paths = image_pairs[pair_name]

        for path in image_paths:
            if event is not None:
                processor = ImageProcessor(path)
            else:
                processor = ImageProcessor(path, scale_factor=0.5)
            self.image_processors.append(processor)

        self.display_images()

        self.setup_options()

    def setup_options(self):
        """Sets up interactive buttons for various image processing options in the GUI."""

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

        # Adding buttons for SIFT and ORB matching options
        tk.Label(self.option_frame, text="3. Feature Matching Options", bg='lightgrey').pack(anchor=tk.W,
                                                                                                        padx=10)
        tk.Button(self.option_frame, text='Match using SIFT', width=button_width,
                  command=lambda: self.select_option('Match using SIFT')).pack(anchor=tk.W, padx=20, pady=5)
        tk.Button(self.option_frame, text='Match using ORB', width=button_width,
                  command=lambda: self.select_option('Match using ORB')).pack(anchor=tk.W, padx=20, pady=5)

        # Image Stitching Option
        tk.Label(self.option_frame, text="4. Image Stitching", bg='lightgrey').pack(anchor=tk.W, padx=10)
        tk.Button(self.option_frame, text='Stitch Images', width=button_width,
                  command=lambda: self.select_option('Stitch Images')).pack(anchor=tk.W, padx=20, pady=5)

    def display_images(self):
        """Displays the loaded images on the GUI."""

        scale_factor = 0.5
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
        """
        Helper function to display an image in the GUI.

        Parameters:
            image (Image): The image to be displayed.
        """

        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def select_option(self, option):
        """
        Handles option selection from the GUI and triggers appropriate methods.

        Parameters:
            option (str): The selected operation to perform on the images.
        """

        self.selected_option.set(f"Selected Option: \n {option}")
        if 'Detection' in option:
            method = option.split()[0]
            self.detect_features(method)
        elif 'Description' in option:
            method = option.split()[0]
            self.describe_features(method)
        elif 'Match using SIFT' in option:
            self.match_features(descriptor="sift")
        elif 'Match using ORB' in option:
            self.match_features(descriptor="orb")
        elif 'Stitch Images' in option:
            self.stitch_images()

    def detect_features(self, method):
        """
        Applies feature detection algorithms based on the selected method.

        Parameters:
            method (str): The method of feature detection to apply ('Harris' or 'SIFT').
        """

        self.load_selected_pair()
        for processor in self.image_processors:
            if method == 'Harris':
                processor.apply_harris_corners()
            elif method == 'SIFT':
                keypoints, descriptors, time_taken = processor.apply_sift_features()
        self.display_images()

    def describe_features(self, method):
        """
        Describes features using the specified method.

        Parameters:
            method (str): The method of feature description to apply ('SIFT' or 'ORB').
        """

        self.load_selected_pair()
        for processor in self.image_processors:
            if method == 'SIFT':
                keypoints, descriptors, time_taken = processor.apply_sift_features()
            elif method == 'ORB':
                keypoints, descriptors, time_taken = processor.apply_orb_features()
        self.display_images()
        print(f"Generated descriptors shape: {descriptors.shape}")

    def match_features(self, descriptor):
        """
        Matches features between two images using the specified descriptor.

        Parameters:
            descriptor (str): The descriptor to use for feature matching ('sift' or 'orb').
        """

        self.load_selected_pair()

        if len(self.image_processors) != 2:
            print("A valid image pair must contain exactly two images.")
            return

        processor1, processor2 = self.image_processors

        if descriptor == "sift":
            if not processor1.keypoints or not processor1.descriptors:
                processor1.apply_sift_features(match=True)

            if not processor2.keypoints or not processor2.descriptors:
                processor2.apply_sift_features(match=True)

        else:  # ORB
            if not processor1.keypoints or not processor1.descriptors:
                processor1.apply_orb_features(match=True)

            if not processor2.keypoints or not processor2.descriptors:
                processor2.apply_orb_features(match=True)

        def resize_image(image, width):
            aspect_ratio = width / float(image.shape[1])
            dim = (width, int(image.shape[0] * aspect_ratio))
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        processor1.image = resize_image(processor1.image, 600)
        processor2.image = resize_image(processor2.image, 600)

        matches_img, matches, matching_accuracy, computational_time = processor1.match_features(
            processor2)

        # Display matches
        matches_img_ssd_pil = Image.fromarray(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        self.display_image(matches_img_ssd_pil)

        pair_name = self.selected_pair.get()

        # Save results
        matches_img_pil = Image.fromarray(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        matches_img_pil.save(f"./output/{pair_name}_feature_matching_{descriptor}.png")

        # Performance metrics
        print(
            f"{descriptor} Matching: Number of Matches = {len(matches)}, "
            f"Matching Accuracy = {matching_accuracy:.2f}, Computational Time = {computational_time:.2f} s")

    def stitch_images(self):
        """
        Stitches two images together using features matched between them.

        Uses SIFT features and RANSAC for homography estimation followed by image warping and stitching.
        """

        self.load_selected_pair()
        if len(self.image_processors) != 2:
            print("A valid image pair must contain exactly two images.")
            return

        processor1, processor2 = self.image_processors

        processor1.gray_image = cv2.cvtColor(processor1.image, cv2.COLOR_BGR2GRAY)
        processor2.gray_image = cv2.cvtColor(processor2.image, cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1, _ = processor1.apply_sift_features(match=True)
        keypoints2, descriptors2, _ = processor2.apply_sift_features(match=True)

        # Match features
        match_result = processor1.match_features(processor2)
        if not match_result:
            print("Feature matching did not return valid matches.")
            return

        _, matches, _, _ = match_result
        if not matches:
            print("No matches to process.")
            return

        stitched_image = stitch_images(processor1, processor2, matches)
        stitched_image_pil = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))

        pair_name = self.selected_pair.get()
        stitched_image_pil.save(f"./output/{pair_name}_stitched.png")

        self.display_image(stitched_image_pil)


