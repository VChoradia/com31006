# utils.py
# Author: Vivek Choradia

from PIL import Image
import tkinter as tk

image_pairs = {
    'Choose a pair': (),
    'Pair 1': ('./images/left_1.png', './images/right_1.png'),
    'Pair 2': ('./images/left_2.jpg', './images/right_2.jpg'),
    'Pair 3': ('./images/left_3.jpg', './images/right_3.jpg'),
    'Pair 4': ('./images/left_4.jpg', './images/right_4.jpg'),
    'Pair 5': ('./images/left_5.jpg', './images/right_5.jpg'),
    'Pair 6': ('./images/left_6.jpg', './images/right_6.jpg'),
    'Pair 7': ('./images/left_7.jpg', './images/right_7.jpg'),
    'Pair 8': ('./images/left_8.jpg', './images/right_8.jpg'),
    'Pair 9': ('./images/left_9.jpg', './images/right_9.jpg'),
    'Pair 10': ('./images/left_10.jpg', './images/right_10.jpg'),
}

def combine_images(images, scale_factor=1.0):
    """Combine images side by side with an optional scale factor."""
    if scale_factor != 1.0:
        images = [i.resize((int(i.width * scale_factor), int(i.height * scale_factor)), Image.Resampling.LANCZOS)
                  for i in images]
    total_width = sum(i.width for i in images)
    max_height = max(i.height for i in images)
    combined_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        combined_image.paste(im, (x_offset, 0))
        x_offset += im.width
    return combined_image


def create_selection_popup(root, title, options, variable, callback):
    """Create a popup window for selecting options such as detection methods."""
    popup = tk.Toplevel(root)
    popup.title(title)

    for text, method in options.items():
        tk.Radiobutton(popup, text=text, variable=variable, value=method).pack(anchor=tk.W)

    tk.Button(popup, text='OK', command=callback).pack()
    return popup
