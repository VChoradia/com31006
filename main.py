# main.py
# Author: Vivek Choradia

import tkinter as tk
from FeatureDetectionUI import FeatureDetectionUI

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureDetectionUI(root)
    root.mainloop()
