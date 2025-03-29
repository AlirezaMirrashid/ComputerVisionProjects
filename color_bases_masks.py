import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk

class RangeSlider:
    def __init__(self, parent, label, min_val, max_val, command):
        self.frame = tk.Frame(parent)
        self.frame.pack(pady=5)

        self.label = tk.Label(self.frame, text=label)
        self.label.grid(row=0, column=0, padx=5)

        self.min_val = tk.IntVar(value=min_val)
        self.max_val = tk.IntVar(value=max_val)

        self.min_slider = tk.Scale(self.frame, from_=min_val, to=max_val, orient="horizontal", variable=self.min_val, command=lambda x: command())
        self.min_slider.grid(row=0, column=1, padx=5)

        self.max_slider = tk.Scale(self.frame, from_=min_val, to=max_val, orient="horizontal", variable=self.max_val, command=lambda x: command())
        self.max_slider.grid(row=0, column=2, padx=5)

    def get(self):
        return self.min_val.get(), self.max_val.get()

def open_image():
    global img, photo, image_label
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img,(600,400))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        image_label.config(image=photo)
        image_label.image = photo

def update_mask():
    global img, mask_label
    if img is None:
        return

    # Get slider values
    h_min, h_max = hue_slider.get()
    s_min, s_max = saturation_slider.get()
    v_min, v_max = value_slider.get()
    r_min, r_max = red_slider.get()
    g_min, g_max = green_slider.get()
    b_min, b_max = blue_slider.get()
    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create HSV mask
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    hsv_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # Create RGB mask
    lower_rgb = np.array([b_min, g_min, r_min])
    upper_rgb = np.array([b_max, g_max, r_max])
    rgb_mask = cv2.inRange(img, lower_rgb, upper_rgb)

    # Combine masks
    combined_mask = cv2.bitwise_and(hsv_mask, rgb_mask)

    # Display mask
    mask_image = Image.fromarray(combined_mask)
    mask_image = mask_image.resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS)
    mask_photo = ImageTk.PhotoImage(mask_image)
    mask_label.config(image=mask_photo)
    mask_label.image = mask_photo

# Initialize Tkinter
root = tk.Tk()
root.title("Image Masking Tool")

# Variables
img = None

# Layout
main_frame = tk.Frame(root)
main_frame.pack(pady=10, padx=10)

# Left column for image and mask
image_frame = tk.Frame(main_frame)
image_frame.grid(row=0, column=0, padx=10)

# Image Display
image_label = tk.Label(image_frame)
image_label.pack(pady=5)

# Mask Display
mask_label = tk.Label(image_frame)
mask_label.pack(pady=5)

# Right column for sliders and button
controls_frame = tk.Frame(main_frame)
controls_frame.grid(row=0, column=1, padx=10)

# Open Image Button
open_button = tk.Button(controls_frame, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Sliders for HSV channels
slider_frame_hsv = tk.LabelFrame(controls_frame, text="Adjust HSV Channels")
slider_frame_hsv.pack(pady=10)

hue_slider = RangeSlider(slider_frame_hsv, "Hue", 0, 179, update_mask)
saturation_slider = RangeSlider(slider_frame_hsv, "Saturation", 0, 255, update_mask)
value_slider = RangeSlider(slider_frame_hsv, "Value", 0, 255, update_mask)

# Sliders for RGB channels
slider_frame_rgb = tk.LabelFrame(controls_frame, text="Adjust RGB Channels")
slider_frame_rgb.pack(pady=10)

red_slider = RangeSlider(slider_frame_rgb, "Red", 0, 255, update_mask)
green_slider = RangeSlider(slider_frame_rgb, "Green", 0, 255, update_mask)
blue_slider = RangeSlider(slider_frame_rgb, "Blue", 0, 255, update_mask)

# Run application
root.mainloop()
