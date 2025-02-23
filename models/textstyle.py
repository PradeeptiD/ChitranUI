# !pip install colorthief
# pip install --upgrade pillow
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
import random
from colorthief import ColorThief
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import io

model = models.densenet201(pretrained=True)
model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


def preprocess_image(image):
    original_img = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(original_img).unsqueeze(0).to("cpu")
    return original_img, img_tensor


#Generating saliency map
def generate_saliency_map(model, img_tensor, target_class=None):
    img_tensor.requires_grad_()

    output = model(img_tensor)  # Forward pass
    probabilities = F.softmax(output, dim=1)
    top5_probs, top5_classes = torch.topk(probabilities, 5)
    print("Top 5 predicted classes:")
    for i in range(5):
        print(f"Class {top5_classes[0, i].item()}: Probability {top5_probs[0, i].item():.4f}")

    if target_class is None:
        target_class = output.argmax().item()  # Get the predicted class if none specified

    #Backward pass to calculate gradients
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    #Get the saliency map from gradients
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()  # Convert to NumPy array

    #Normalize the saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency


#Visualizing with rectangle covering the densest salient area
def find_coordinates(original_img, saliency_map, threshold=0.08):
    saliency_resized = cv2.resize((saliency_map * 255).astype(np.uint8), original_img.size, interpolation=cv2.INTER_LINEAR)

    # Apply threshold to get high-saliency areas
    _, binary_map = cv2.threshold(saliency_resized, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Find contours of high-saliency regions
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No high-saliency regions detected.")
        return np.array(original_img)

    # Find the contour with the largest number of bright pixels
    best_contour = None
    max_bright_pixels = 0

    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(binary_map)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Count the number of bright pixels within the contour
        bright_pixels = cv2.countNonZero(cv2.bitwise_and(binary_map, binary_map, mask=mask))

        # Update if this contour has more bright pixels
        if bright_pixels > max_bright_pixels:
            max_bright_pixels = bright_pixels
            best_contour = contour

    if best_contour is None:
        print("No valid salient region found.")
        return np.array(original_img)

    # Get the bounding rectangle for the densest salient region
    x, y, w, h = cv2.boundingRect(best_contour)

    return (x,y,w,h)


def generate_text_colour(image):
    # temp_path = "/tmp/temp_image.jpg"
    # image.save(temp_path)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Save as PNG or JPEG
    img_byte_arr.seek(0)  # Reset file pointer to the start

    
    color_thief = ColorThief(img_byte_arr)

    dominant_color = color_thief.get_color(quality=1)
    dominant_color

    palette = color_thief.get_palette(color_count=6)
    palette

    def rgb_distance(color1, color2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

    # Largest Euclidean distance
    max_distance = 0
    best_contrast_color = None
    black = (0,0,0)
    white = (255, 255, 255)

    def luminance(color):
        """Calculate the relative luminance of an RGB color."""
        r, g, b = [c / 255.0 for c in color]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for color in palette:
        dist = rgb_distance(dominant_color, color)
        if dist > max_distance:
            max_distance = dist
            text_color = color
        if max_distance<100:
            text_color = white if luminance(dominant_color)<0.5 else black
    
    return text_color


def generate_font_style():
    fonts_path = "models/fonts"
    font_files = [os.path.join(fonts_path, font) for font in os.listdir(fonts_path) if font.endswith(('.ttf', '.otf', '.TTF'))]
    selected_font = random.choice(font_files)
    return selected_font


def calculate_max_font_size(image, text, coordinates, font_path):
    x,y,w,h = coordinates

    max_font_size = 50 # Start with a large font size
    min_font_size = 10   # Define a minimum font size to avoid infinite loops
    best_font_size = min_font_size
    best_wrapped_lines = []

    while max_font_size >= min_font_size:
        # Try the current font size
        current_font_size = (max_font_size + min_font_size) // 2
        font = ImageFont.truetype(font_path, current_font_size)

        # Wrap text based on the width of the bounding box
        draw = ImageDraw.Draw(image)

        parts = text.split("!", maxsplit=1)
        first_part = parts[0] + "!"
        second_part = parts[1].strip() if len(parts)>1 else ""

        wrapped_lines = []
        for part in [first_part, second_part]:
            if part:  # Process each part if it exists
                words = part.split(" ")
                line = ""
                for word in words:
                    test_line = line + word + " "
                    line_width = draw.textlength(test_line, font=font)  # Use the font object here
                    if line_width <= w:
                        line = test_line
                    else:
                        wrapped_lines.append(line.strip())
                        line = word + " "
                if line:
                    wrapped_lines.append(line.strip())

        # Calculate total height of the text with the current font size
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        total_text_height = len(wrapped_lines) * line_height

        # Check if the total text height fits within the height of the bounding box
        if total_text_height <= h:
            best_font_size = current_font_size  # Update best font size
            best_wrapped_lines = wrapped_lines[:]
            min_font_size = current_font_size + 1  # Try larger sizes
        else:
            max_font_size = current_font_size - 1  # Try smaller sizes

    return best_font_size, wrapped_lines


def generate_styled_image(text, image):
    original_img, img_tensor = preprocess_image(image)
    saliency = generate_saliency_map(model, img_tensor)
    coordinates = find_coordinates(original_img, saliency, threshold=0.08)
    text_color = generate_text_colour(image)
    font_style = generate_font_style()
    font_size, wrapped_lines = calculate_max_font_size(image, text, coordinates, font_style)

    font = ImageFont.truetype(font_style, font_size)

    # text_image = Image.new("RGB", (frame_width, frame_height), dominant_color)
    text_image = image
    draw = ImageDraw.Draw(text_image)

    #saliency map
    x,y,w,h = coordinates

    # Calculate the total text height for vertical centering
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    line_spacing = int(line_height * 0.3)
    # line_height_2 = font_size.getbbox("A")[3] - font_size.getbbox("A")[1]
    total_text_height = len(wrapped_lines) * (line_height + line_spacing) - line_spacing
    # y_offset = (image.height - total_text_height) // 2

    # Draw the first part of the text
    y_offset = y + (h - total_text_height) // 2  # starting y position, adjust as needed

    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)  # Get bounding box for text
        text_width = bbox[2] - bbox[0]  # width
        # text_height = bbox[3] - bbox[1]  # height

        x_offset = x + (w - text_width) // 2  # Center the text horizontally
        draw.text((x_offset, y_offset), line, font=font, fill=text_color)
        y_offset += line_height  # Move the y_offset down by the height of the text

    return text_image