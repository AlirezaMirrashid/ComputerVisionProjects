
import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an image by a given angle.
    """
    # Get the center of the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, matrix, (w, h))
    
    return rotated, matrix  # Return the rotated image and rotation matrix

def detect_card_using_template(image, template):
    """
    Detect the card in the image using template matching with rotation.
    """
    best_match = None
    best_angle = 0
    max_val = -1
    best_rotated_template = None
    best_matrix = None
    best_rect = None
    
    # Try different rotation angles (from 0 to 360 degrees)
    for angle in range(0, 360, 5):  # You can adjust the step for finer rotations
        rotated_image, matrix = rotate_image(image, angle)
        
        # Perform template matching
        result = cv2.matchTemplate(rotated_image, template, cv2.TM_CCOEFF_NORMED)
        
        # Get the maximum match value and location
        min_val, max_val_temp, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # If this is the best match, update the best match and angle
        if max_val_temp > max_val:
            max_val = max_val_temp
            best_match = max_loc
            best_angle = angle
            best_rotated_image = rotated_image
            best_matrix = matrix
            
            # Calculate the rotated bounding box using the match location and size of the rotated template
            rect = cv2.minAreaRect(np.array([[[max_loc[0], max_loc[1]]], [[max_loc[0] + template.shape[1], max_loc[1]]], [[max_loc[0], max_loc[1] + template.shape[0]]]]))
            best_rect = rect

    # Return the best match location, rotation angle, and bounding box
    return best_match, best_angle, best_rotated_image, best_matrix, best_rect

def draw_rotated_bounding_box(image, best_match, best_rotated_image, best_angle, best_rect):
    """
    Draws the rotated bounding box around the detected card on the rotated image.
    """
    # Get the center, size, and angle of the rotated rectangle
    center, (w, h), angle = best_rect
    
    # Get the coordinates of the rotated rectangle
    box = cv2.boxPoints(best_rect)  # Get the four corners of the rotated rectangle
    box = np.int0(box)  # Convert to integer values
    
    # Draw the rotated bounding box on the rotated template image
    rotated_image_with_box = best_rotated_image.copy()  # Copy the rotated template to draw on it
    cv2.polylines(rotated_image_with_box, [box], True, (0, 255, 0), 2)
    
    # Now rotate the image back to the original image's orientation
    original_image_with_box, _ = rotate_image(rotated_image_with_box, -best_angle)

    # # Draw the corrected bounding box on the original image
    # original_image_with_box = image.copy()
    # cv2.polylines(original_image_with_box, [np.int0(box_rotated_back)], True, (0, 255, 0), 2)
    
    return original_image_with_box, rotated_image_with_box, (w, h)


# Function to calculate pixel size scale based on real dimensions (in cm)
def calculate_pixel_size(real_width_cm, real_height_cm, pixel_width, pixel_height):
    """
    Given the real width and height of the card in cm, and the dimensions in pixels,
    calculate the scale factor (cm per pixel).
    """
    scale_width = real_width_cm / pixel_width
    scale_height = real_height_cm / pixel_height
    
    return scale_width, scale_height

# Example usage:
image = cv2.imread('samples/samp2.jpg')  # Input image with a bank card
template = cv2.imread('samples/bank_card.jpg')  # Template image of the card




# Step 1: Detect the card's position and orientation
best_match, best_angle, best_rotated_template, best_matrix, best_rect = detect_card_using_template(image, template)

# Step 2: Draw the rotated bounding box around the detected card
if best_match and best_rect:
    original_image_with_box, rotated_image_with_box, (pixel_width, pixel_height) = draw_rotated_bounding_box(image.copy(), best_match, best_rotated_template, best_angle, best_rect)
    
    # Step 3: Show the image with the detected card and rotated bounding box
    cv2.imshow("Image with Corrected Bounding Box", original_image_with_box)
    # cv2.imshow("Rotated Image with Bounding Box", rotated_image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Known dimensions of the card (real width and height in cm)
    real_card_width = 8.6  # in cm
    real_card_height = 5.4  # in cm
    # Step 4: Calculate the scale factor (cm per pixel)
    scale_width, scale_height = calculate_pixel_size(real_card_width, real_card_height, pixel_width, pixel_height)
    print(f"Scale factor: {scale_width:.2f} cm per pixel (Width), {scale_height:.2f} cm per pixel (Height)")

print(f"Detected card with best angle: {best_angle} degrees")
