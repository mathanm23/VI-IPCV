import cv2
import numpy as np

class EyeExtractor:
    def __init__(self, padding=10):
        """
        Initialize the Eye Extractor.
        
        Args:
            padding: Padding in pixels to add around the eye landmarks before cropping.
        """
        self.padding = padding

    def _get_eye_bounding_box(self, image_shape, eye_points):
        """
        Compute a square bounding box for a given set of eye points.
        """
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y
        
        # Find the center
        center_x = min_x + width // 2
        center_y = min_y + height // 2
        
        # Determine the size of the square (use the larger dimension + padding)
        side = max(width, height) + (self.padding * 2)
        
        # Calculate new coordinates
        start_x = max(0, center_x - side // 2)
        start_y = max(0, center_y - side // 2)
        end_x = min(image_shape[1], start_x + side)
        end_y = min(image_shape[0], start_y + side)
        
        # Final adjustment to ensure it stays within bounds if we shifted it
        if end_x == image_shape[1]: start_x = max(0, end_x - side)
        if end_y == image_shape[0]: start_y = max(0, end_y - side)
        
        return int(start_x), int(start_y), int(end_x), int(end_y)

    def extract_eyes(self, image, left_eye_points, right_eye_points):
        """
        Crop the left and right eye regions from the image based on landmarks.
        
        Args:
            image: Original BGR image
            left_eye_points: list of (x, y) coordinates
            right_eye_points: list of (x, y) coordinates
            
        Returns:
            left_eye_img, right_eye_img (or None, None if extraction fails)
        """
        if not left_eye_points or not right_eye_points:
            return None, None
            
        # Get bounding boxes
        lx_min, ly_min, lx_max, ly_max = self._get_eye_bounding_box(image.shape, left_eye_points)
        rx_min, ry_min, rx_max, ry_max = self._get_eye_bounding_box(image.shape, right_eye_points)
        
        # Crop images
        left_eye_img = image[ly_min:ly_max, lx_min:lx_max]
        right_eye_img = image[ry_min:ry_max, rx_min:rx_max]
        
        # Ensure crops are valid
        if left_eye_img.size == 0 or right_eye_img.size == 0:
            return None, None
            
        return left_eye_img, right_eye_img
