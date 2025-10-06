"""
Shared image processing utilities for hero image labeling.

Provides consistent image resizing and encoding across all labeling methods.
"""

import base64
import io
from PIL import Image


class ImageProcessor:
    """Handles image loading, resizing, and encoding for VLM processing."""

    @staticmethod
    def resize_image_to_bytes(image_path: str, max_size: int = 768) -> bytes:
        """
        Resize image to specified max dimension on long side and return as JPEG bytes.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension for the long side (default: 768)

        Returns:
            JPEG image as bytes

        Raises:
            Exception: If image processing fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size

                # Resize to max_size on long side while preserving aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)

                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                img_byte_arr = io.BytesIO()
                resized_img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                return img_byte_arr.getvalue()

        except Exception as e:
            raise Exception(f"Failed to process image {image_path}: {str(e)}")

    @staticmethod
    def resize_image_to_base64(image_path: str, max_size: int = 768) -> str:
        """
        Resize image to specified max dimension on long side and return as base64 string.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension for the long side (default: 768)

        Returns:
            Base64 encoded JPEG image string

        Raises:
            Exception: If image processing fails
        """
        image_bytes = ImageProcessor.resize_image_to_bytes(image_path, max_size)
        return base64.b64encode(image_bytes).decode('utf-8')
