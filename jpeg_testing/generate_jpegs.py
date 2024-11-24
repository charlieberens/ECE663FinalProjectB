from PIL import Image
import os

def compress_image(input_path, output_dir, quality_levels):
    """
    Compresses an image at multiple quality levels.
    
    Args:
        input_path (str): Path to the input image.
        output_dir (str): Directory to save compressed images.
        quality_levels (list of int): List of quality levels for compression.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the input image
    with Image.open(input_path) as img:
        # Convert to RGB if the image is not in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        for quality in quality_levels:
            # Define the output file name
            output_path = os.path.join(output_dir, f"compressed_quality_{quality}.jpg")
            # Save the image with the specified quality
            img.save(output_path, "JPEG", quality=quality)
            print(f"Saved: {output_path} (Quality: {quality})")

# Path to the input image
input_image_path = "../../data/midjourney/train/a/photo_1925@03-03-2023_06-35-24.jpg"  # Replace with your image path
# Directory to save compressed images
output_directory = "compressed_images"
# Quality levels to use for compression
quality_settings = range(10, 101, 10)

# Compress the image
compress_image(input_image_path, output_directory, quality_settings)
