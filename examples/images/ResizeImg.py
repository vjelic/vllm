from PIL import Image

def resize_image(image_path, output_size, output_path):
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Resize the image
        img_resized = img.resize(output_size, Image.LANCZOS)
        
        # Save the resized image
        img_resized.save(output_path)
        
        return img_resized
        
    except IOError:
        print(f"Unable to open image file: {image_path}")

# Example usage:
image_path = "cherry_blossom.jpg"  # Replace with your image path

# Resize to 1024x768 and save
output_size_1024x768 = (1024, 768)
output_path_1024x768 = "cherry_blossom_1024x768.jpg"
resize_image(image_path, output_size_1024x768, output_path_1024x768)
print(f"Image resized to {output_size_1024x768} and saved as {output_path_1024x768}")

# Resize to 1920x1080 and save
output_size_1920x1080 = (1920, 1080)
output_path_1920x1080 = "cherry_blossom_1920x1080.jpg"
resize_image(image_path, output_size_1920x1080, output_path_1920x1080)
print(f"Image resized to {output_size_1920x1080} and saved as {output_path_1920x1080}")

# Resize to 3840x2160 and save
output_size_3840x2160 = (3840, 2160)
output_path_3840x2160 = "cherry_blossom_3840x2160.jpg"
resize_image(image_path, output_size_3840x2160, output_path_3840x2160)
print(f"Image resized to {output_size_3840x2160} and saved as {output_path_3840x2160}")
