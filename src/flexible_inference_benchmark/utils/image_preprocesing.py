from PIL import Image, ImageEnhance
from io import BytesIO
import base64
import random
import numpy as np

def vary_image(img: Image.Image):
    # Slight brightness variation
    brightness_factor = 1 + random.uniform(-0.02, 0.02)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Add very light random pixel noise
    arr = np.array(img)
    noise = np.random.randint(-2, 3, arr.shape, dtype='int16')
    noisy_arr = np.clip(arr.astype('int16') + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_arr)

def change_image_pixels(image_bytes: bytes, iterations: int = 1) -> list[str]:
    """
    Change the pixels of an image represented as a base64 string or a file path.
    
    Args:
        image_str (str): Base64 encoded image.
        iterations (int): Number of times to change the pixels.
        
    Returns:
        list[str]: List of modified image strings in base64 format.
    """

    images = []
    
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))

    for _ in range(iterations):
        # Modify the image pixels
        modified_image = vary_image(image.copy()) 
        buffered = BytesIO()
        modified_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        images.append(f"data:image/jpeg;base64,{img_str}")
    
    return images