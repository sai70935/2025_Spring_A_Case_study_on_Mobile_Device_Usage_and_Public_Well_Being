import argparse
import os
from PIL import Image
import sys

from facenet_pytorch import MTCNN
import numpy as np
from tqdm import tqdm

def crop(image_path, output_path):
    # Use MTCNN to detect the face
    img = Image.open(image_path)
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        # Get the first detected face
        x1, y1, x2, y2 = boxes[0]
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        
        # Crop the face from the image
        img_cropped = img.crop((x1, y1, x2, y2))
    else:
        img_cropped = None

    # Check if any faces were detected
    if img_cropped is None:
        raise Exception("No faces detected in the image")
    
    
    max_dim = 256
    # Resize the cropped image to the desired size
    scale = max_dim / max(img_cropped.size)
    new_size = (int(img_cropped.size[0] * scale), int(img_cropped.size[1] * scale))
    img_cropped = img_cropped.resize(new_size, Image.LANCZOS)

    final_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    final_img.paste(img_cropped, ((max_dim - img_cropped.size[0]) // 2, (max_dim - img_cropped.size[1]) // 2))

    # Save the cropped face image
    final_img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder for cropped images')
    args = parser.parse_args()
    folder = args.folder
    output = args.output

    mtcnn = MTCNN(image_size=256, device='cuda', post_process=False)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)
    # Iterate over all files in the folder
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, filename)
            output_path = os.path.join(output, filename)
            try:
                crop(image_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")