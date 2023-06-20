import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
import shutil

def save_model(model, model_name):
    # Create the "models" folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save the model
    torch.save(model.state_dict(), f"models/{model_name}.pt")


# Load Images
def load_images(folder_path):
    target_size = (512, 512)
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(target_size)  # Resize image to target size
            image = transforms.ToTensor()(image)
            image_list.append(image)
    images = torch.stack(image_list)
    return images

# Reshape images to tensor
def reshape_images_to_tensor(images):
    num_images = images.size(0)
    image_channels = images.size(1)
    image_height = images.size(2)
    image_width = images.size(3)

    # Reshape images tensor to fit the semantic encoder input shape
    images = images.view(num_images, image_channels, image_height, image_width)
    
    
def empty_directory(path):
    # Define the directory path
    directory = path

    # Empty the directory if it exists
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # Create the directory
    os.makedirs(directory)
