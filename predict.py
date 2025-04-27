import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y
import os
from interface import TheModel

inference_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def validate_data_folder(data_path="data/"):
    all_images = []

    # Go through each subfolder (each class)
    classes = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    if len(classes) == 0:
        raise ValueError(f"No class folders found inside `{data_path}/`.")

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < 10:
            raise ValueError(f"Class `{cls}` has only {len(images)} images. Need at least 10.")
        
        all_images.extend(images[:10])  # Take only first 10 images from each class

    # print(f"✅ Found {len(classes)} classes, with 10 images each.")
    return all_images



def cryptic_inf_f( image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TheModel().to(device)
    model.eval()

    loaded_model = TheModel().to(device)
    loaded_model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
    loaded_model.eval()
    
    images = []
    
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = inference_transform(image).unsqueeze(0)
        images.append(image)

    images = torch.cat(images).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(1).cpu().tolist()

    return predictions
