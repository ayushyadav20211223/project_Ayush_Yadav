import torch
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader
from config import *
import os

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

    print(f"✅ Found {len(classes)} classes, with 10 images each.")
    return all_images

def test_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Step 1: Validate data folder
    sample_image_paths = validate_data_folder()

    # Step 2: Load model
    model = TheModel().to(device)
    print("✅ Model initialized.")

    # Step 3: Load dataset and dataloader
    dataset = TheDataset()
    loader = the_dataloader(dataset=dataset, shuffle=True)
    print(f"✅ Dataset and Dataloader initialized. Dataset size: ")

    # Step 4: Train model
    print("📚 Starting training...")
    the_trainer(model, loader,epochs=epochs,optimizer='afas',loss_fn='adsf')
    print("✅ Training completed.")

    # Step 5: Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/final_weights_ignore.pth")
    print("✅ Model weights saved at checkpoints/final_weights_ignore.pth")

    # # Step 6: Load weights
    print("🛠️ Loading weights from checkpoint...")
    loaded_model = TheModel().to(device)
    loaded_model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
    loaded_model.eval()
    print("✅ Weights loaded into new model.")
    
    # Step 6: Inference check
    print("🔎 Running predictions on sample images...")
    predictions = the_predictor( sample_image_paths[:5])  # Predict on first 5 images
    print(f"Predictions on 5 images: {predictions}")

if __name__ == "__main__":
    test_all()
