import torch
import torchvision.models as models
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def generate_dummy_model():
    print("Initializing ResNet-18 model with ImageNet weights...")
    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the final classification layer for binary output (1 logit for cancer probability)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)
    
    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Save the model state dictionary
    save_path = os.path.join('model', 'lung_cancer_model.pth')
    print(f"Saving simulated trained weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    
    print("Done! You can now load this file in app.py")
    print("IMPORTANT: This model is untrained for lung cancer. You must train it on a real dataset!")

if __name__ == "__main__":
    generate_dummy_model()
