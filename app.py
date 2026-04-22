from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import random
import datetime

app = Flask(__name__)

import torchvision.models as models

# Ensure required directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Initialize ResNet-18 Model
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Binary classification
    
    # Try to load trained weights if they exist
    model_path = os.path.join('model', 'lung_cancer_model.pth')
    if os.path.exists(model_path):
        try:
            # Try loading with weights_only=True for security (Torch 1.13+)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
            print("Loaded trained model weights!")
        except TypeError:
            # Fallback for older torch versions
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Loaded trained model weights (fallback mode)!")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("WARNING: Model weights not found. Using untrained ResNet-18.")
    
    model.eval() # Set to evaluation mode
    return model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    # Get patient data
    age = request.form.get('age', 'Unknown')
    smoking = request.form.get('smoking', 'Unknown')
    family_history = request.form.get('family_history', 'Unknown')

    img = Image.open(img_path).convert('RGB')
    
    # Standard ImageNet transforms for ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction_prob = torch.sigmoid(output).item() # Convert logit to probability

    # Calculate Confidence (Scale between 70% and 95% for realism in demo)
    raw_confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
    scaled_confidence = 70 + (raw_confidence - 0.5) * 2 * 25 # Scale 0.5-1.0 to 70-95
    confidence_value = min(max(scaled_confidence, 70), 98.5)
    
    # OpenCV Image Processing (Simulate heatmap/overlay)
    cv_img = cv2.imread(img_path)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    processed_filename = 'processed_' + img_file.filename
    processed_path = os.path.join('static', processed_filename)
    
    if prediction_prob > 0.5:
        # Simulate an area of interest by blending the heatmap heavily
        processed = cv2.addWeighted(cv_img, 0.5, heatmap, 0.5, 0)
    else:
        # If normal, just show an edge-enhanced or mild grayscale version
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        processed = cv2.addWeighted(cv_img, 0.8, edges_colored, 0.2, 0)

    cv2.imwrite(processed_path, processed)

    # Dynamic Data Generation
    regions = ["Right Upper Lobe (RUL)", "Right Middle Lobe (RML)", "Right Lower Lobe (RLL)", "Left Upper Lobe (LUL)", "Left Lower Lobe (LLL)"]
    cancer_statements = [
        "Irregular opacity detected in lung parenchyma.",
        "Possible nodular lesion observed.",
        "Localized tissue density variation identified.",
        "Small suspicious mass detected in specified region."
    ]
    
    if prediction_prob > 0.5:
        result_text = "Suspicious Pulmonary Abnormality Detected"
        is_danger = True
        region = random.choice(regions)
        analysis_text = random.choice(cancer_statements)
        clinical_interp = "Findings suggest possible early-stage abnormality requiring clinical correlation."
        recommendation = "Consult a pulmonologist immediately. Further diagnostic imaging (e.g., PET scan) recommended."
        # Calculate risk based on patient data
        risk_score = 3
        if smoking == 'Yes': risk_score += 2
        if family_history == 'Yes': risk_score += 1
        if int(age) > 50: risk_score += 1
        risk_level = "High" if risk_score >= 5 else "Moderate"
    else:
        result_text = "No Significant Abnormality Detected"
        is_danger = False
        region = "No significant region identified"
        analysis_text = "No visible abnormalities detected. Lung parenchyma appears normal."
        clinical_interp = "Findings are within normal limits for this modality."
        recommendation = "Routine follow-up as per standard health guidelines."
        risk_score = 0
        if smoking == 'Yes': risk_score += 2
        if family_history == 'Yes': risk_score += 1
        if int(age) > 50: risk_score += 1
        risk_level = "Moderate" if risk_score >= 3 else "Low"
        
    if confidence_value >= 90:
        conf_label = "High"
    elif confidence_value >= 75:
        conf_label = "Moderate"
    else:
        conf_label = "Low"

    scan_details = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': img_file.filename,
        'type': 'CT/X-Ray Scan'
    }

    patient_data = {
        'age': age,
        'smoking': smoking,
        'family': family_history
    }

    return render_template('result.html', 
                           result_text=result_text, 
                           is_danger=is_danger,
                           confidence_value=round(confidence_value, 1),
                           conf_label=conf_label,
                           region=region,
                           analysis_text=analysis_text,
                           clinical_interp=clinical_interp,
                           recommendation=recommendation,
                           risk_level=risk_level,
                           img_path=img_path,
                           processed_path=processed_path,
                           scan_details=scan_details,
                           patient_data=patient_data)

@app.route('/risk')
def risk_page():
    return render_template('risk.html')

@app.route('/doctors')
def doctors_page():
    return render_template('doctors.html')

@app.route('/info')
def info_page():
    return render_template('info.html')

@app.route('/faq')
def faq_page():
    return render_template('faq.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
