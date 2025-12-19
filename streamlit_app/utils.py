"""
utils.py - Helper functions for face verification
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import spatial
import onnx
from onnx2pytorch import ConvertModel


def load_onnx_model(model_path, device='cuda'):
    """
    Load ONNX model and convert to PyTorch
    
    Args:
        model_path: Path to ONNX model file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded PyTorch model
    """
    print(f"Loading ONNX model from: {model_path}")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    
    # Convert to PyTorch
    model = ConvertModel(onnx_model)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    return model, device


def preprocess_image(image_path, image_size=(112, 112)):
    """
    Load and preprocess image for face recognition
    
    Args:
        image_path: Path to image file
        image_size: Target size (height, width)
    
    Returns:
        tensor: Preprocessed image tensor
        original_image: Original PIL Image for display
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    # Define transforms (matching your test script)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    image_tensor = transform(original_image)
    
    return image_tensor, original_image


def extract_features(model, image_tensor, device):
    """
    Extract feature embeddings from image
    
    Args:
        model: Loaded model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        features: Feature embeddings (numpy array)
    """
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
    
    # Convert to numpy
    features = features.cpu().numpy()
    
    return features


def compute_cosine_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors
    
    Args:
        features1: Feature vector 1 (numpy array)
        features2: Feature vector 2 (numpy array)
    
    Returns:
        similarity: Cosine similarity score (0 to 1)
    """
    # Flatten features
    feat1 = features1.flatten()
    feat2 = features2.flatten()
    
    # Compute cosine similarity
    similarity = 1 - spatial.distance.cosine(feat1, feat2)
    
    # Clamp to [0, 1]
    if similarity < 0:
        similarity = 0.0
    
    return float(similarity)


def predict_same_person(similarity, threshold=0.1998):
    """
    Predict if two images are of the same person
    
    Args:
        similarity: Cosine similarity score
        threshold: Decision threshold
    
    Returns:
        is_same: Boolean indicating if same person
        confidence: Confidence level string
    """
    is_same = similarity >= threshold
    
    # Determine confidence level
    distance_from_threshold = abs(similarity - threshold)
    
    if distance_from_threshold >= 0.2:
        confidence = "High"
    elif distance_from_threshold >= 0.1:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return is_same, confidence


def format_result(similarity, threshold, is_same, confidence):
    """
    Format the prediction result for display
    
    Args:
        similarity: Cosine similarity score
        threshold: Decision threshold
        is_same: Boolean indicating if same person
        confidence: Confidence level string
    
    Returns:
        result_dict: Dictionary with formatted results
    """
    result_dict = {
        'similarity': similarity,
        'threshold': threshold,
        'prediction': "✅ Same Person" if is_same else "❌ Different Person",
        'confidence': confidence,
        'raw_decision': is_same
    }
    
    return result_dict

