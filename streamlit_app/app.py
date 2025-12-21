"""
app.py - Streamlit Face Verification Application with Face Detection and Age Prediction

Usage:
    streamlit run app.py
"""

import streamlit as st
import os
from pathlib import Path
import torch
from utils import (
    load_onnx_model,
    preprocess_image,
    extract_features,
    compute_cosine_similarity,
    predict_same_person,
    format_result
)
import tempfile
import gdown
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import MTCNN for face detection
try:
    from mtcnn import MTCNN
    face_detector = MTCNN()
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'mtcnn'])
    from mtcnn import MTCNN
    face_detector = MTCNN()

# Import age prediction model
try:
    from model import predict_age_gender
except ImportError:
    st.error("⚠️ model.py not found. Please ensure the age prediction model is in the same directory.")
    predict_age_gender = None


# Configuration (hardcoded instead of YAML)
CONFIG = {
    'model': {
        'path': '../pretrained_models/R100_Glint360K.onnx',
        'type': 'onnx'
    },
    'inference': {
        'threshold': 0.1998,
        'image_size': [112, 112],
        'normalization': {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }
    },
    'ui': {
        'title': 'Age-Invariant Face Recognition',
        'description': 'Compare faces across ages to verify identity',
        'example_image1': '../val_data/3_MariaCallas_35_f.jpg',
        'example_image2': '../val_data/4_MariaCallas_41_f.jpg'
    }
}


MODEL_PATH = Path(__file__).parent / "../pretrained_models/R100_Glint360K.onnx"
MODEL_PATH = MODEL_PATH.resolve()  # Converts to absolute path

# Google Drive direct download URL
GDRIVE_FILE_ID = "1XboNCeVBU42B-rz3-bqiY7W6jgsfOTzZ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Download model if it doesn't exist
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
    with st.spinner("Downloading ONNX model (~260 MB)..."):
        gdown.download(GDRIVE_URL, str(MODEL_PATH), quiet=False)


# Page configuration
st.set_page_config(
    page_title="Face Verification",
    page_icon="👤",
    layout="wide"
)


@st.cache_resource
def load_model(model_path):
    """Load model (cached to avoid reloading)"""
    # Convert relative path to absolute
    abs_model_path = Path(__file__).parent / model_path
    device = 'cpu'
    model, device = load_onnx_model(str(abs_model_path), device)
    return model, device


def detect_single_face(image_path):
    """
    Detect exactly one face in the image using MTCNN
    
    Args:
        image_path: Path to image file
    
    Returns:
        tuple: (success, face_crop, bbox, error_message, original_image)
            - success: Boolean indicating if exactly one face was found
            - face_crop: PIL Image of cropped face (or None)
            - bbox: [x1, y1, x2, y2] coordinates (or None)
            - error_message: Error description (or None)
            - original_image: Original PIL Image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # Detect faces
    detections = face_detector.detect_faces(img_array)
    
    # Check number of faces
    if len(detections) == 0:
        return False, None, None, "❌ No face detected in the image. Please upload an image with a visible face.", image
    
    if len(detections) > 1:
        return False, None, None, f"❌ Multiple faces detected ({len(detections)} faces). Please upload an image with only one face.", image
    
    # Exactly one face detected
    detection = detections[0]
    box = detection['box']
    
    # Extract face region with padding
    x, y, w, h = box
    padding = int(0.2 * max(w, h))  # 20% padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_array.shape[1], x + w + padding)
    y2 = min(img_array.shape[0], y + h + padding)
    
    # Crop face
    face_crop = image.crop((x1, y1, x2, y2))
    bbox = [x1, y1, x2, y2]
    
    return True, face_crop, bbox, None, image


def draw_bbox_with_age(image, bbox, age):
    """
    Draw bounding box and age info on image
    
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2]
        age: Predicted age
    
    Returns:
        PIL Image with annotations
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
    
    # Try to load font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
    
    # Prepare text
    age_text = f"Age: {age} years"
    
    # Calculate text position (above bbox if possible, below if not enough space)
    text_y = y1 - 40
    if text_y < 0:
        text_y = y2 + 10
    
    # Draw text background for better readability
    text_bbox = draw.textbbox((x1, text_y), age_text, font=font_large)
    draw.rectangle([(text_bbox[0]-5, text_bbox[1]-5), (text_bbox[2]+5, text_bbox[3]+5)], fill="lime")
    
    # Draw text
    draw.text((x1, text_y), age_text, fill="black", font=font_large)
    
    return img_copy


def verify_faces(face_crop1, face_crop2, model, device, threshold):
    """
    Main verification function using cropped face images
    
    Args:
        face_crop1: PIL Image of first face
        face_crop2: PIL Image of second face
        model: Loaded verification model
        device: Device for inference
        threshold: Decision threshold
    
    Returns:
        result: Dictionary with verification results
    """
    # Save face crops temporarily for preprocessing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
        face_crop1.save(tmp1.name)
        tmp_path1 = tmp1.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
        face_crop2.save(tmp2.name)
        tmp_path2 = tmp2.name
    
    try:
        # Preprocess images
        tensor1, _ = preprocess_image(tmp_path1)
        tensor2, _ = preprocess_image(tmp_path2)
        
        # Extract features
        features1 = extract_features(model, tensor1, device)
        features2 = extract_features(model, tensor2, device)
        
        # Compute similarity
        similarity = compute_cosine_similarity(features1, features2)
        
        # Predict
        is_same, confidence = predict_same_person(similarity, threshold)
        
        # Format result
        result = format_result(similarity, threshold, is_same, confidence)
        
        return result
    
    finally:
        # Clean up temp files
        os.unlink(tmp_path1)
        os.unlink(tmp_path2)


def main():
    # Custom CSS to remove top padding and compress layout
    st.markdown(
        """
        <style>
        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Reduce header spacing */
        h1 {
            margin-top: 0;
            padding-top: 0;
        }
        
        /* Compact file uploader */
        .stFileUploader {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header (compact version)
    st.markdown(
        f"""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="font-size: 36px; margin: 10px 0; padding: 0; color: #ffffff;">{CONFIG['ui']['title']}</h1>
            <p style="margin: 5px 0; font-size: 16px; color: #aaaaaa;">{CONFIG['ui']['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path (editable)
        model_path = st.text_input(
            "Model Path",
            value=CONFIG['model']['path'],
            help="Path to ONNX model file (relative to app.py)"
        )
        
        # Use fixed threshold from config
        threshold = CONFIG['inference']['threshold']
        
        st.markdown("---")
        st.info("📌 **Face Detection Active**\n\nImages must contain exactly one face.")
    
    # Main content - Image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Image 1")
        uploaded_file1 = st.file_uploader(
            "Upload first image",
            type=['jpg', 'jpeg', 'png'],
            key="img1",
            help="Upload the first face image"
        )
    
    with col2:
        st.subheader("📷 Image 2")
        uploaded_file2 = st.file_uploader(
            "Upload second image",
            type=['jpg', 'jpeg', 'png'],
            key="img2",
            help="Upload the second face image"
        )
    
    st.markdown("---")
    
    # Verify button
    if st.button("🔍 Verify Faces", type="primary", use_container_width=True):
        # Validate inputs
        if uploaded_file1 is None or uploaded_file2 is None:
            st.error("⚠️ Please upload both images")
            return
        
        # Check if age model is available
        if predict_age_gender is None:
            st.error("⚠️ Age prediction model not found. Please ensure model.py is in the streamlit_app directory.")
            return
        
        # Save uploaded files to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save image 1
            temp_path1 = os.path.join(temp_dir, uploaded_file1.name)
            with open(temp_path1, 'wb') as f:
                f.write(uploaded_file1.getbuffer())
            
            # Save image 2
            temp_path2 = os.path.join(temp_dir, uploaded_file2.name)
            with open(temp_path2, 'wb') as f:
                f.write(uploaded_file2.getbuffer())
            
            # Detect faces in both images
            with st.spinner("Detecting faces..."):
                success1, face_crop1, bbox1, error1, original_img1 = detect_single_face(temp_path1)
                success2, face_crop2, bbox2, error2, original_img2 = detect_single_face(temp_path2)
            
            # Check if face detection was successful for both images
            if not success1:
                st.error(f"**Image 1:** {error1}")
                return
            
            if not success2:
                st.error(f"**Image 2:** {error2}")
                return
            
            # Both faces detected successfully
            st.success("✅ Faces detected successfully in both images!")
            
            # Predict age for both faces
            with st.spinner("Predicting age..."):
                try:
                    age_result1 = predict_age_gender(face_crop1)
                    age_result2 = predict_age_gender(face_crop2)
                except Exception as e:
                    st.error(f"❌ Error during age prediction: {str(e)}")
                    return
            
            # Draw bounding boxes with age info (without gender)
            annotated_img1 = draw_bbox_with_age(
                original_img1, 
                bbox1, 
                age_result1['age']
            )
            
            annotated_img2 = draw_bbox_with_age(
                original_img2, 
                bbox2, 
                age_result2['age']
            )
            
            # Load verification model
            with st.spinner("Loading verification model..."):
                try:
                    model, device = load_model(model_path)
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    return
            
            # Perform face verification using cropped faces
            with st.spinner("Verifying identity..."):
                try:
                    result = verify_faces(
                        face_crop1,
                        face_crop2,
                        model,
                        device,
                        threshold
                    )
                except Exception as e:
                    st.error(f"❌ Error during verification: {str(e)}")
                    return
        
        # Display results
        st.success("✅ Verification Complete!")
        st.markdown("---")
        
        # Display images side by side with bounding boxes
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.image(annotated_img1, caption="Image 1", width="stretch")
            st.markdown(f"<h3 style='text-align: center;'>Age: {age_result1['age']} years</h3>", unsafe_allow_html=True)

        with img_col2:
            st.image(annotated_img2, caption="Image 2", width="stretch")
            st.markdown(f"<h3 style='text-align: center;'>Age: {age_result2['age']} years</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display verification results
        st.subheader("📊 Verification Results")
        
        # Result card
        if result['raw_decision']:
            st.success(f"### {result['prediction']}")
        else:
            st.error(f"### {result['prediction']}")
        
        # Metrics in columns
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Cosine Similarity",
                f"{result['similarity']:.4f}",
                help="Similarity score between 0 and 1"
            )
        
        with metric_col2:
            st.metric(
                "Threshold",
                f"{result['threshold']:.4f}",
                help="Decision boundary for same/different person"
            )
        
        with metric_col3:
            st.metric(
                "Confidence",
                result['confidence'],
                help="Confidence level based on distance from threshold"
            )
        
        # Additional info
        with st.expander("ℹ️ Interpretation Guide"):
            st.markdown("""
            **Cosine Similarity Score:**
            - **>= Threshold**: Predicted as **Same Person**
            - **< Threshold**: Predicted as **Different Person**
            
            **Confidence Levels:**
            - **High**: Score is >0.2 away from threshold
            - **Medium**: Score is 0.1-0.2 away from threshold
            - **Low**: Score is <0.1 away from threshold
            
            **Age Prediction:**
            - Age is predicted for each detected face independently
            - The age-invariant face verification model compares facial features regardless of age differences
            
            **Note:** The threshold (0.1998) was determined from base model evaluation 
            to achieve optimal accuracy on the FG-NET dataset.
            """)


if __name__ == "__main__":
    main()