"""
app.py - Streamlit Face Verification Application with Age-Adaptive Thresholds

Usage:
    streamlit run app.py
    
Features:
    - Face detection using InsightFace (buffalo_l model)
    - Face verification using InsightFace buffalo_l or R100 ONNX
    - Age prediction using ViT from model.py
    - Adaptive thresholds based on age gap between faces
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
import cv2

# Import InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'insightface', 'onnxruntime'])
    import insightface
    from insightface.app import FaceAnalysis

# Import age prediction model
try:
    from model import predict_age_gender
except ImportError:
    st.error("⚠️ model.py not found. Please ensure the age prediction model is in the same directory.")
    predict_age_gender = None


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'model': {
        'onnx_path': '../pretrained_models/R100_Glint360K.onnx',
        'insightface_default': 'buffalo_l',  # Default model
        'ctx_id': -1,  # CPU only for Streamlit deployment
        'det_size': (320, 320)  # Larger detection size for buffalo_l
    },
    'inference': {
        'r100_threshold': 0.1998,  # Fallback threshold for R100 ONNX
        'insightface_buffalo_l_threshold': 0.1156,  # Fallback threshold for buffalo_l
        'image_size': [112, 112],
        'normalization': {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        # Age-adaptive thresholds (EER thresholds per age gap)
        'age_adaptive_thresholds': {
            '0-5': 0.3083,
            '5-10': 0.2565,
            '10-20': 0.2130,
            '20-30': 0.1394,
            '30+': 0.1381
        }
    },
    'ui': {
        'title': 'Age-Invariant Face Recognition',
        'description': 'Compare faces across ages with adaptive thresholds'
    }
}

# Model paths
MODEL_PATH = Path(__file__).parent / "../pretrained_models/R100_Glint360K.onnx"
MODEL_PATH = MODEL_PATH.resolve()

# Google Drive direct download URL for R100 ONNX
GDRIVE_FILE_ID = "1XboNCeVBU42B-rz3-bqiY7W6jgsfOTzZ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Face Verification",
    page_icon="👤",
    layout="wide"
)


# ============================================================
# Age-Adaptive Threshold Selection
# ============================================================

def get_adaptive_threshold(age1, age2):
    """
    Get adaptive threshold based on age gap between two faces
    
    Args:
        age1: Age of first person
        age2: Age of second person
    
    Returns:
        tuple: (threshold, age_gap, age_bin)
    """
    age_gap = abs(age1 - age2)
    
    # Determine age bin
    if age_gap <= 5:
        age_bin = '0-5'
    elif age_gap <= 10:
        age_bin = '5-10'
    elif age_gap <= 20:
        age_bin = '10-20'
    elif age_gap <= 30:
        age_bin = '20-30'
    else:
        age_bin = '30+'
    
    threshold = CONFIG['inference']['age_adaptive_thresholds'][age_bin]
    
    return threshold, age_gap, age_bin


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_insightface_model(model_name='buffalo_l'):
    """Load InsightFace model for face detection and recognition (cached)"""
    try:
        app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition']
        )
        app.prepare(
            ctx_id=CONFIG['model']['ctx_id'],
            det_size=CONFIG['model']['det_size']
        )
        return app
    except Exception as e:
        st.error(f"Error loading InsightFace model: {str(e)}")
        return None


@st.cache_resource
def load_r100_onnx_model(model_path):
    """Load R100 ONNX model for face verification (cached)"""
    try:
        # Download model if it doesn't exist
        if not Path(model_path).exists():
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            with st.spinner("Downloading R100 ONNX model (~260 MB)..."):
                gdown.download(GDRIVE_URL, str(model_path), quiet=False)
        
        # Load model
        device = 'cpu'
        model, device = load_onnx_model(str(model_path), device)
        return model, device
    except Exception as e:
        st.error(f"Error loading R100 ONNX model: {str(e)}")
        return None, None


# ============================================================
# Face Detection using InsightFace
# ============================================================

def detect_single_face_insightface(image_path, app):
    """
    Detect exactly one face using InsightFace
    
    Args:
        image_path: Path to image file
        app: InsightFace FaceAnalysis object
    
    Returns:
        tuple: (success, face_crop, bbox, embedding, error_message, original_image)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = app.get(img_bgr)
        
        # Check number of faces
        if len(faces) == 0:
            return False, None, None, None, "❌ No face detected in the image. Please upload an image with a visible face.", image
        
        if len(faces) > 1:
            return False, None, None, None, f"❌ Multiple faces detected ({len(faces)} faces). Please upload an image with only one face.", image
        
        # Exactly one face detected
        face = faces[0]
        
        # Get bounding box
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        # Add padding to bbox for better crop
        padding = int(0.2 * max(x2 - x1, y2 - y1))
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img_array.shape[1], x2 + padding)
        y2_padded = min(img_array.shape[0], y2 + padding)
        
        # Crop face
        face_crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        
        # Get normalized embedding from InsightFace
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return True, face_crop, [x1, y1, x2, y2], embedding, None, image
        
    except Exception as e:
        return False, None, None, None, f"❌ Error processing image: {str(e)}", None


# ============================================================
# Face Verification Functions
# ============================================================

def verify_faces_r100_onnx(face_crop1, face_crop2, model, device, threshold):
    """Verify faces using R100 ONNX model"""
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


def verify_faces_insightface(embedding1, embedding2, threshold):
    """Verify faces using InsightFace embeddings"""
    # Compute cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    # Make decision
    is_same = similarity >= threshold
    
    # Calculate confidence based on distance from threshold
    distance_from_threshold = abs(similarity - threshold)
    
    if distance_from_threshold > 0.2:
        confidence = "High"
    elif distance_from_threshold > 0.1:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Format result
    result = {
        'similarity': float(similarity),
        'threshold': float(threshold),
        'raw_decision': bool(is_same),
        'prediction': "✅ Same Person" if is_same else "❌ Different Person",
        'confidence': confidence,
        'distance_from_threshold': float(distance_from_threshold)
    }
    
    return result


# ============================================================
# Visualization Functions
# ============================================================

def draw_bbox_with_age(image, bbox, age):
    """Draw bounding box and age info on image"""
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
    
    # Calculate text position
    text_y = y1 - 40
    if text_y < 0:
        text_y = y2 + 10
    
    # Draw text background
    text_bbox = draw.textbbox((x1, text_y), age_text, font=font_large)
    draw.rectangle([(text_bbox[0]-5, text_bbox[1]-5), (text_bbox[2]+5, text_bbox[3]+5)], fill="lime")
    
    # Draw text
    draw.text((x1, text_y), age_text, fill="black", font=font_large)
    
    return img_copy


def resize_image_to_fixed_size(image, target_width=500, target_height=500):
    """Resize image to fixed size while maintaining aspect ratio"""
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    final_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    final_image.paste(resized_image, (paste_x, paste_y))
    
    return final_image


# ============================================================
# Main Application
# ============================================================

def main():
    # Custom CSS
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        h1 {
            margin-top: 0;
            padding-top: 0;
        }
        
        .stFileUploader {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .image-container {
            background-color: #333;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
            margin: 10px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .age-gap-info {
            background-color: #1e3a5f;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header
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
        
        # Model selection dropdown
        st.subheader("Recognition Model")
        recognition_model = st.selectbox(
            "Choose verification model",
            options=["InsightFace (buffalo_l)", "R100 ONNX"],
            index=0,  # Default to buffalo_l
            help="Select the model for face verification"
        )
        
        # Set model info based on selected model
        if recognition_model == "InsightFace (buffalo_l)":
            insightface_model_name = 'buffalo_l'
            model_info = f"""
            **Model:** InsightFace (buffalo_l)
            **Type:** Large & Accurate
            **Thresholding:** Age-Adaptive
            **Optimized for:** Age-invariant matching
            """
        else:  # R100 ONNX
            insightface_model_name = 'buffalo_l'  # Still use buffalo_l for detection
            model_info = f"""
            **Model:** R100 (ResNet-100)
            **Dataset:** Glint360K
            **Thresholding:** Age-Adaptive
            **Optimized for:** Age-invariant matching
            """
        
        st.markdown(model_info)
        
        st.markdown("---")
        
        st.info(f"""
        
        📌 **Recognition:** {recognition_model}
        
        📌 **Age Prediction:** Vision Transformer (ViT)
        
        📌 **Thresholding:** Age-Adaptive (0-5, 5-10, 10-20, 20-30, 30+ years)
        
        Images must contain exactly one face.
        """)
        
        st.markdown("---")
        
        # Display age-adaptive thresholds
        with st.expander("📊 Age-Adaptive Thresholds"):
            st.markdown("""
            **EER Thresholds by Age Gap:**
            - **0-5 years:** 0.264
            - **5-10 years:** 0.209
            - **10-20 years:** 0.161
            - **20-30 years:** 0.106
            - **30+ years:** 0.100
            
            The threshold is automatically selected based on the age difference between the two faces.
            """)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_custom_threshold = st.checkbox("Override with Custom Threshold", value=False)
            
            if use_custom_threshold:
                custom_threshold = st.slider(
                    "Verification Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.01,
                    help="Lower threshold = more lenient matching"
                )
            else:
                custom_threshold = None
    
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
        
        # Load InsightFace model (buffalo_l by default)
        with st.spinner(f"Loading InsightFace (buffalo_l) model..."):
            insightface_app = load_insightface_model('buffalo_l')
            if insightface_app is None:
                return
        
        # Load recognition model if R100 ONNX is selected
        if recognition_model == "R100 ONNX":
            with st.spinner("Loading R100 ONNX model..."):
                r100_model, device = load_r100_onnx_model(MODEL_PATH)
                if r100_model is None:
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
            
            # Detect faces using InsightFace
            with st.spinner("Detecting faces with InsightFace (buffalo_l)..."):
                success1, face_crop1, bbox1, embedding1, error1, original_img1 = detect_single_face_insightface(temp_path1, insightface_app)
                success2, face_crop2, bbox2, embedding2, error2, original_img2 = detect_single_face_insightface(temp_path2, insightface_app)
            
            # Check if face detection was successful
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
            
            # Get adaptive threshold based on age gap
            age1 = age_result1['age']
            age2 = age_result2['age']
            
            if custom_threshold is not None:
                threshold = custom_threshold
                age_gap = abs(age1 - age2)
                age_bin = "Custom"
            else:
                threshold, age_gap, age_bin = get_adaptive_threshold(age1, age2)
            
            # # Display age gap and selected threshold
            # st.markdown(
            #     f"""
            #     <div class="age-gap-info">
            #         <h3 style="margin: 0; color: #4CAF50;">🎯 Age-Adaptive Threshold Selected</h3>
            #         <p style="margin: 10px 0 5px 0; font-size: 18px;"><strong>Age Gap:</strong> {age_gap} years</p>
            #         <p style="margin: 5px 0; font-size: 18px;"><strong>Age Bin:</strong> {age_bin}</p>
            #         <p style="margin: 5px 0 0 0; font-size: 18px;"><strong>Threshold:</strong> {threshold:.4f}</p>
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )
            
            # Perform face verification based on selected model
            with st.spinner(f"Verifying identity using {recognition_model}..."):
                try:
                    if recognition_model == "R100 ONNX":
                        result = verify_faces_r100_onnx(
                            face_crop1,
                            face_crop2,
                            r100_model,
                            device,
                            threshold
                        )
                    else:  # InsightFace (buffalo_l)
                        result = verify_faces_insightface(
                            embedding1,
                            embedding2,
                            threshold
                        )
                except Exception as e:
                    st.error(f"❌ Error during verification: {str(e)}")
                    return
            
            # Draw bounding boxes with age info
            annotated_img1 = draw_bbox_with_age(original_img1, bbox1, age_result1['age'])
            annotated_img2 = draw_bbox_with_age(original_img2, bbox2, age_result2['age'])
            
            # Resize images to fixed size
            display_img1 = resize_image_to_fixed_size(annotated_img1, target_width=500, target_height=500)
            display_img2 = resize_image_to_fixed_size(annotated_img2, target_width=500, target_height=500)
        
        # Display results
        st.success("✅ Verification Complete!")
        st.markdown("---")
        
        # Display images side by side
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(display_img1, caption="Image 1", width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Age: {age_result1['age']} years</h3>", unsafe_allow_html=True)

        with img_col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(display_img2, caption="Image 2", width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
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
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Cosine Similarity",
                f"{result['similarity']:.4f}",
                help="Similarity score between face embeddings"
            )
        
        with metric_col2:
            st.metric(
                "Threshold",
                f"{result['threshold']:.4f}",
                help="Age-adaptive decision boundary"
            )
        
        with metric_col3:
            st.metric(
                "Age Gap",
                f"{age_gap} years",
                help="Absolute age difference between faces"
            )
        
        with metric_col4:
            st.metric(
                "Confidence",
                result['confidence'],
                help="Confidence level based on distance from threshold"
            )
        
        # Additional info
        with st.expander("ℹ️ Interpretation Guide"):
            st.markdown(f"""
            **Age-Adaptive Thresholding:**
            - **Age Gap:** {age_gap} years → **Age Bin:** {age_bin}
            - **Selected Threshold:** {threshold:.4f}
            
            **Decision Rule:**
            - **>= {threshold:.4f}**: Predicted as **Same Person**
            - **< {threshold:.4f}**: Predicted as **Different Person**
            
            **Confidence Levels:**
            - **High**: Score is >0.2 away from threshold
            - **Medium**: Score is 0.1-0.2 away from threshold
            - **Low**: Score is <0.1 away from threshold
            
            **Models Used:**
            - **Face Detection:** InsightFace (buffalo_l)
            - **Face Verification:** {recognition_model}
            - **Age Prediction:** Vision Transformer (ViT)
            
            **Age-Adaptive Thresholds (EER-optimized):**
            - **0-5 years:** 0.264
            - **5-10 years:** 0.209
            - **10-20 years:** 0.161
            - **20-30 years:** 0.106
            - **30+ years:** 0.100
            
            All thresholds were optimized on the FG-NET dataset for age-invariant face matching.
            Larger age gaps require lower thresholds to account for appearance changes over time.
            """)
        
        # Debug info
        with st.expander("🔧 Debug Information"):
            st.json({
                "Detection Model": "InsightFace (buffalo_l)",
                "Recognition Model": recognition_model,
                "Image 1": {
                    "Age": age_result1['age'],
                    "Gender": age_result1.get('gender', 'N/A'),
                    "BBox": bbox1
                },
                "Image 2": {
                    "Age": age_result2['age'],
                    "Gender": age_result2.get('gender', 'N/A'),
                    "BBox": bbox2
                },
                "Age-Adaptive Threshold": {
                    "Age Gap": age_gap,
                    "Age Bin": age_bin,
                    "Threshold": float(threshold),
                    "Custom Override": custom_threshold is not None
                },
                "Verification": {
                    "Similarity": float(result['similarity']),
                    "Threshold": float(result['threshold']),
                    "Distance from Threshold": float(result.get('distance_from_threshold', 0)),
                    "Decision": result['raw_decision']
                }
            })


if __name__ == "__main__":
    main()