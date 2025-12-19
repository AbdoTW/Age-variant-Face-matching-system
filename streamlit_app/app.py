"""
app.py - Streamlit Face Verification Application

Usage:
    streamlit run app.py
"""

import streamlit as st
import yaml
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
else:
    st.success(f"ONNX model already exists at {MODEL_PATH}")




# Page configuration
st.set_page_config(
    page_title="Face Verification",
    page_icon="👤",
    layout="wide"
)


@st.cache_resource
def load_config():
    """Load configuration from YAML file"""
    try:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.warning(f"Could not load config.yaml: {e}. Using defaults.")
        return {
            'model': {'path': '../pretrained_models/R100_Glint360K.onnx'},
            'inference': {'threshold': 0.1998},
            'ui': {
                'title': 'Face Verification App',
                'description': 'Upload two face images to determine if they belong to the same person'
            }
        }


@st.cache_resource
def load_model(model_path):
    """Load model (cached to avoid reloading)"""
    # Convert relative path to absolute
    abs_model_path = Path(__file__).parent / model_path
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model, device = load_onnx_model(str(abs_model_path), device)
    return model, device


def verify_faces(image_path1, image_path2, model, device, threshold):
    """
    Main verification function
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        model: Loaded model
        device: Device for inference
        threshold: Decision threshold
    
    Returns:
        result: Dictionary with verification results
        img1: Original PIL image 1
        img2: Original PIL image 2
    """
    # Preprocess images
    tensor1, img1 = preprocess_image(image_path1)
    tensor2, img2 = preprocess_image(image_path2)
    
    # Extract features
    features1 = extract_features(model, tensor1, device)
    features2 = extract_features(model, tensor2, device)
    
    # Compute similarity
    similarity = compute_cosine_similarity(features1, features2)
    
    # Predict
    is_same, confidence = predict_same_person(similarity, threshold)
    
    # Format result
    result = format_result(similarity, threshold, is_same, confidence)
    
    return result, img1, img2


def main():
    # Load configuration
    config = load_config()
    
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
            <h1 style="font-size: 36px; margin: 10px 0; padding: 0; color: #ffffff;">{config['ui']['title']}</h1>
            <p style="margin: 5px 0; font-size: 16px; color: #aaaaaa;">{config['ui']['description']}</p>
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
            value=config['model']['path'],
            help="Path to ONNX model file (relative to app.py)"
        )
        
        # Use fixed threshold from config
        threshold = config['inference']['threshold']
        
        st.markdown("---")
        # st.markdown("**Device Info:**")
        # device_info = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
        # st.markdown(f"{device_info}")
    
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
            
            # Load model
            with st.spinner("Loading model..."):
                try:
                    model, device = load_model(model_path)
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    return
            
            # Perform verification
            with st.spinner("Processing images..."):
                try:
                    result, img1, img2 = verify_faces(
                        temp_path1,
                        temp_path2,
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
        
        # Display images side by side
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
            st.image(img1, caption="Image 1", width=400)
            st.markdown("</div>", unsafe_allow_html=True)

        with img_col2:
            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
            st.image(img2, caption="Image 2", width=400)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display results
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
            
            **Note:** The threshold (0.1998) was determined from base model evaluation 
            to achieve optimal accuracy on the FG-NET dataset.
            """)


if __name__ == "__main__":
    main()