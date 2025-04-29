import streamlit as st
import requests
import os
import io
from PIL import Image
import time
import json

# API configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# App configuration
st.set_page_config(
    page_title="Plant Disease Visual Q&A",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388e3c;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #7cb342;
    }
    .stAlert {
        margin-top: 10px;
    }
    .question-btn {
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

# App title and description
st.markdown('<h1 class="main-header">🌿 Plant Disease Visual Q&A</h1>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://i0.wp.com/texasfarmbureau.org/wp-content/uploads/2022/12/plants-sm.png?resize=300%2C300&ssl=1", width=100)
    st.markdown("### About")
    st.write("""
    This application uses AI to identify and answer questions about plant diseases.
    Upload an image of a diseased plant and ask questions to get detailed information.
    """)
    
    st.markdown("### How to use")
    st.write("""
    1. Upload an image of the diseased plant
    2. Type a custom question or select from the template questions
    3. Click 'Ask' to get your answer
    """)
    
    # Check API status
    api_status = check_api_health()
    if api_status:
        st.success("API is online and running")
    else:
        st.error("API is offline. Please check the backend server.")
    
    st.markdown("### Credits")
    st.write("Built with Streamlit, FastAPI, BLIP and Flan-T5 models")

# Create two columns for upload and display
col1, col2 = st.columns([1, 1])

# File uploader in the first column
with col1:
    st.markdown('<h2 class="sub-header">Upload Plant Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Question input and display in the second column
with col2:
    st.markdown('<h2 class="sub-header">Ask a Question</h2>', unsafe_allow_html=True)
    
    # Predefined questions
    questions = [
        "What is the disease?",
        "What is the treatment?",
        "What is the scientific name?",
        "What is the severity?",
        "What are the features?",
        "What are the color changes?",
        "What is the prevention?",
        "What is the impact?",
        "What are the favorable conditions?",
        "What are its spread mechanisms?",
        "What are its resistance strategies?"
    ]
    
    # Question input options
    question_type = st.radio("Question type:", ["Choose from template", "Custom question"])
    
    if question_type == "Choose from template":
        question = st.selectbox("Select a question:", questions)
    else:
        question = st.text_input("Type your question about the plant disease:", "")
    
    # Ask button
    if st.button("🔍 Ask", type="primary", disabled=not uploaded_file or not question):
        if not api_status:
            st.error("Cannot process request: API is offline.")
        else:
            with st.spinner("Analyzing image and generating answer..."):
                try:
                    # Convert image to bytes for the request
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Make API request
                    start_time = time.time()
                    response = requests.post(
                        f"{API_URL}/predict/",
                        files={"image": (uploaded_file.name, img_byte_arr, f"image/{image.format.lower() if image.format else 'jpeg'}")},
                        data={"question": question},
                        timeout=60  # 60-second timeout
                    )
                    end_time = time.time()
                    
                    # Process response
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer returned from the model.")
                        processing_time = result.get("processing_time_seconds", end_time - start_time)
                        
                        # Display answer
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown(f"### Question: {question}")
                        st.markdown(f"### Answer:")
                        st.markdown(answer)
                        st.markdown(f"*Processing time: {processing_time:.2f} seconds*")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    elif response.status_code == 429:
                        st.error("Rate limit exceeded. Please wait a moment before trying again.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The server might be under heavy load or the image might be too complex.")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error. Please check if the API server is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

# Sample images section
st.markdown('<h2 class="sub-header">Sample Questions</h2>', unsafe_allow_html=True)
st.write("If you've uploaded an image, you can quickly ask these common questions:")

# Create a grid of question buttons
cols = st.columns(3)
for i, sample_q in enumerate(questions[:6]):  # Show first 6 questions
    with cols[i % 3]:
        if st.button(f"{sample_q}", key=f"sample_q_{i}", disabled=not uploaded_file):
            # This will automatically set the question and trigger a click on the main "Ask" button
            st.session_state.question = sample_q
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("### Tips for best results")
st.write("""
- Use clear, well-lit images of the affected plant parts
- Try to capture close-ups of the diseased areas
- Ask specific questions about symptoms, treatment, or prevention
- For complex cases, consider asking multiple questions
""")

# Run instructions as a note at the bottom
st.info("""
**Note:** This application requires both the Streamlit frontend and FastAPI backend to be running.
Make sure the backend server is running at the specified API URL.

Run the backend with: `uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000`  
Run this frontend with: `streamlit run streamlit_app.py`
""")