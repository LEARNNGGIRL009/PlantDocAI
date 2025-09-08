# PlantDoc AI - Streamlit Mobile Web App (Fixed Version)
# Run with: streamlit run plantdoc_streamlit.py

import streamlit as st
import os
import json
from datetime import datetime
import requests
from PIL import Image
import io

# Import your PlantDoc AI
try:
    from plantdoc_core import PlantDocAI
except ImportError:
    st.error("Please make sure 'plantdoc_core.py' is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🌱 PlantDoc AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile-friendly design
# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .disease-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .healthy { border-left: 5px solid #2ecc71; background-color: #d4edda;  color: #000000 !important;  }
    .medium-risk { border-left: 5px solid #f39c12; background-color: #fff3cd;  color: #000000 !important;  }
    .high-risk { border-left: 5px solid #e67e22; background-color: #ffeaa7;  color: #000000 !important;  }
    .critical-risk { border-left: 5px solid #e74c3c; background-color: #f8d7da;  color: #000000 !important; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* ADD THIS TREATMENT BOX STYLING */
    .treatment-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        color: #000000 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .treatment-text {
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.8rem; }
        .metric-card { margin: 0.5rem 0; }
        .treatment-box { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCTION DEFINITIONS (MOVED TO TOP) ====================
def display_results(result, is_demo=False):
    """Display analysis results in mobile-friendly format"""
    
    # Determine risk class for styling
    risk_class = "healthy"
    if "Critical" in result['severity']:
        risk_class = "critical-risk"
    elif "High Risk" in result['severity']:
        risk_class = "high-risk"
    elif "Medium Risk" in result['severity']:
        risk_class = "medium-risk"
    
    # Main result card (FIXED - was showing treatment instead of disease info)
    st.markdown(f"""
    <div class="disease-card {risk_class}">
        <h2>🦠 {result['disease']}</h2>
        <h3>📊 Confidence: {result['confidence']:.1%}</h3>
        <p><strong>⚠️ Assessment:</strong> {result['severity']}</p>
        {('<p><strong>🔬 Early Detection:</strong> Potential early-stage symptoms detected</p>' if result.get('early_detection') else '')}
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    if 'features' in result and result['features']:
        st.subheader("📈 Image Analysis")
        
        # Create 4 columns for metrics
        metric_cols = st.columns(4)
        features = result['features']
        
        metrics_data = [
            ("🟢 Green", features.get('green_ratio', 0), "%"),
            ("🟡 Yellow", features.get('yellow_ratio', 0), "%"),
            ("🟤 Brown", features.get('brown_ratio', 0), "%"),
            ("⚫ Dark", features.get('dark_spot_ratio', 0), "%")
        ]
        
        for i, (label, value, unit) in enumerate(metrics_data):
            with metric_cols[i]:
                st.metric(label, f"{value:.1f}{unit}")
    
    # Treatment recommendation (FIXED - now uses your CSS classes)
    st.subheader("💊 Treatment Recommendation")
    st.markdown(f"""
    <div class="treatment-box">
        <div class="treatment-text">
            {result['treatment']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("📄 Generate Report", key=f"report_{datetime.now().strftime('%H%M%S')}"):
            create_mobile_report(result)
    
    with col_b:
        if st.button("📱 Share Results", key=f"share_{datetime.now().strftime('%H%M%S')}"):
            share_results(result)
    
    with col_c:
        if st.button("🔄 Analyze Another", key=f"another_{datetime.now().strftime('%H%M%S')}"):
            st.rerun()
    
    # Success message for demo
    if is_demo:
        st.success("🎬 This was a demo analysis! Upload your own image above.")

def create_mobile_report(result):
    """Create and download mobile-friendly report"""
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlantDoc AI Report - {result['disease']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background: #2ecc71; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .result {{ background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 8px; }}
            .critical {{ border-left: 5px solid #e74c3c; }}
            .high {{ border-left: 5px solid #f39c12; }}
            .medium {{ border-left: 5px solid #f39c12; }}
            .healthy {{ border-left: 5px solid #2ecc71; }}
            @media (max-width: 600px) {{ body {{ margin: 10px; }} .header h1 {{ font-size: 1.5rem; }} }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌱 PlantDoc AI Report</h1>
            <p>Mobile Plant Disease Analysis</p>
        </div>
        
        <div class="result {'critical' if 'Critical' in result['severity'] else 'high' if 'High Risk' in result['severity'] else 'medium' if 'Medium Risk' in result['severity'] else 'healthy'}">
            <h2>🦠 Disease: {result['disease']}</h2>
            <p><strong>📊 Confidence:</strong> {result['confidence']:.1%}</p>
            <p><strong>⚠️ Severity:</strong> {result['severity']}</p>
            <p><strong>🕒 Analysis Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="result">
            <h2>💊 Treatment</h2>
            <p>{result['treatment']}</p>
        </div>
        
        <div class="result">
            <h2>⚠️ Important</h2>
            <p>This is an AI-assisted diagnosis. For critical diseases, consult agricultural specialists immediately.</p>
        </div>
    </body>
    </html>
    """
    
    # Create download button
    st.download_button(
        label="📄 Download Report",
        data=html_content,
        file_name=f"plantdoc_report_{result['disease'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )
    
    st.success("📄 Report ready for download!")

def share_results(result):
    """Create shareable text summary"""
    
    share_text = f"""
🌱 PlantDoc AI Analysis Results

🦠 Disease: {result['disease']}
📊 Confidence: {result['confidence']:.1%}
⚠️ Risk Level: {result['severity']}

💊 Treatment: {result['treatment'][:100]}...

🤖 Analyzed with PlantDoc AI - Mobile Plant Disease Detection

#PlantDoc #AI #Agriculture #UAE #Hydroponics
    """.strip()
    
    st.text_area(
        "📱 Share this text:",
        value=share_text,
        height=200,
        help="Copy this text to share on social media"
    )
    
    st.success("📱 Copy the text above to share your results!")

# ==================== MAIN APP ====================

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'plantdoc' not in st.session_state:
    st.session_state.plantdoc = PlantDocAI()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌱 PlantDoc AI</h1>
    <p>Mobile Plant Disease Detection</p>
    <p>Detect 5 tomato diseases instantly with AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Quick Stats")
    
    # Disease classes info
    st.subheader("🦠 Detectable Diseases:")
    diseases_info = [
        ("🟢 Healthy", "No Risk"),
        ("🟡 Spider Mites", "Medium Risk"),
        ("🟡 Bacterial Spot", "Medium Risk"),
        ("🟠 Curl Virus", "High Risk"),
        ("🔴 Late Blight", "Critical Risk")
    ]
    
    for disease, risk in diseases_info:
        st.write(f"{disease} - {risk}")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.subheader("📈 Recent Analysis")
        for i, result in enumerate(st.session_state.analysis_history[-3:], 1):
            confidence_emoji = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
            st.write(f"{confidence_emoji} {result['disease']} ({result['confidence']:.0%})")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📷 Upload Plant Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a tomato leaf for analysis"
    )
    
    # Camera input (works on mobile browsers)
    camera_image = st.camera_input("📸 Or take a photo with your camera")
    
    # Use camera image if available, otherwise use uploaded file
    image_to_analyze = camera_image if camera_image else uploaded_file
    
    if image_to_analyze:
        # Display image
        image = Image.open(image_to_analyze)
        st.image(image, caption="Image to analyze", use_column_width=True)
        
        # Analysis button
        if st.button("🔍 Analyze Disease", key="analyze_btn"):
            with st.spinner("🤖 AI is analyzing your plant..."):
                # Save image temporarily
                temp_path = f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image.save(temp_path)
                
                try:
                    # Run analysis
                    result = st.session_state.plantdoc.analyze_disease(temp_path)
                    
                    if result:
                        # Add to history
                        st.session_state.analysis_history.append(result)
                        
                        # Display results
                        display_results(result)
                    else:
                        st.error("❌ Analysis failed. Please try another image.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

with col2:
    st.header("🎯 Quick Demo")
    
    if st.button("🎬 Run Demo"):
        demo_result = {
            'disease': 'Late Blight',
            'confidence': 0.89,
            'severity': 'Critical Risk - High Confidence',
            'treatment': 'CRITICAL - ACT IMMEDIATELY: Remove and destroy affected plants entirely.',
            'features': {
                'green_ratio': 35.2,
                'yellow_ratio': 15.8,
                'brown_ratio': 28.4,
                'dark_spot_ratio': 20.6,
                'brightness': 98.3
            },
            'early_detection': False,
            'timestamp': datetime.now().isoformat()
        }
        
        display_results(demo_result, is_demo=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌱 <strong>PlantDoc AI</strong> - Mobile Plant Disease Detection</p>
    <p>Built with ❤️ for sustainable agriculture | Powered by Roboflow AI</p>
    <p>📧 Perfect for: Farmers, Students, Researchers, Garden Enthusiasts</p>
</div>

""", unsafe_allow_html=True)
