import streamlit as st
import joblib
import time

@st.cache_resource
def load_model():
    return joblib.load("stress_model.pkl")

st.set_page_config(page_title="Bangla Stress Detection", page_icon="🧠", layout="centered")

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"মডেল লোড করতে সমস্যা হচ্ছে: {e}")
    st.stop()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Beautiful Modern SaaS UI */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    .main-header {
        text-align: center;
        padding: 10px 0 5px 0;
        animation: fadeIn 0.8s ease-in-out;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Text Area */
    .stTextArea>label {
        font-size: 1rem !important;
        color: #334155 !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    }

    /* Beautiful Button */
    .stButton>button {
        background: linear-gradient(to right, #4f46e5, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        margin-top: 5px;
        box-shadow: 0 5px 10px -3px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4) !important;
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Results Card */
    .result-card {
        padding: 15px 20px;
        border-radius: 16px;
        text-align: center;
        margin-top: 10px;
        background-color: #ffffff;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stress-bg {
        border-bottom: 5px solid #ef4444;
    }
    
    .non-stress-bg {
        border-bottom: 5px solid #10b981;
    }

    .result-icon {
        font-size: 2.5rem;
        margin-bottom: 5px;
        display: block;
        animation: scaleIn 0.5s ease-out;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .stress-text { color: #dc2626; }
    .non-stress-text { color: #059669; }
    
    .confidence {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 5px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 20px;
        padding-top: 15px;
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Keyframes */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scaleIn {
        from { transform: scale(0.5); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">Bangla Stress Detection</div>
    <div class="main-subtitle">আপনার লেখা থেকে মানসিক চাপ বা স্ট্রেস নির্ণয় করুন</div>
</div>
""", unsafe_allow_html=True)

# Input area
user_input = st.text_area(
    "💬 এখানে আপনার যেকোনো লেখা বা স্ট্যাটাস দিন (বাংলা বা বাংলিশ):",
    height=80,
    placeholder="উদাহরণ: আজকাল কোনো কাজেই মন বসাতে পারছি না..."
)

if st.button("ফলাফল চেক করুন"):
    if user_input.strip() == "":
        st.error("⚠️ অনুগ্রহ করে বক্সে কিছু লিখুন!")
    else:
        with st.spinner('একটু অপেক্ষা করুন, অ্যানালাইসিস চলছে...'):
            time.sleep(1)
            
            prediction = pipeline.predict([user_input])[0]
            probabilities = pipeline.predict_proba([user_input])[0]
            
            if prediction == 1:
                confidence = probabilities[1] * 100
                st.markdown(f"""
                <div class='result-card stress-bg'>
                    <span class='result-icon'>😟</span>
                    <div class='result-title stress-text'>মানসিক চাপে আছেন (Stressed)</div>
                    <div class='confidence'>মডেলের আত্মবিশ্বাস: <b>{confidence:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = probabilities[0] * 100
                st.markdown(f"""
                <div class='result-card non-stress-bg'>
                    <span class='result-icon'>✅</span>
                    <div class='result-title non-stress-text'>স্বাভাবিক আছেন (Normal)</div>
                    <div class='confidence'>মডেলের আত্মবিশ্বাস: <b>{confidence:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by <b>Support Vector Machine (SVM)</b> & <b>BanglaBERT</b><br>
    Developed by <b>Istiaq Rahman Mahi</b></p>
</div>
""", unsafe_allow_html=True)
