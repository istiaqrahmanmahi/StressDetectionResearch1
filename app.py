import streamlit as st
import joblib

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("stress_model.pkl")

# Set page config
st.set_page_config(page_title="Bangla Stress Detection", page_icon="🧠", layout="centered")

# Load model
try:
    pipeline = load_model()
except Exception as e:
    st.error(f"মডেল লোড করতে সমস্যা হচ্ছে: {e}")
    st.stop()

# Custom CSS for aesthetic design
st.markdown("""
<style>
    .stress-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #ff4b4b, #ff7676);
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .non-stress-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #00cc96, #21e2b2);
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 204, 150, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Bangla/Banglish Stress Detection")
st.markdown("<p style='text-align: center; color: #a1a1aa; font-size: 18px;'>Automated Stress Detection in Bengali and Banglish Text</p>", unsafe_allow_html=True)
st.markdown("---")

user_input = st.text_area("✍️ আপনার টেক্সট (বাংলা বা বাংলিশ) এখানে লিখুন:", height=150, placeholder="উদাহরণ: মাঝে মাঝে ক্যারিয়ার নিয়ে মাথা কাজ করছে না।")

if st.button("🔍 চেক করুন", use_container_width=True):
    if user_input.strip() == "":
        st.warning("অনুগ্রহ করে বক্সের ভেতর কিছু টেক্সট লিখুন!")
    else:
        # Predict using the loaded pipeline
        prediction = pipeline.predict([user_input])[0]
        probabilities = pipeline.predict_proba([user_input])[0]
        
        st.markdown("### ফলাফল (Result):")
        
        if prediction == 1:
            st.markdown("<div class='stress-box'>⚠️ মানসিক চাপে আছেন (Stressed)</div>", unsafe_allow_html=True)
            confidence = probabilities[1] * 100
            st.info(f"**মডেলের আত্মবিশ্বাস (Confidence):** {confidence:.2f}%")
        else:
            st.markdown("<div class='non-stress-box'>✅ স্বাভাবিক আছেন (Non-Stressed)</div>", unsafe_allow_html=True)
            confidence = probabilities[0] * 100
            st.info(f"**মডেলের আত্মবিশ্বাস (Confidence):** {confidence:.2f}%")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px; color: #888;'>Powered by Machine Learning (SVM) | Developed for Research</p>", unsafe_allow_html=True)
