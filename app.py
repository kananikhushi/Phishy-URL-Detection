import streamlit as st
import pandas as pd
import joblib

# ========== Load ==========
model = joblib.load('rf_model.joblib')
feature_names = joblib.load('rf_features.joblib')

# ========== Feature Extractor ==========
def extract_features(url):
    return {
        'url_length': len(url),
        'has_https': int('https' in url.lower()),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_digits': sum(c.isdigit() for c in url),
        'has_suspicious_words': int(any(word in url.lower() for word in ['login', 'verify', 'update', 'secure', 'account', 'free'])),
        'count_special_chars': sum(c in url for c in ['@', '=', '?', '&']),
    }

# ========== Streamlit UI ==========
st.title("🔍 Phishy URL Checker")
st.write("Enter a URL below to find out if it's malicious or safe.")

url_input = st.text_input("Enter URL:")

if st.button("Check URL"):
    if url_input:
        features = pd.DataFrame([extract_features(url_input)])
        features = features[feature_names]
        pred = model.predict(features)[0]

        if pred == 1:
            st.error("🚨 Malicious URL detected!")
        else:
            st.success("✅ Safe URL.")
    else:
        st.warning("⚠ Please enter a URL.")
