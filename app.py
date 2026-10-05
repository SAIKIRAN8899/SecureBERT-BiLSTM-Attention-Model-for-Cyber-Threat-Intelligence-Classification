import streamlit as st
import torch
import torch.nn as nn
import joblib
import os
from transformers import AutoTokenizer, AutoModel

# --- Model Architecture 
class HybridClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, lstm_hidden, num_classes):
        super().__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(encoder_out)
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.fc(self.dropout(context))


st.set_page_config(page_title="Cybersecurity Threat Classifier", layout="wide")

# --- Session State for Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# --- Constants & Loading ---
MODELS_DIR = "models_outputs"
SECUREBERT_CHECKPOINT = "ehsanaghaei/SecureBERT"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reject low-confidence predictions
CONFIDENCE_THRESHOLD = 0.70

@st.cache_resource
def load_artifacts():
    # Load LabelEncoder
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, 'securebert_tokenizer'))
    # Load Model
    base_encoder = AutoModel.from_pretrained(SECUREBERT_CHECKPOINT)
    model = HybridClassifier(base_encoder, hidden_size=768, lstm_hidden=128, num_classes=len(le.classes_))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'hybrid_secbert_model.pth'), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return le, tokenizer, model

# --- Login / Signup Page ---
def login_page():
    st.title("🔐 Cybersecurity Threat Classifier")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome")
        
        # Tabs for Login and Signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", key="login_btn"):
                if login_username and login_password:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success(f"Welcome back, {login_username}! 🎉")
                    st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        with tab2:
            st.subheader("Create Account")
            signup_username = st.text_input("Choose Username", key="signup_user")
            signup_password = st.text_input("Choose Password", type="password", key="signup_pass")
            signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm")
            
            if st.button("Sign Up", key="signup_btn"):
                if signup_username and signup_password:
                    if signup_password == signup_password_confirm:
                        st.session_state.logged_in = True
                        st.session_state.username = signup_username
                        st.success(f"Account created successfully, {signup_username}! Welcome aboard! 🎉")
                        st.rerun()
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

# --- Threat Classification Page ---
def classifier_page():
    st.title("🛡️ Hybrid SecureBERT Threat Classifier")
    st.markdown(f"Welcome, **{st.session_state.username}**! Input cybersecurity-related text to classify it into threat categories.")
    
    # Logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    le, tokenizer, model = load_artifacts()
    
    # --- UI Layout ---
    input_text = st.text_area("Enter Cyber Threat Intelligence (CTI) text:", height=150,
                              placeholder="e.g., A cybersquatting domain is launching DoS attacks...")

    if st.button("Classify Threat"):
        if input_text.strip():
            with st.spinner("Analyzing..."):

                inputs = tokenizer(
                    input_text,
                    padding='max_length',
                    truncation=True,
                    max_length=64,
                    return_tensors='pt'
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model(inputs['input_ids'], inputs['attention_mask'])
                    probs = torch.softmax(outputs, dim=1)

                    confidence, pred_idx = torch.max(probs, dim=1)

                confidence = confidence.item()
                pred_idx = pred_idx.item()

                # Confidence-based rejection
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚪ Not a Cyber Threat")
                    st.write(f"Confidence Score: {confidence:.2%}")

                else:
                    label = le.inverse_transform([pred_idx])[0]

                    st.subheader(f"Prediction: **{label}**")
                    st.progress(confidence)
                    st.write(f"Confidence Score: {confidence:.2%}")

    else:
        st.warning("Please enter some text to classify.")

# --- Main App Logic ---
if st.session_state.logged_in:
    classifier_page()
else:
    login_page()
