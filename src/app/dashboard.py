"""DocuShield AI - Interactive Streamlit Web Interface.

Developed by Mohammed AL-Haj
Run via: streamlit run src/app/dashboard.py
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.problem3_privacy.pii_detectors import detect_all_pii, guess_language
from src.problem3_privacy.redaction import redact_text

st.set_page_config(
    page_title="DocuShield AI — Document Privacy & Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DocuShield AI")
st.subheader("Multilingual & Arabic Privacy-Preserving Document Intelligence")
st.caption("Engineered & Developed by **Mohammed AL-Haj**")

st.markdown("""
---
**DocuShield AI** automatically detects sensitive Personal Identifiable Information (**PII**), 
including **Saudi National IDs, GCC/Yemen Phone Numbers, Emails, Medical Record Numbers, and Names**, 
and allows instant deterministic redaction for GDPR & Saudi Data Privacy (PDPL) compliance.
""")

col1, col2 = st.columns([1, 1])

SAMPLE_ARABIC = """تقرير فحص مخبري سري
المريض: محمد عبدالله القحطاني
رقم الهوية الوطنية: 1089234567
رقم الملف الطبي: MRN-89412
تاريخ المراجعة: 2026-04-12
رقم الجوال: 0551234567
البريد الإلكتروني: m.alqahtani@example.sa
التشخيص المبدئي: التهاب حاد في الجيوب الأنفية، يرجى متابعة العلاج مع دكتور أحمد الزهراني."""

SAMPLE_ENGLISH = """CLINICAL EXAMINATION REPORT
Patient Name: John Doe
SSN: 123-45-6789
Medical Record Number: MD-87483
Date of Birth: 15/03/1982
Contact Phone: (555) 123-4567
Email Address: john.doe@healthcorp.org
Diagnosis: Patient diagnosed with Type 2 Diabetes Mellitus under supervision of Dr. Sarah Jenkins."""

with col1:
    st.markdown("### 📄 Input Document Text")
    sample_choice = st.radio("Load Sample Document:", ["Arabic (تقرير طبي سعودي)", "English (Medical Report)", "Custom Text"], horizontal=True)
    
    if sample_choice == "Arabic (تقرير طبي سعودي)":
        default_text = SAMPLE_ARABIC
    elif sample_choice == "English (Medical Report)":
        default_text = SAMPLE_ENGLISH
    else:
        default_text = ""
        
    input_text = st.text_area("Document Content:", value=default_text, height=280)
    
    st.markdown("### ⚙️ Redaction Configuration")
    mode = st.selectbox("Redaction Strategy:", ["typed", "strict", "surrogate"], index=0, 
                        help="'typed' replaces with [TYPE], 'strict' with [REDACTED], 'surrogate' with fake realistic values.")
    arabic_labels = st.checkbox("Use Arabic Tags (e.g. [هوية_وطنية], [محجوب])", value=True)

with col2:
    st.markdown("### 🔍 Analysis & Redaction Results")
    if st.button("🚀 Analyze & Redact Document", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please provide document text to analyze.")
        else:
            lang = guess_language(input_text)
            pii_matches = detect_all_pii(input_text)
            redacted_result, _ = redact_text(
                input_text, 
                pii_matches, 
                mode=mode, 
                language=lang,
                arabic_labels=arabic_labels or (lang == "ar")
            )
            
            st.success(f"Detected **{len(pii_matches)}** sensitive PII entities in **{lang.upper()}** language context.")
            
            # Display detected entities table
            if pii_matches:
                with st.expander("📋 View Detected Entities Summary", expanded=True):
                    st.dataframe([
                        {"Type": p["type"], "Extracted Value": p["text"], "Start": p["start"], "End": p["end"]}
                        for p in pii_matches
                    ], use_container_width=True)
            
            st.markdown("#### 🔒 Anonymized Output:")
            st.text_area("Protected Content:", value=redacted_result, height=220)
            
            st.download_button(
                label="📥 Download Redacted Document",
                data=redacted_result,
                file_name="redacted_document.txt",
                mime="text/plain"
            )

st.markdown("---")
st.markdown("💡 *Powered by PyTorch, LayoutLMv3, XLM-RoBERTa, and FastAPI.*")
