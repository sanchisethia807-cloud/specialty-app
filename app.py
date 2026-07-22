import streamlit as st
import joblib
import pandas as pd
import random

# Load the trained model and vectorizer
model = joblib.load('specialty_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
sample_notes = pd.read_csv('sample_notes.csv')

# --- Specialty detail content (expand this over time) ---
SPECIALTY_INFO = {
    "Surgery": "General surgery covers operative procedures — notes here describe pre-op diagnoses, the operation performed, anesthesia, blood loss, and post-op condition. This is the model's largest and most dominant category.",
    "Consult - History and Phy.": "A History & Physical (H&P) is the structured intake document a clinician writes when first evaluating a patient: history of present illness, past medical/surgical history, review of systems, and physical exam. It's a document *format* more than a clinical field.",
    "Cardiovascular / Pulmonary": "Covers the heart, blood vessels, and lungs — conditions like heart failure, hypertension, arrhythmias, COPD, and pneumonia.",
    "Orthopedic": "Deals with the musculoskeletal system — bones, joints, ligaments, and muscles. Think fractures, joint pain, arthritis, and sports injuries.",
    "Neurology": "Focuses on the brain, spinal cord, and nervous system — conditions like seizures, migraines, stroke, and multiple sclerosis.",
    "Obstetrics / Gynecology": "Women's reproductive health and pregnancy — prenatal care, deliveries, and gynecologic conditions.",
    "Gastroenterology": "The digestive system — esophagus, stomach, intestines, liver — covering issues like reflux, constipation, and IBD.",
    "Urology": "The urinary tract and male reproductive system — kidney stones, UTIs, prostate conditions.",
    "Radiology": "Interpretation of medical imaging — X-rays, CT, MRI, ultrasound — to diagnose disease.",
    "General Medicine": "Broad internal-medicine notes not tied to one organ system; often primary-care style visits.",
    "Discharge Summary": "A document type: the summary written when a patient leaves the hospital, recapping their stay, treatment, and follow-up plan.",
    "SOAP / Chart / Progress Notes": "A document format (Subjective, Objective, Assessment, Plan) used for ongoing visit notes rather than a clinical specialty.",
}

def render_detail(specialty):
    """The detail 'page' for one specialty. Enrich this later with ICD-10, tables, charts."""
    if st.button("← Back to results"):
        st.session_state.selected_specialty = None
        st.rerun()

    st.header(specialty)
    st.write(SPECIALTY_INFO.get(specialty, "No description available yet for this category."))

    # --- Future: ICD-10 codes, tables, charts go here ---
    st.caption("More detail (ICD-10 codes, related conditions) coming soon.")


st.title("Medical Specialty Predictor")

# initialize navigation state
if "selected_specialty" not in st.session_state:
    st.session_state.selected_specialty = None

tab1, tab2 = st.tabs(["Clinical Notes", "Symptom Checklist (coming soon)"])

with tab1:
    # If a specialty is selected, show its detail page INSTEAD of the main view
    if st.session_state.selected_specialty:
        render_detail(st.session_state.selected_specialty)
    else:
        st.write("Paste in a clinical note below, and the model will guess which medical specialty it belongs to.")

        if st.button("🎲 Simulate with Random Sample"):
            random_note = sample_notes.sample(1).iloc[0]
            st.session_state.user_note = random_note['note']

        user_note = st.text_area("Clinical Note", height=200, key="user_note")

        if st.button("Predict Specialty", key="notes_predict"):
            if user_note.strip() == "":
                st.warning("Please paste in a note first.")
            else:
                note_vec = vectorizer.transform([user_note])
                probabilities = model.predict_proba(note_vec)[0]
                specialty_names = model.classes_
                results = sorted(zip(specialty_names, probabilities), key=lambda x: x[1], reverse=True)
                st.session_state.last_results = results[:3]

        # Render cards if we have results (kept in session_state so they survive reruns)
        if "last_results" in st.session_state:
            st.subheader("Most Likely Specialties")
            for specialty, prob in st.session_state.last_results:
                pct = prob * 100
                st.markdown(f"""
                <div style="background:#1a1a2e; border:1px solid #2a2a4a; border-radius:12px;
                            padding:16px 20px; margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;
                              margin-bottom:10px;">
                    <span style="font-size:1.05rem; font-weight:600; color:#fff;">{specialty}</span>
                    <span style="font-size:1.05rem; font-weight:700; color:#ff6b6b;">{pct:.1f}%</span>
                  </div>
                  <div style="background:#2a2a4a; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%;
                                background:linear-gradient(90deg,#ff6b6b,#ff9e7d);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"View details →", key=f"detail_{specialty}"):
                    st.session_state.selected_specialty = specialty
                    st.rerun()

with tab2:
    st.write("Coming soon — structured symptom selection.")
