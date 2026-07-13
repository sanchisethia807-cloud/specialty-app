import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('specialty_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Medical Specialty Predictor")
st.write("Paste in a clinical note below, and the model will guess which medical specialty it belongs to.")

# Text box for user input
user_note = st.text_area("Clinical Note", height=200)

if st.button("Predict Specialty"):
    if user_note.strip() == "":
        st.warning("Please paste in a note first.")
    else:
        # Translate the note into numbers, then predict
        note_vec = vectorizer.transform([user_note])
        prediction = model.predict(note_vec)[0]
        
        st.success(f"Predicted Specialty: **{prediction}**")

