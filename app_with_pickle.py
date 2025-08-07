
import streamlit as st
import pickle

# Load pickled model, vectorizer, and encoder
with open("resume_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Resume Classification App")
st.write("Paste a resume below to classify the job category.")

resume_text = st.text_area("Resume Text", height=300)

if st.button("Classify"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        X_input = vectorizer.transform([resume_text])
        prediction = model.predict(X_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Job Category: **{predicted_label.title()}**")
