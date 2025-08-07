
import streamlit as st
import pickle
import re

# Load pickled model, vectorizer, and encoder
with open("resume_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Resume Classification App")
st.write("Paste a resume below to extract features and predict the job category.")

resume_text = st.text_area("Resume Text", height=300)

def extract_name(text):
    lines = text.strip().split('\n')
    if lines:
        return lines[0]
    return "Not found"

def extract_skills(text):
    skills_keywords = [
        "python", "java", "sql", "react", "machine learning", "data analysis",
        "javascript", "html", "css", "excel", "communication", "leadership"
    ]
    found_skills = set()
    text_lower = text.lower()
    for skill in skills_keywords:
        if skill in text_lower:
            found_skills.add(skill)
    return ", ".join(found_skills) if found_skills else "Not found"

def extract_experience(text):
    match = re.search(r'(\d+\+?)\s+years?', text.lower())
    return match.group() if match else "Not found"

if st.button("Analyze & Classify"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        name = extract_name(resume_text)
        skills = extract_skills(resume_text)
        experience = extract_experience(resume_text)

        X_input = vectorizer.transform([resume_text])
        prediction = model.predict(X_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Extracted Features:")
        st.write(f"**Name:** {name}")
        st.write(f"**Skills:** {skills}")
        st.write(f"**Experience:** {experience}")
        st.write(f"**Predicted Category:** 🎯 {predicted_label.title()}")
