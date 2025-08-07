
import streamlit as st
import pickle
import re
from docx import Document
import pandas as pd
import io

# Load model, vectorizer, and label encoder
with open("resume_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Resume Classifier with Feature Extraction & Download")
st.write("Upload a resume (.docx) to extract features and predict the job category.")

uploaded_file = st.file_uploader("Choose a DOCX resume file", type=["docx"])

def read_docx(file):
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def extract_name(text):
    lines = text.strip().split('\n')
    return lines[0] if lines else "Not found"

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group() if match else "Not found"

def extract_phone(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group() if match else "Not found"

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
    match = re.search(r"(\d+\+?)\s+years?", text.lower())
    return match.group() if match else "Not found"

if uploaded_file is not None:
    resume_text = read_docx(uploaded_file)

    if resume_text.strip() == "":
        st.error("The uploaded file seems empty or unreadable.")
    else:
        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        skills = extract_skills(resume_text)
        experience = extract_experience(resume_text)

        # Predict job category
        X_input = vectorizer.transform([resume_text])
        prediction = model.predict(X_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display features
        st.subheader("📄 Extracted Features:")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Skills:** {skills}")
        st.write(f"**Experience:** {experience}")
        st.success(f"🎯 **Predicted Job Category:** {predicted_label.title()}")

        # Prepare data for download
        result_data = {
            "Name": [name],
            "Email": [email],
            "Phone": [phone],
            "Skills": [skills],
            "Experience": [experience],
            "Predicted Category": [predicted_label.title()]
        }
        df = pd.DataFrame(result_data)

        # Download as text
        text_output = f"""
Name: {name}
Email: {email}
Phone: {phone}
Skills: {skills}
Experience: {experience}
Predicted Category: {predicted_label.title()}
"""
        st.download_button("📄 Download as Text", text_output, file_name="resume_features.txt")

        # Download as Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        st.download_button("📊 Download as Excel", excel_buffer.getvalue(), file_name="resume_features.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
