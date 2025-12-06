import streamlit as st
import pdfplumber
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load vectorizer model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📄 Resume Shortlisting System")
st.write("Upload resume and match against job description")

# Upload PDF resume
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# Job description input box
job_description = st.text_area("Paste Job Description Here")

# Function to extract text from uploaded resume
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# Action button
if st.button("Check Match"):
    if resume_file is None:
        st.error("⚠ Please upload a resume")
    elif job_description.strip() == "":
        st.error("⚠ Please paste job description")
    else:
        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)

        # Convert to vectors
        vectors = vectorizer.transform([resume_text, job_description])

        # Cosine similarity calculation
        similarity_score = cosine_similarity(vectors)[0][1] * 100

        # Display match score
        st.subheader(f"🎯 Matching Score: {similarity_score:.2f}%")

        # Missing keyword logic
        jd_words = job_description.lower().split()
        resume_words = resume_text.lower().split()

        missing = [word for word in jd_words if word not in resume_words]

        st.write("---")
        st.write("🟠 Missing Keywords:")
        if missing:
            st.write(missing)
        else:
            st.success("🎉 Resume has all important keywords!")
