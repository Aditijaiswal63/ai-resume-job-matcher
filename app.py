import streamlit as st
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume–Job Match Analyzer",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📄 AI Resume–Job Match Analyzer")
st.write(
    "Upload your resume and paste a job description "
    "to check how well your resume matches the job."
)

st.divider()


# --------------------------------------------------
# FUNCTION TO EXTRACT TEXT FROM PDF
# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# FUNCTION TO CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# SKILLS LIST
# --------------------------------------------------

skills = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "computer vision",
    "opencv",
    "power bi",
    "tableau",
    "excel",
    "statistics",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "gcp",
    "fastapi",
    "flask",
    "streamlit",
    "html",
    "css",
    "javascript",
    "react",
    "mongodb",
    "mysql",
    "postgresql",
    "llm",
    "generative ai",
    "genai",
    "rag",
    "transformers"
]


# --------------------------------------------------
# FUNCTION TO FIND SKILLS
# --------------------------------------------------

def find_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills:

        if skill in text:
            found_skills.append(skill)

    return sorted(set(found_skills))


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )


with col2:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the complete job description..."
    )


st.divider()


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("🔍 Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.warning("Please upload your resume PDF.")

    elif not job_description.strip():

        st.warning("Please paste a job description.")

    else:

        # Extract resume text

        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text.strip():

            st.error(
                "Could not extract text from this PDF. "
                "Try a text-based PDF."
            )

        else:

            # Clean both documents

            clean_resume = clean_text(resume_text)

            clean_job = clean_text(job_description)


            # --------------------------------------------------
            # TF-IDF + COSINE SIMILARITY
            # --------------------------------------------------

            documents = [
                clean_resume,
                clean_job
            ]

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = vectorizer.fit_transform(documents)

            similarity = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )[0][0]

            match_score = round(similarity * 100, 2)


            # --------------------------------------------------
            # SKILL ANALYSIS
            # --------------------------------------------------

            resume_skills = find_skills(resume_text)

            job_skills = find_skills(job_description)

            matched_skills = sorted(
                set(resume_skills) & set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) - set(resume_skills)
            )


            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            st.success("Analysis completed successfully!")

            st.subheader("📊 Resume Match Score")

            st.progress(
                min(match_score / 100, 1.0)
            )

            st.metric(
                label="Match Score",
                value=f"{match_score}%"
            )


            st.divider()


            # --------------------------------------------------
            # SKILLS
            # --------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.subheader("✅ Matched Skills")

                if matched_skills:

                    for skill in matched_skills:

                        st.write(f"✓ {skill.title()}")

                else:

                    st.write("No matching skills detected.")


            with col2:

                st.subheader("⚠️ Missing Skills")

                if missing_skills:

                    for skill in missing_skills:

                        st.write(f"• {skill.title()}")

                else:

                    st.write(
                        "Great! No missing skills detected "
                        "from our current skill list."
                    )


            st.divider()


            # --------------------------------------------------
            # RECOMMENDATIONS
            # --------------------------------------------------

            st.subheader("💡 Resume Improvement Suggestions")


            if missing_skills:

                st.write(
                    "Consider adding these skills to your resume "
                    "if you genuinely have experience with them:"
                )

                for skill in missing_skills[:8]:

                    st.write(f"→ {skill.title()}")

            else:

                st.write(
                    "Your resume contains the main skills "
                    "detected in this job description."
                )


            st.divider()


            # --------------------------------------------------
            # TEXT STATISTICS
            # --------------------------------------------------

            resume_words = len(resume_text.split())

            job_words = len(job_description.split())


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Resume Words",
                    resume_words
                )


            with col2:

                st.metric(
                    "Job Description Words",
                    job_words
                )


            with col3:

                st.metric(
                    "Matched Skills",
                    len(matched_skills)
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit and Scikit-learn"
)