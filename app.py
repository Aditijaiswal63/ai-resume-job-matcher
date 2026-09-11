import streamlit as st
import re

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume–Job Match Analyzer",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 AI Resume–Job Match Analyzer")

st.write(
    "Analyze your resume against a job description using "
    "semantic similarity and NLP-based skill analysis."
)

st.divider()


# --------------------------------------------------
# PDF TEXT EXTRACTION
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
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

skills = [

    "python",
    "java",
    "c++",
    "c",
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
    "large language model",

    "generative ai",
    "genai",

    "rag",
    "retrieval augmented generation",

    "transformers",

    "scipy",
    "matplotlib",
    "seaborn",

    "keras",

    "data visualization",

    "feature engineering",

    "model deployment",

    "rest api"
]


# --------------------------------------------------
# FIND SKILLS
# --------------------------------------------------

def find_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills:

        if skill in text:

            found_skills.append(skill)

    return sorted(set(found_skills))


# --------------------------------------------------
# SEMANTIC SIMILARITY
# --------------------------------------------------

def calculate_similarity(resume_text, job_text):

    resume_embedding = model.encode(
        [resume_text]
    )

    job_embedding = model.encode(
        [job_text]
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    score = similarity * 100

    return round(score, 2)


# --------------------------------------------------
# SCORE INTERPRETATION
# --------------------------------------------------

def score_message(score):

    if score >= 80:

        return (
            "🔥 Excellent match! "
            "Your resume is highly aligned with this job."
        )

    elif score >= 65:

        return (
            "🟢 Strong match. "
            "Your resume has good alignment with the role."
        )

    elif score >= 50:

        return (
            "🟡 Moderate match. "
            "Some improvements could increase your chances."
        )

    else:

        return (
            "🔴 Low match. "
            "Consider tailoring your resume for this role."
        )


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

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning(
            "Please upload your resume PDF."
        )

    elif not job_description.strip():

        st.warning(
            "Please paste a job description."
        )

    else:

        # ------------------------------------------
        # EXTRACT RESUME
        # ------------------------------------------

        resume_text = extract_text_from_pdf(
            uploaded_file
        )


        if not resume_text.strip():

            st.error(
                "Could not extract text from this PDF. "
                "Try a text-based PDF."
            )

        else:

            # --------------------------------------
            # CLEAN TEXT
            # --------------------------------------

            clean_resume = clean_text(
                resume_text
            )

            clean_job = clean_text(
                job_description
            )


            # --------------------------------------
            # SEMANTIC MATCHING
            # --------------------------------------

            with st.spinner(
                "🤖 Analyzing resume using NLP..."
            ):

                match_score = calculate_similarity(
                    clean_resume,
                    clean_job
                )


            # --------------------------------------
            # SKILL ANALYSIS
            # --------------------------------------

            resume_skills = find_skills(
                resume_text
            )

            job_skills = find_skills(
                job_description
            )


            matched_skills = sorted(
                set(resume_skills)
                &
                set(job_skills)
            )


            missing_skills = sorted(
                set(job_skills)
                -
                set(resume_skills)
            )


            # --------------------------------------
            # SKILL COVERAGE
            # --------------------------------------

            if len(job_skills) > 0:

                skill_coverage = (
                    len(matched_skills)
                    /
                    len(job_skills)
                ) * 100

            else:

                skill_coverage = 0


            skill_coverage = round(
                skill_coverage,
                2
            )


            # --------------------------------------
            # ATS SCORE
            # --------------------------------------

            ats_score = round(
                (match_score * 0.7)
                +
                (skill_coverage * 0.3),
                2
            )


            # --------------------------------------
            # SUCCESS MESSAGE
            # --------------------------------------

            st.success(
                "✅ Resume analysis completed successfully!"
            )


            # --------------------------------------
            # MAIN SCORE
            # --------------------------------------

            st.subheader(
                "🎯 Overall Resume Match"
            )


            st.progress(
             float(min(match_score / 100, 1.0))
            )


            st.metric(
                "Semantic Match Score",
                f"{match_score}%"
            )


            st.info(
                score_message(match_score)
            )


            st.divider()


            # --------------------------------------
            # ATS METRICS
            # --------------------------------------

            st.subheader(
                "📊 Resume Analysis Dashboard"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "ATS Score",
                    f"{ats_score}%"
                )


            with col2:

                st.metric(
                    "Skill Coverage",
                    f"{skill_coverage}%"
                )


            with col3:

                st.metric(
                    "Matched Skills",
                    len(matched_skills)
                )


            with col4:

                st.metric(
                    "Missing Skills",
                    len(missing_skills)
                )


            st.divider()


            # --------------------------------------
            # SKILLS
            # --------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "✅ Matched Skills"
                )


                if matched_skills:

                    for skill in matched_skills:

                        st.write(
                            f"✓ {skill.title()}"
                        )

                else:

                    st.write(
                        "No matching skills detected."
                    )


            with col2:

                st.subheader(
                    "⚠️ Missing Skills"
                )


                if missing_skills:

                    for skill in missing_skills:

                        st.write(
                            f"• {skill.title()}"
                        )

                else:

                    st.write(
                        "🎉 No major missing skills "
                        "were detected."
                    )


            st.divider()


            # --------------------------------------
            # RESUME IMPROVEMENT
            # --------------------------------------

            st.subheader(
                "💡 Resume Improvement Suggestions"
            )


            if missing_skills:

                st.write(
                    "The following job-related skills "
                    "were detected in the job description "
                    "but not in your resume:"
                )


                for skill in missing_skills[:10]:

                    st.write(
                        f"→ **{skill.title()}**"
                    )


                st.warning(
                    "Only add skills you genuinely "
                    "know or have experience with."
                )

            else:

                st.success(
                    "Your resume covers the major "
                    "skills detected in this job."
                )


            st.divider()


            # --------------------------------------
            # TEXT STATISTICS
            # --------------------------------------

            resume_words = len(
                resume_text.split()
            )

            job_words = len(
                job_description.split()
            )


            st.subheader(
                "📄 Document Statistics"
            )


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
                    "Job Skills Detected",
                    len(job_skills)
                )


            st.divider()


            # --------------------------------------
            # PROJECT INFORMATION
            # --------------------------------------

            st.caption(
                "🤖 Semantic matching powered by "
                "Sentence Transformers"
            )

            st.caption(
                "Built with Python • Streamlit • "
                "Scikit-learn • NLP"
            )