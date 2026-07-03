"""
Streamlit Web Application for Resume Screening
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import tempfile
from io import StringIO

# Add parent directory to path
sys.path.append(".")

from src.parser import ResumeParser
from src.preprocess import TextPreprocessor
from src.similarity import SimilarityCalculator
from src.ranking import CandidateRanker

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
    }
    .stButton button:hover {
        transform: scale(1.02);
        transition: transform 0.2s;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize components
@st.cache_resource
def init_components():
    """Initialize all components with caching"""
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    calculator = SimilarityCalculator()
    ranker = CandidateRanker()
    return parser, preprocessor, calculator, ranker


# Initialize
parser, preprocessor, calculator, ranker = init_components()

# ====================================================================
# PROCESSING FUNCTIONS
# ====================================================================


def process_resume_text(text):
    """Process a single resume text"""
    if not text:
        return ""
    return preprocessor.get_full_preprocessed_text(text)


def process_jd_text(text):
    """Process a single job description text"""
    if not text:
        return ""
    return preprocessor.get_full_preprocessed_text(text)


def extract_skills(text):
    """Extract skills from text"""
    skill_keywords = [
        "python",
        "java",
        "javascript",
        "react",
        "angular",
        "vue",
        "node",
        "django",
        "flask",
        "spring",
        "tensorflow",
        "pytorch",
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "data science",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "git",
        "linux",
        "html",
        "css",
        "rest api",
        "graphql",
        "c++",
        "c#",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "go",
        "typescript",
        "mern",
        "full stack",
    ]

    text_lower = text.lower()
    found_skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills


def render_jd_results(results, resume_texts, resume_names, jd_label, key_suffix):
    """Render metrics, table, charts, top candidate and download button for a single JD's ranking"""

    # Extract skills (independent of which JD this is)
    skills_list = []
    for text in resume_texts:
        skills = extract_skills(text)
        skills_list.append(", ".join(skills[:5]) if skills else "No skills found")

    results = results.copy()
    results["Skills"] = skills_list

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Candidates", len(results))
    with col2:
        st.metric("🏆 Top Score", f"{results.iloc[0]['Similarity_Score']:.1f}%")
    with col3:
        st.metric("📈 Average Score", f"{results['Similarity_Score'].mean():.1f}%")
    with col4:
        st.metric("📁 Resumes", len(resume_texts))

    # Results table
    st.subheader("📋 Ranked Candidates")

    display_df = results.copy()
    display_df["Similarity_Score"] = display_df["Similarity_Score"].round(1)

    # Color coding function
    def color_score(val):
        if val >= 80:
            return "background-color: #28a745; color: white"
        elif val >= 60:
            return "background-color: #ffc107; color: black"
        else:
            return "background-color: #dc3545; color: white"

    # Apply styling
    styled_df = display_df.style.applymap(color_score, subset=["Similarity_Score"])

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Visualization
    st.subheader("📈 Score Visualization")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        top_n = min(10, len(display_df))
        colors = [
            "#2ecc71" if i < 3 else "#f1c40f" if i < 6 else "#e74c3c"
            for i in range(top_n)
        ]

        ax1.barh(
            display_df["Candidate"][:top_n][::-1],
            display_df["Similarity_Score"][:top_n][::-1],
            color=colors[::-1],
        )
        ax1.set_xlabel("Similarity Score (%)")
        ax1.set_title("Top Candidates")
        ax1.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        # Histogram
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(
            display_df["Similarity_Score"],
            bins=10,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.axvline(
            display_df["Similarity_Score"].mean(),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f'Mean: {display_df["Similarity_Score"].mean():.1f}%',
        )
        ax2.axvline(
            display_df["Similarity_Score"].median(),
            color="green",
            linestyle="dashed",
            linewidth=2,
            label=f'Median: {display_df["Similarity_Score"].median():.1f}%',
        )
        ax2.set_xlabel("Similarity Score (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Score Distribution")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Top candidate details
    st.subheader("🏆 Top Candidate")
    top = results.iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Candidate:** {top['Candidate']}")
    with col2:
        st.success(f"**Score:** {top['Similarity_Score']:.1f}%")
    with col3:
        st.info(f"**Skills:** {top.get('Skills', 'N/A')}")

    # Download button
    csv = display_df.to_csv(index=False)
    safe_label = "".join(c if c.isalnum() else "_" for c in jd_label)[:40]
    st.download_button(
        label=f"📥 Download Results CSV ({jd_label})",
        data=csv,
        file_name=f"ranked_candidates_{safe_label}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"download_{key_suffix}",
    )

    return results


def display_results(resume_texts, jd_texts, resume_names, jd_names=None):
    """Display ranking results. Supports multiple job descriptions via tabs —
    each JD gets its own ranking, computed from the same similarity matrix."""
    if not resume_texts or not jd_texts:
        st.warning("⚠️ Please provide both resumes and job descriptions")
        return

    if jd_names is None:
        jd_names = [f"Job Description {i+1}" for i in range(len(jd_texts))]

    try:
        with st.spinner("🔄 Processing and ranking candidates..."):
            # Preprocess texts (done once, regardless of how many JDs)
            cleaned_resumes = [process_resume_text(t) for t in resume_texts]
            cleaned_jds = [process_jd_text(t) for t in jd_texts]

            # Calculate similarity matrix: shape (n_resumes, n_jds)
            similarity_matrix = calculator.calculate_similarity_scores(
                cleaned_resumes, cleaned_jds
            )

            st.success("✅ Processing complete!")

            if len(jd_texts) == 1:
                # Single JD — render directly, no tabs needed
                results = ranker.rank_candidates(
                    similarity_matrix, resume_names, resume_names, jd_index=0
                )
                render_jd_results(results, resume_texts, resume_names, jd_names[0], key_suffix="0")
            else:
                # Multiple JDs — one tab per job description, each with its own ranking
                st.info(
                    f"📋 {len(jd_texts)} job descriptions provided — showing ranked "
                    f"candidates separately for each. Select a tab below."
                )
                tabs = st.tabs([f"🎯 {name}" for name in jd_names])
                for jd_index, tab in enumerate(tabs):
                    with tab:
                        results = ranker.rank_candidates(
                            similarity_matrix,
                            resume_names,
                            resume_names,
                            jd_index=jd_index,
                        )
                        render_jd_results(
                            results,
                            resume_texts,
                            resume_names,
                            jd_names[jd_index],
                            key_suffix=str(jd_index),
                        )

            return similarity_matrix

    except Exception as e:
        st.error(f"❌ Error processing: {str(e)}")
        return None


def process_files(resume_files, jd_files):
    """Process uploaded files"""
    if not resume_files or not jd_files:
        st.warning("⚠️ Please upload both resumes and job descriptions")
        return

    resume_texts = []
    resume_names = []

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Process resumes
        for i, file in enumerate(resume_files):
            status_text.text(
                f"📄 Processing resume {i+1}/{len(resume_files)}: {file.name}"
            )

            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            try:
                text = parser.extract_text(tmp_path)
                if text:
                    resume_texts.append(text)
                    resume_names.append(
                        file.name.replace(".pdf", "").replace("_", " ").title()
                    )
                else:
                    resume_texts.append("")  # Empty text
                    resume_names.append(file.name.replace(".pdf", ""))
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            progress_bar.progress((i + 1) / (len(resume_files) + len(jd_files)))

        # Process job descriptions
        jd_texts = []
        jd_names = []
        for i, file in enumerate(jd_files):
            status_text.text(f"📄 Processing JD {i+1}/{len(jd_files)}: {file.name}")

            try:
                text = file.getvalue().decode("utf-8")
                jd_texts.append(text)
                jd_names.append(file.name.replace(".txt", "").replace("_", " ").title())
            except Exception as e:
                st.warning(f"Could not read {file.name}: {str(e)}")
                jd_texts.append("")
                jd_names.append(file.name)

            progress_bar.progress(
                (len(resume_files) + i + 1) / (len(resume_files) + len(jd_files))
            )

        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)

        # Display results (one tab per JD if more than one was uploaded)
        display_results(resume_texts, jd_texts, resume_names, jd_names)

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")


def process_text(resume_text, jd_text):
    """Process text input"""
    if not resume_text or not jd_text:
        st.warning("⚠️ Please enter both resume and job description text")
        return

    display_results([resume_text], [jd_text], ["Candidate"])


def process_sample_data(resumes, jds):
    """Process sample data"""
    if not resumes or not jds:
        st.warning("⚠️ No sample data available")
        return

    resume_names = [f"Sample {i+1}" for i in range(len(resumes))]
    jd_names = [jd.strip().split("\n")[0] for jd in jds]
    display_results(resumes, jds, resume_names, jd_names)


def load_sample_data():
    """Load sample data"""
    sample_resumes = [
        """John Smith
Python Developer | 5 years experience
Skills: Python, Machine Learning, Deep Learning, NLP, SQL, TensorFlow, PyTorch
Experience: Senior Data Scientist at Google (2020-2024)""",
        """Sarah Johnson
Full Stack Developer | 4 years experience
Skills: React, Node.js, MongoDB, Express, JavaScript, TypeScript, AWS
Experience: Full Stack Developer at Microsoft (2021-2024)""",
        """Mike Chen
Data Scientist | 3 years experience
Skills: Python, Machine Learning, Data Analysis, SQL, Scikit-learn, Pandas
Experience: Data Scientist at Amazon (2021-2024)""",
        """Emily Davis
Frontend Developer | 3 years experience
Skills: React, Angular, Vue.js, JavaScript, HTML5, CSS3, TypeScript
Experience: Frontend Developer at Meta (2020-2023)""",
        """David Wilson
Backend Developer | 5 years experience
Skills: Python, Django, Flask, PostgreSQL, Docker, AWS, Redis
Experience: Senior Backend Developer at Netflix (2019-2024)""",
    ]

    sample_jds = [
        """Senior Data Scientist
Skills Required: Python, Machine Learning, Deep Learning, NLP, SQL
Experience: 3+ years in data science""",
        """Full Stack Developer
Skills Required: React, Node.js, MongoDB, Express, AWS
Experience: 3+ years in full stack development""",
        """AI Engineer
Skills Required: Python, Machine Learning, Deep Learning, NLP, TensorFlow
Experience: 2+ years in AI/ML""",
    ]

    return sample_resumes, sample_jds


# ====================================================================
# UI COMPONENTS
# ====================================================================

# Sidebar
st.sidebar.title("🤖 Resume Screening")
st.sidebar.markdown("---")

upload_option = st.sidebar.radio(
    "Select Input Method", ["📤 Upload Files", "📝 Enter Text", "📊 Use Sample Data"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** Upload PDF resumes and TXT job descriptions "
    "to get AI-powered candidate rankings."
)

st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0 | Made with ❤️")

# ====================================================================
# MAIN CONTENT
# ====================================================================

# Header
st.markdown(
    '<h1 class="main-header">📄 AI Resume Screening & Ranking</h1>',
    unsafe_allow_html=True,
)

if upload_option == "📤 Upload Files":
    st.header("📤 Upload Resumes and Job Descriptions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resumes")
        resume_files = st.file_uploader(
            "Upload Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_uploader",
        )

        if resume_files:
            st.success(f"✅ {len(resume_files)} resume(s) uploaded")
            for f in resume_files:
                st.caption(f"📎 {f.name} ({f.size/1024:.1f} KB)")

    with col2:
        st.subheader("📋 Job Descriptions")
        jd_files = st.file_uploader(
            "Upload Job Descriptions (TXT)",
            type=["txt"],
            accept_multiple_files=True,
            key="jd_uploader",
        )

        if jd_files:
            st.success(f"✅ {len(jd_files)} job description(s) uploaded")
            for f in jd_files:
                st.caption(f"📎 {f.name} ({f.size/1024:.1f} KB)")

    if st.button("🚀 Start Screening", type="primary", use_container_width=True):
        if resume_files and jd_files:
            process_files(resume_files, jd_files)
        else:
            st.warning("⚠️ Please upload both resumes and job descriptions")

elif upload_option == "📝 Enter Text":
    st.header("📝 Enter Resume and Job Description Text")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Text")
        resume_text = st.text_area(
            "Paste resume text here...",
            height=300,
            placeholder="Enter resume text...",
            key="resume_text_input",
        )

        # Example resume text
        if st.button("📋 Load Example Resume"):
            st.session_state.resume_text_input = """John Doe
Python Developer | 5 years experience
Skills: Python, Machine Learning, Deep Learning, SQL, TensorFlow
Experience: Data Scientist at Google (2020-2024)"""
            st.rerun()

    with col2:
        st.subheader("📋 Job Description Text")
        jd_text = st.text_area(
            "Paste job description here...",
            height=300,
            placeholder="Enter job description text...",
            key="jd_text_input",
        )

        # Example JD text
        if st.button("📋 Load Example JD"):
            st.session_state.jd_text_input = """Senior Data Scientist
Skills: Python, Machine Learning, Deep Learning, NLP, SQL
Experience: 3+ years"""
            st.rerun()

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if resume_text and jd_text:
            process_text(resume_text, jd_text)
        else:
            st.warning("⚠️ Please enter both resume and job description text")

else:  # "📊 Use Sample Data"
    st.header("📊 Sample Data Analysis")

    st.info(
        "📌 This will use pre-loaded sample resumes and job descriptions "
        "to demonstrate the system's capabilities."
    )

    # Show sample data
    with st.expander("📄 View Sample Resumes"):
        sample_resumes, _ = load_sample_data()
        for i, resume in enumerate(sample_resumes, 1):
            st.text_area(f"Resume {i}", resume, height=100, disabled=True)

    with st.expander("📋 View Sample Job Descriptions"):
        _, sample_jds = load_sample_data()
        for i, jd in enumerate(sample_jds, 1):
            st.text_area(f"Job Description {i}", jd, height=80, disabled=True)

    if st.button(
        "📥 Load & Process Sample Data", type="primary", use_container_width=True
    ):
        with st.spinner("Loading sample data..."):
            sample_resumes, sample_jds = load_sample_data()
            process_sample_data(sample_resumes, sample_jds)

# Footer
st.markdown("---")
st.caption(
    "🔬 AI Resume Screening System | Built with Streamlit, Scikit-learn, and NLP"
)
