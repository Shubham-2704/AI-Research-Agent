import streamlit as st
from langgraph_graph import run_research_agent
import fitz  

st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
        /* General body styling */
        html, body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
            background-color: #0d1117; /* Dark background */
        }
        /* Title */
        h1 {
            color: #c9d1d9;
            text-align: center;
            padding-bottom: 1rem;
        }
        /* Subheaders */
        h2, h3 {
            color: #58a6ff; /* Blue accent for headers */
            border-bottom: 2px solid #30363d;
            padding-bottom: 0.3em;
        }
        /* Text input */
        .stTextInput > div > div > input {
            background-color: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
        }
        /* Buttons */
        .stButton > button {
            background-color: #238636;
            color: white;
            border-radius: 6px;
            border: 1px solid #2ea043;
        }
        .stButton > button:hover {
            background-color: #2ea043;
            border: 1px solid #3fb950;
        }
        /* Success message */
        .stAlert[data-baseweb="alert"] {
            background-color: rgba(35, 134, 54, 0.2);
            color: #3fb950;
        }
        /* Info message */
        .stAlert[data-baseweb="alert"][class*="st-emotion-cache-l99u6y"] {
            background-color: rgba(56, 139, 253, 0.2);
            color: #58a6ff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Generative AI Research Assistant")

# Initialize session state
if "report" not in st.session_state:
    st.session_state.report = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ---- PDF Upload Section ----
pdf = st.file_uploader("Optionally upload a PDF for additional context", type=["pdf"])

# Extract PDF Text
st.session_state.pdf_text = ""
if pdf is not None:
    with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
        for page in doc:
            st.session_state.pdf_text += page.get_text()

# ---- Research Section ----
# Text input field
topic = st.text_input(
    "Enter your research topic or question",
    placeholder="e.g., Impact of AI on healthcare",
    key=f"topic_input_{st.session_state.input_key}"
)

# Start Research button directly under the text field
submitted = st.button("Start Research", type="primary")

if submitted and topic:
    try:
        with st.spinner("Running the research agent..."):
            st.session_state.current_topic = topic
            st.session_state.report = run_research_agent(topic, pdf_text=st.session_state.pdf_text)
        # Clear input by changing the key
        st.session_state.input_key += 1
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.report = None

# ---- Display current research topic ----
if st.session_state.current_topic:
    st.info(f"Research completed: {st.session_state.current_topic}")

# ---- Results Section ----
if st.session_state.report:
    st.success("Research completed!")
    report = st.session_state.report

    summary = report.get("summary", "No summary generated.")
    detailed_report = report.get("report", "No report available.")

    st.subheader("Executive Summary")
    st.markdown(summary)

    if summary != detailed_report:
        st.subheader("Detailed Report")
        st.markdown(detailed_report)

    st.subheader("Citations")
    citations = report.get("citations", [])
    show_all = st.checkbox(" Show all citations", value=False)

    if show_all:
        for src in citations:
            st.markdown(f"- [{src}]({src})")
    else:
        for src in citations[:5]:
            st.markdown(f"- [{src}]({src})")
        if len(citations) > 5:
            st.caption(f"Showing 5 of {len(citations)} citations")

    st.subheader("Confidence Scores")
    for i, score in enumerate(report.get("confidence_scores", []), 1):
        st.markdown(f"**Summary {i}** — Confidence: {score:.1f}%")

    import streamlit as st

# Inject custom CSS for green download button
    st.markdown("""
        <style>
        div.stDownloadButton > button:first-child {
            background-color: #28a745;
            color: white;
            border: None;
            border-radius: 5px;
            padding: 0.5em 1em;
        }
        div.stDownloadButton > button:first-child:hover {
            background-color: #218838;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.download_button(
        "Download Full Report",
        data=report.get("report", ""),
        file_name="research_report.txt"
    )

    # Add a button to start new research
    # if st.button("Start New Research"):
    #     st.session_state.report = None
    #     st.session_state.current_topic = ""
    #     st.rerun()

elif not st.session_state.current_topic and not st.session_state.report:
    st.info("Enter a topic to start the research.")

# Add footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Developed by Shubham Badresiya</p>", unsafe_allow_html=True)
