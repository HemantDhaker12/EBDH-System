import os
import sys
import subprocess
import tempfile
import pandas as pd
import streamlit as st

# ============================================================================
# Page Configuration and Header
# ============================================================================
st.set_page_config(
    page_title="EDHC Candidate Ranking Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("EDHC Candidate Ranking Demo")
st.markdown("""
This application is a demonstration interface for the candidate discovery and ranking system. 
It accepts candidate datasets, runs the offline ranking pipeline under CPU constraints, 
and outputs a list-wise ranked list of the Top 100 candidates along with their justifications.
""")

# ============================================================================
# Option 1: Built-in Demonstration (Recommended)
# ============================================================================
st.markdown("## Option 1 — Run Built-in Demo (Recommended)")
st.info("Directly test the pipeline using the built-in sample candidate dataset containing candidates matching the search rubric.")

run_demo = st.button("▶ Run Demo", type="primary")

st.markdown("--- OR ---")

# ============================================================================
# Option 2: Custom Candidates Ingestion
# ============================================================================
st.markdown("## Option 2 — Upload Your Own Candidate File")
uploaded_file = st.file_uploader(
    "Upload candidate file (.json, .jsonl, .jsonl.gz)",
    type=["json", "jsonl", "jsonl.gz"],
    help="Upload candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json"
)

run_upload = False
if uploaded_file is not None:
    st.markdown(f"**Selected File:** `{uploaded_file.name}`")
    run_upload = st.button("▶ Run Ranking")

# ============================================================================
# Core Pipeline Execution and Display Logic
# ============================================================================
def execute_ranking(candidates_input_source, is_upload=False):
    """Executes the pipeline script via subprocess and manages temporary directories/files."""
    # Create a temporary directory that is automatically cleaned up on context exit
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_out_path = os.path.join(temp_dir, "submission.csv")
        
        # Prepare input path
        if is_upload:
            # Preserve file extension/suffix to ensure correct loading routing in rank.py
            name_lower = uploaded_file.name.lower()
            if name_lower.endswith(".gz"):
                suffix = ".jsonl.gz"
            elif name_lower.endswith(".json"):
                suffix = ".json"
            else:
                suffix = ".jsonl"
                
            input_path = os.path.join(temp_dir, f"uploaded_candidates{suffix}")
            # Write file buffer to temporary input file
            with open(input_path, "wb") as f:
                f.write(candidates_input_source.getbuffer())
        else:
            # Direct path to the built-in sample candidates file
            input_path = candidates_input_source
            
        # Execute the ranking CLI via subprocess using the current python executable
        with st.spinner("Running ranking pipeline..."):
            cmd = [
                sys.executable,
                "rank.py",
                "--candidates", input_path,
                "--out", temp_out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        # Process exit codes
        if result.returncode == 0:
            st.success("✓ Ranking completed successfully")
            
            if os.path.exists(temp_out_path):
                # Load generated CSV to render preview
                df = pd.read_csv(temp_out_path)
                
                st.markdown("### Top 10 Candidates")
                st.dataframe(df.head(10), width="stretch")
                
                # Expose download button for submission output
                with open(temp_out_path, "rb") as f:
                    csv_data = f.read()
                    
                st.download_button(
                    label="Download submission.csv",
                    data=csv_data,
                    file_name="submission.csv",
                    mime="text/csv"
                )
            else:
                st.error("Output file `submission.csv` was not created by the pipeline.")
        else:
            # Subprocess failed
            st.error("An error occurred during execution of the ranking script:")
            
            with st.expander("Show Stdout", expanded=False):
                st.code(result.stdout)
                
            with st.expander("Show Stderr", expanded=True):
                st.code(result.stderr)

# ============================================================================
# Action Execution Triggers
# ============================================================================
if run_demo:
    sample_path = "hackathon_assets/sample_candidates.json"
    if os.path.exists(sample_path):
        execute_ranking(sample_path, is_upload=False)
    else:
        st.error(f"Sample candidates file not found at expected path: `{sample_path}`")

elif run_upload:
    execute_ranking(uploaded_file, is_upload=True)
