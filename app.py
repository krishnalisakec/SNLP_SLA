
import streamlit as st
import json
import os
import pandas as pd

from healthcare_nlp.pipeline import HealthcareNLPPipeline


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Healthcare NLP System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Intelligent Healthcare Document Understanding")
st.subheader("Clinical Information Extraction System")

st.caption(
    "Upload a clinical document or paste clinical text "
    "to extract important medical information."
)


# -----------------------------------
# Load NLP Pipeline
# -----------------------------------

@st.cache_resource
def load_pipeline():
    return HealthcareNLPPipeline()


pipeline = load_pipeline()


# -----------------------------------
# Input Clinical Document
# -----------------------------------

st.header("1. Input Clinical Document")

uploaded_file = st.file_uploader(
    "Upload Clinical Document (.txt)",
    type=["txt"]
)

clinical_text = ""

if uploaded_file is not None:
    clinical_text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

clinical_text = st.text_area(
    "Enter or edit Clinical Text",
    value=clinical_text,
    height=150,
    placeholder="Paste your clinical patient report here..."
)


# -----------------------------------
# Analyze Document
# -----------------------------------

if st.button("Analyze Document", type="primary"):

    if not clinical_text.strip():
        st.warning("Please enter or upload a clinical document.")

    else:

        try:
            with st.spinner("Processing clinical document..."):
                result = pipeline.process_document(clinical_text)

            # Save complete JSON output
            os.makedirs("outputs", exist_ok=True)

            output_path = os.path.join(
                "outputs",
                "clinical_output.json"
            )

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            st.success("Clinical document analyzed successfully!")

            # -----------------------------------
            # 2. Preprocessing Results
            # -----------------------------------

            st.header("2. Preprocessing Results")

            preprocessing = result.get("preprocessing", {})

            cleaned_text = preprocessing.get("cleaned_text", "")
            expanded_text = preprocessing.get("expanded_text", "")

            with st.expander("View Cleaned Text"):
                st.write(
                    cleaned_text[:300] +
                    ("..." if len(cleaned_text) > 300 else "")
                )

                st.caption(
                    f"Total characters: {len(cleaned_text)}"
                )

                st.download_button(
                    "Download Full Cleaned Text",
                    cleaned_text,
                    file_name="cleaned_text.txt",
                    mime="text/plain",
                    key="cleaned_download"
                )

            with st.expander("View Expanded Text"):
                st.write(
                    expanded_text[:300] +
                    ("..." if len(expanded_text) > 300 else "")
                )

                st.caption(
                    f"Total characters: {len(expanded_text)}"
                )

                st.download_button(
                    "Download Full Expanded Text",
                    expanded_text,
                    file_name="expanded_text.txt",
                    mime="text/plain",
                    key="expanded_download"
                )

            # -----------------------------------
            # Extract Result Data
            # -----------------------------------

            entities = result.get("entities", [])
            assertions = result.get("assertions", [])
            relations = result.get("relations", [])
            salience = result.get("salience", [])

            # Handle clinical summary dictionary
            summary_data = result.get("clinical_summary", "")

            if isinstance(summary_data, dict):
                summary_text = summary_data.get("summary", "")
                sentence_scores = summary_data.get(
                    "sentence_scores", []
                )
            else:
                summary_text = summary_data
                sentence_scores = []

            # -----------------------------------
            # 3. Result Overview
            # -----------------------------------

            st.header("3. Result Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Entities", len(entities))
            col2.metric("Assertions", len(assertions))
            col3.metric("Relations", len(relations))
            col4.metric("Salience Records", len(salience))

            # -----------------------------------
            # 4. Clinical Summary
            # -----------------------------------

            st.header("4. Clinical Summary")

            if summary_text:
                st.info(summary_text)
            else:
                st.warning("No clinical summary generated.")

            # -----------------------------------
            # 5. Detailed Extraction Results
            # -----------------------------------

            st.header("5. Detailed Extraction Results")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Named Entities",
                "Assertions",
                "Relations",
                "Salience Scores"
            ])

            with tab1:
                st.subheader("Clinical Named Entities")

                if entities:
                    st.dataframe(
                        pd.DataFrame(entities),
                        use_container_width=True,
                        height=250
                    )
                else:
                    st.info("No entities detected.")

            with tab2:
                st.subheader("Assertion Detection")

                if assertions:
                    st.dataframe(
                        pd.DataFrame(assertions),
                        use_container_width=True,
                        height=250
                    )
                else:
                    st.info("No assertions detected.")

            with tab3:
                st.subheader("Relation Extraction")

                if relations:
                    st.dataframe(
                        pd.DataFrame(relations),
                        use_container_width=True,
                        height=250
                    )
                else:
                    st.info("No relations detected.")

            with tab4:
                st.subheader("Clinical Salience Scores")

                if salience:
                    st.dataframe(
                        pd.DataFrame(salience),
                        use_container_width=True,
                        height=250
                    )
                else:
                    st.info("No salience scores available.")

            # -----------------------------------
            # 6. Sentence Scores
            # -----------------------------------

            if sentence_scores:
                with st.expander("View Sentence Salience Scores"):

                    st.dataframe(
                        pd.DataFrame(sentence_scores),
                        use_container_width=True,
                        height=250
                    )

            # -----------------------------------
            # 7. Download Complete Results
            # -----------------------------------

            st.header("6. Download Complete Results")

            json_data = json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )

            st.download_button(
                label="Download Complete JSON Output",
                data=json_data,
                file_name="clinical_output.json",
                mime="application/json"
            )

            st.caption(
                "Complete results are also saved in "
                "outputs/clinical_output.json"
            )

        except Exception as e:
            st.error(f"Error during processing: {e}")
