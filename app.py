import streamlit as st
import pandas as pd
import tempfile
import os
import io
import logging
from lease_doc_analyzer.pdf_parser import parse_lease_pdf
from lease_doc_analyzer.qa_system import LeaseQuerySystem
from lease_doc_analyzer.normalizer import normalize_date, normalize_currency
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.set_page_config(layout="wide", page_title="Lease Analyzer")

    st.title("Lease Document Analyzer")

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'excel_output' not in st.session_state:
        st.session_state.excel_output = None
    if 'extraction_stats' not in st.session_state:
        st.session_state.extraction_stats = Counter()
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Configuration")
        llm_provider = st.selectbox("Choose LLM Provider", ["gemini", "ollama", "openai"])
        
        api_key = None
        if llm_provider == "gemini":
            api_key = st.text_input("Enter Gemini API Key", type="password")
        elif llm_provider == "openai":
            api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        st.header("Upload Files")
        uploaded_file = st.file_uploader("Choose a Lease PDF", type="pdf")
        template_file = st.file_uploader("Choose an Excel template", type=["xlsx"])

    if uploaded_file and template_file:
        df_template = pd.read_excel(template_file)
        # We must read the field names from the first column of the sheet.
        # Convert all items to string and filter out any empty/NaN rows.
        queries = [str(field) for field in df_template.iloc[:, 0].dropna().tolist()]
        logging.info(f"Read {len(queries)} fields from the first column of the Excel template: {queries}")
        st.info(f"Fields to be extracted: {', '.join(queries)}")


        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_pdf_path = tmp.name

        try:
            if st.button("Analyze Lease"):
                    if (llm_provider == "gemini" or llm_provider == "openai") and not api_key:
                        st.error(f"Please enter your {llm_provider.capitalize()} API Key to proceed.")
                    else:
                        with st.spinner("Analyzing document... This may take a few minutes."):
                            lease_chunks = parse_lease_pdf(temp_pdf_path)
                            qa_system = LeaseQuerySystem(
                                lease_chunks, 
                                llm_provider_name=llm_provider, 
                                api_key=api_key
                            )
                            st.session_state.qa_system = qa_system 
                            # 'queries' is now correctly defined above
                            results = qa_system.run_queries(queries)
                    
                    processed_data = []
                    for field, response in results.items():
                        # Default to the raw answer
                        normalized_value = response.answer

                        # Only attempt to normalize if the answer is not a "not found" type
                        if response.answer and "not found" not in response.answer.lower() and "not explicitly stated" not in response.answer.lower():
                            if "date" in field.lower():
                                normalized_value = normalize_date(response.answer)
                            elif "rent" in field.lower() or "deposit" in field.lower():
                                normalized_value = normalize_currency(response.answer)
                        
                        processed_data.append({
                            "Field": field,
                            "Extracted Value": normalized_value,
                            "Source Page": response.page_number,
                            "Source Section": response.section_number,
                            "Supporting Text": response.source_chunk_text
                        })
                    
                    df_results = pd.DataFrame(processed_data)
                    
                    output_filename = "lease_analysis_results.xlsx"
                    # Save the dataframe to an Excel file
                    df_results.to_excel(output_filename, index=False, sheet_name='Lease Analysis')

                    # Read the file back into bytes for the download button
                    with open(output_filename, "rb") as f:
                        st.session_state.excel_output = f.read()
                    
                    st.session_state.analysis_complete = True
                    st.session_state.extraction_stats = qa_system.get_stats()
        finally:
            os.unlink(temp_pdf_path)

    if st.session_state.analysis_complete:
        st.success("Analysis complete!")
        st.download_button(
            label="Download Analysis as Excel",
            data=st.session_state.excel_output,
            file_name="lease_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.subheader("Extraction Audit")
        stats = st.session_state.extraction_stats
        total_extractions = sum(stats.values())
        
        if total_extractions > 0:
            st.metric(label="Deterministic Success", value=f"{stats.get('deterministic_success', 0)} ({stats.get('deterministic_success', 0) / total_extractions:.1%})")
            st.metric(label="RAG Fallback", value=f"{stats.get('rag_fallback', 0)} ({stats.get('rag_fallback', 0) / total_extractions:.1%})")
            st.metric(label="Direct RAG", value=f"{stats.get('rag_direct', 0)} ({stats.get('rag_direct', 0) / total_extractions:.1%})")
            st.metric(label="Not Found", value=f"{stats.get('not_found', 0)} ({stats.get('not_found', 0) / total_extractions:.1%})")
        else:
            st.write("No extraction statistics to display.")

    # --- Chat Interface ---
    if st.session_state.qa_system:
        st.subheader("Ask Questions About the Lease")

        # Display chat history
        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.markdown(message)

        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message to history
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get answer from QA system
            with st.spinner("Finding an answer..."):
                response = st.session_state.qa_system._run_single_rag_query(prompt)
                
                # Format the answer with source information
                answer = f"{response.answer}\n\n*Source: Page {response.page_number}, Section {response.section_number}*"
                
                # Add assistant message to history
                st.session_state.chat_history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)


if __name__ == "__main__":
    main()