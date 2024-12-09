import streamlit as stream_app
import openai
from PIL import Image
import PyPDF2
import tempfile
import os
from dotenv import load_dotenv
import time
from openai import OpenAI
from crewai import Crew, Process
from agents import blood_test_analyst, article_researcher, health_advisor
from tasks import create_tasks

# Load environment variables
load_dotenv()

# Retrieve API key for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    stream_app.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    stream_app.stop()

openai.api_key = OPENAI_API_KEY

# Constants for retry mechanism
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

def analyze_medical_report(content, content_format):
    """
    Generate an AI-powered analysis report for a given medical input.

    Args:
        content (str): The content to be analyzed (text or image description).
        content_format (str): Format of the content, either 'image' or 'text'.

    Returns:
        str: The generated analysis report or a generic fallback message.
    """
    analysis_prompt = (
        "Analyze the provided medical report comprehensively. Provide key observations, diagnoses, and actionable recommendations:"
    )

    api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            
            # Get the analysis result from the agent
            assigned_tasks = create_tasks(content)

            # Initialize the crew with the required agents and process configuration
            crew_setup = Crew(
                agents=[blood_test_analyst, article_researcher, health_advisor],
                tasks=assigned_tasks,
                process=Process.sequential
            )
            analysis_result = crew_setup.kickoff(inputs={'blood_test_data': content})
            return analysis_result.raw
        except Exception as err:
            print(f"Attempt {attempt + 1}: Error encountered - {err}")
            time.sleep(RETRY_DELAY)

    return "Unable to generate detailed analysis. Please consult a professional."

def extract_text_from_uploaded_pdf(pdf_file):
    """
    Extract textual content from an uploaded PDF file.

    Args:
        pdf_file (file): The uploaded PDF file.

    Returns:
        str: Extracted textual data.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_content = "".join([page.extract_text() for page in pdf_reader.pages])
    return extracted_content

def application():
    """
    Core function to initialize and manage the Streamlit application.
    """
    stream_app.set_page_config(page_title="Medical Insights Generator",layout="centered")

    stream_app.title("Medical Insights Generator")
    stream_app.markdown("Upload a medical report (PDF) for AI-assisted analysis and recommendations.")
    stream_app.markdown("**Disclaimer:** This tool is for informational purposes only and should not be substituted for professional medical advice. Always consult a licensed medical professional for accurate diagnosis and treatment.")

    # Sidebar settings
    stream_app.sidebar.header("User Settings")
    stream_app.sidebar.info("Please upload your report")

    # File upload handler
    uploaded_pdf = stream_app.file_uploader("Select a medical report (PDF format only):", type=["pdf"])

    if uploaded_pdf is not None:
        stream_app.success("PDF successfully uploaded.")

        if stream_app.button("Analyze Report"):
            with stream_app.spinner("Processing your report. Please wait..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                    temp_pdf.write(uploaded_pdf.getvalue())
                    pdf_path = temp_pdf.name

                with open(pdf_path, 'rb') as pdf_file:
                    document_text = extract_text_from_uploaded_pdf(pdf_file)

                analysis_result = analyze_medical_report(document_text, "text")

                stream_app.subheader("Analysis Outcome:")
                stream_app.write(analysis_result)

                os.unlink(pdf_path)

    # Footer section
    stream_app.sidebar.markdown("---")
    stream_app.sidebar.caption("Built by Aakanksha Reddy.")

if __name__ == "__main__":
    application()
