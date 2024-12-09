from crewai_tools import WebsiteSearchTool
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration for OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

openai.api_key = OPENAI_API_KEY

web_search_tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="openai",
            config=dict(
                model="gpt-4",  
                temperature=0.7,  
                api_key=openai.api_key,  
            ),
        ),
        embedder=dict(
            provider="openai",
            config=dict(
                model="text-embedding-ada-002",  
                api_key=openai.api_key, 
            ),
        ),
    )
)