from crewai import Agent
from tools import web_search_tool
from dotenv import load_dotenv
import os

# Load environment settings from a .env file
load_dotenv()

# Get the OpenAI API key from the environment file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OpenAI API key is missing! Please set it up in the .env file.")
    exit()

# Function to connect to OpenAI
# This sets up how the AI will behave (how creative or accurate it will be)
def setup_ai(temperature=0.7, model="gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name=model,
        temperature=temperature
    )

# Create the AI connection with default settings
ai_connection = setup_ai()

# Agent for explaining blood test reports
blood_test_analyst = Agent(
    role='Blood Test Helper',
    goal="Explain blood test results in simple language and flag anything unusual.",
    backstory=(
        "This AI acts like a friendly and knowledgeable medical assistant specialized in blood tests. "
        "It understands common medical terms and can explain them in a way that's easy for anyone to grasp. "
        "The AI focuses on making sure you feel confident and informed about your blood test results, "
        "just like having a conversation with an experienced healthcare professional."
    ),
    verbose=True,
    allow_delegation=False,
    llm=ai_connection,
    methods={
        "analyze_report": "Look at the blood test details, compare with normal ranges, and explain what the numbers mean. Flag anything unusual and what it could indicate."
    },
    expected_output="A clear, simple summary of your blood test results with explanations for anything unusual."
)

# Agent for finding medical research
article_researcher = Agent(
    role='Medical Research Finder',
    goal="Find and summarize medical research related to the blood test findings.",
    backstory=(
        "This AI is like a dedicated research assistant with access to the latest medical studies. "
        "It specializes in finding trustworthy information and breaking it down into easy-to-understand language. "
        "Think of it as your personal guide to the world of medical research, helping you make sense of complex studies."
    ),
    tools=[web_search_tool],
    verbose=True,
    allow_delegation=False,
    llm=ai_connection,
    methods={
        "conduct_research": "Search online for reliable medical studies related to the blood test results and explain what they say in a simple way."
    },
    expected_output="A short list of relevant medical studies and easy-to-understand summaries of their findings."
)

# Agent for health advice
health_advisor = Agent(
    role='Health Advisor',
    goal="Give practical health tips based on the blood test and research findings.",
    backstory=(
        "This AI acts like a holistic health coach who combines modern medical knowledge with everyday advice. "
        "It offers practical tips tailored to your health needs, focusing on areas like diet, exercise, and lifestyle. "
        "The AI ensures that its suggestions are both evidence-based and easy to follow, making health improvement accessible to everyone."
    ),
    verbose=True,
    allow_delegation=False,
    llm=ai_connection,
    methods={
        "provide_recommendations": "Look at the blood test results and research to suggest easy-to-follow health tips like diet changes or exercises."
    },
    expected_output="Simple, actionable health advice that fits your situation and explains why it's helpful."
)

# How to use this setup:
# For blood test analysis: call blood_test_analyst.analyze_report()
# For finding related research: call article_researcher.conduct_research()
# For health tips: call health_advisor.provide_recommendations()
