from crewai import Task
from dotenv import load_dotenv
import os
import openai
from agents import blood_test_analyst, article_researcher, health_advisor
from crewai import Task

def create_tasks(text):
    analyze_blood_test_task = Task(
        description=f"""
        Analyze the provided blood test report: "{text}". Carefully review each test result to extract
        key details such as the test name, measured value, and the corresponding normal range. For each 
        result:
        - Compare the measured value against the normal range.
        - If the value is within the normal range, classify it as normal.
        - If the value deviates from the normal range, identify it as abnormal and provide an in-depth 
          explanation of the potential health implications, including possible underlying conditions or 
          causes.
        
        Your analysis should include:
        1. A detailed breakdown of all test results categorized by normal and abnormal values.
        2. Clear explanations of any abnormalities, emphasizing their significance and potential health
           risks.
        3. Practical suggestions for further investigation or consultation, such as additional tests or
           specialist referrals.
        
        Conclude the task with a comprehensive summary that synthesizes all findings into an actionable 
        and clear overview, suitable for informing further medical decisions.""",
        expected_output='A comprehensive analysis and summary of the blood test results.',
        agent=blood_test_analyst,
        async_execution=False,
    )

    find_articles_task = Task(
        description="""
        Conduct a thorough search for authoritative health articles that provide deeper insights into the 
        blood test analysis results. Use the abnormalities and findings identified in the previous task 
        as your primary search criteria. The goal is to find:
        - Articles that explain the significance of the test results.
        - Research papers or reliable health resources that discuss potential causes, conditions, or 
          treatments related to the abnormalities found.
        
        Provide a curated list of these articles, including:
        1. Article titles and concise descriptions of their content.
        2. Direct links to the articles.
        3. Any additional notes on why each article is relevant or valuable for understanding the test
           results.""",
        expected_output='A curated list of health articles with detailed descriptions and links.',
        agent=article_researcher,
        context=[analyze_blood_test_task],  # Uses the analysis as input context
        async_execution=False,
    )

    provide_recommendations_task = Task(
        description="""
        Based on the health articles gathered and the blood test analysis, formulate personalized health
        recommendations. These recommendations should be actionable and tailored to address the 
        findings from the blood test, focusing on:
        - Lifestyle changes (e.g., diet, exercise, stress management).
        - Preventative measures to mitigate potential risks highlighted by the test results.
        - Suggestions for medical follow-up, including consultations with specific healthcare 
          professionals or additional diagnostic tests.
        
        Your output should include:
        1. A clear and concise set of health recommendations, prioritized by urgency and impact.
        2. Direct references to the supporting health articles, with links included.
        3. Any additional context or insights that enhance the user’s understanding of the recommendations.""",
        expected_output='A detailed set of personalized health recommendations with links to supporting articles.',
        agent=health_advisor,
        context=[find_articles_task],  # Uses articles as input context
        async_execution=False,
    )

    return [analyze_blood_test_task, find_articles_task, provide_recommendations_task]