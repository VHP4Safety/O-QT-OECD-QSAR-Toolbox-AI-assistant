import json
from typing import Dict, Any
import openai
import streamlit as st

# System prompts for report generation
OVERVIEW_SYSTEM_PROMPT = """You are an expert in chemical analysis and hazard assessment from ECHA. Create a detailed report focusing on the chemical's basic information and properties.
Explain the significance of each property in relation to the user's context/concern. Use clear, scientific language and include property values from the data."""

PROFILING_SYSTEM_PROMPT = """You are a chemical analysis expert specializing in QSAR Toolbox analysis. Create a detailed report analyzing the chemical's profiling results.
Focus on identifying important structural features, chemical categories, and potential hazards. Use clear, scientific language."""

def generate_report(system_prompt: str, user_prompt: str) -> str:
    """Generate report using OpenAI API"""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return "Error generating report. Please try again."

def generate_overview_report(results: Dict[str, Any], context: str) -> str:
    """Generate overview report focusing on chemical properties"""
    data = {
        'basic_info': results['chemical_data']['basic_info'],
        'properties': results['chemical_data']['properties']
    }
    
    user_prompt = f"""
Context: {context}

Chemical Data:
{json.dumps(data, indent=2)}

Please provide a detailed analysis including:
1. Executive Summary
2. Basic Chemical Information Analysis
3. Key Properties and their Significance
4. Context-Specific Property Analysis ({context})
5. Conclusions
"""
    
    return generate_report(OVERVIEW_SYSTEM_PROMPT, user_prompt)

def generate_profiling_report(results: Dict[str, Any], context: str) -> str:
    """Generate profiling report with hazard assessment"""
    user_prompt = f"""
Context: {context}

Chemical Information:
{json.dumps(results['chemical_data']['basic_info'], indent=2)}

Profiling Data:
{json.dumps(results['profiling'], indent=2)}

Please provide a detailed analysis including:
1. Executive Summary
2. Key Profiling Results
   - Structural features and alerts
   - Chemical categories and classifications
   - Mechanistic insights
3. Hazard Assessment
   - Identified hazards and concerns
   - Relevance to the context
4. Conclusions
"""
    
    return generate_report(PROFILING_SYSTEM_PROMPT, user_prompt)
