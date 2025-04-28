"""
Utility functions for LLM-based report generation using a multi-agent approach.
"""
import os
import asyncio
import json
from typing import Dict, Any, List, Optional # Removed Union, Pydantic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Removed SystemMessage, HumanMessage imports as we'll use tuples
from langchain_core.output_parsers import StrOutputParser # Re-added for synthesizer
# Import the safe serializer
from .data_formatter import safe_json

# Load environment variables
load_dotenv()


# --- LLM Initialization ---

def get_llm():
    """Initializes and returns the LangChain ChatOpenAI instance."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    # Using langchain's integration for async capabilities
    return ChatOpenAI(
        model="gpt-4.1-nano", # Changed to user-specified model
        temperature=0.7,
        openai_api_key=api_key,
        max_retries=3,
        request_timeout=60.0
    )

# Initialize LLM instance globally for reuse
llm = get_llm()
# Output parser for synthesizer (still needed)
output_parser = StrOutputParser()


# --- Specialist Agent Prompts (Adjusted for JSON Input) ---

PHYS_PROPS_SYSTEM_PROMPT = """You are a specialist agent focused on analyzing chemical physical properties.
Your task is to analyze the provided raw chemical properties data, which is formatted as a JSON string in the user message under 'Raw Chemical Properties Data (JSON Format)'.
1. Parse this JSON string to extract the relevant physical properties (e.g., Melting Point, Boiling Point, LogP, Solubility, Vapor Pressure).
2. Explain the significance of key properties found in the data.
3. Relate these properties specifically to the user's provided analysis context.
4. Include specific values and units from the JSON data where relevant. If a value is missing, state that clearly.
5. Structure your analysis clearly. Ignore other data types like experimental or profiling data.
Use clear, scientific language."""

PHYS_PROPS_USER_TEMPLATE = """
Analysis Context: {context}

Raw Chemical Properties Data (JSON Format):
```json
{data_json}
```

Please provide a focused textual analysis of the physical properties found in the JSON data above and their relevance to the context."""

ENV_FATE_SYSTEM_PROMPT = """You are a specialist agent focused on analyzing chemical environmental fate properties.
Your task is to analyze the provided raw chemical properties data, which is formatted as a JSON string in the user message under 'Raw Chemical Properties Data (JSON Format)', focusing ONLY on environmental fate properties (e.g., BCF, BAF, Biodegradation, Half-Life, Water Solubility, Vapor Pressure, log Kow).
1. Parse this JSON string and identify key environmental fate properties.
2. Explain the significance of each property regarding the chemical's persistence, bioaccumulation, and mobility in the environment.
3. Relate these properties specifically to the user's provided analysis context (e.g., environmental risk, bioaccumulation potential).
4. Try to include specific parameter values and units found in the data (e.g., "BCF: 2.5 log(L/kg)"). If a value is missing, state that clearly rather than using placeholders like {value}.
5. Structure your analysis clearly. Ignore other data types like experimental or profiling data.
Use clear, scientific language."""

ENV_FATE_USER_TEMPLATE = """
Analysis Context: {context}

Raw Chemical Properties Data (JSON Format):
```json
{data_json}
```

Please provide a focused textual analysis of the environmental fate properties found in the JSON data above and their relevance to the context."""

PROFILING_REACTIVITY_SYSTEM_PROMPT = """You are a specialist agent focused on analyzing chemical profiling results and reactivity.
Your task is to analyze the provided raw chemical profiling data, which is formatted as a JSON string in the user message under 'Raw Chemical Profiling Data (JSON Format)', focusing ONLY on the profiling results (e.g., DNA binding, protein binding, receptor binding, functional groups, toxicity classifications).
1. Parse this JSON string and interpret the results from chemical profilers found.
2. Explain what each significant profiling result suggests about the chemical's potential mechanism of action, reactivity, or toxicological endpoints.
3. Combine insights across different profilers if possible.
4. Relate these findings specifically to the user's provided analysis context.
5. Structure your analysis clearly. Ignore other data types like physical properties or experimental data.
Use clear, scientific language."""

PROFILING_REACTIVITY_USER_TEMPLATE = """
Analysis Context: {context}

Raw Chemical Profiling Data (JSON Format):
```json
{data_json}
```

Please provide a focused textual analysis of the profiling results found in the JSON data above, interpreting their significance regarding reactivity, mechanisms, and relevance to the context."""

EXPERIMENTAL_DATA_SYSTEM_PROMPT = """You are a specialist agent focused on analyzing experimental chemical data.
Your task is to analyze the provided raw experimental data, which is formatted as a JSON string in the user message under 'Raw Experimental Data (JSON Format)', focusing ONLY on the experimental data points.
1. Parse this JSON string and summarize the key findings from the experimental data points found.
2. Identify any notable trends or significant values.
3. Relate these experimental findings specifically to the user's provided analysis context.
4. Structure your analysis clearly. Ignore other data types like physical properties or profiling data.
Use clear, scientific language."""

EXPERIMENTAL_DATA_USER_TEMPLATE = """
Analysis Context: {context}

Raw Experimental Data (JSON Format):
```json
{data_json}
```

Please provide a focused textual analysis of the experimental data found in the JSON data above and its relevance to the context."""


# --- Synthesizer Agent Prompts (Original - Expecting Text Analysis Strings) ---
# Keeping old prompts for now, but they expect strings, not objects.
# This will be addressed in the next step (modifying the synthesizer).

SYNTHESIZER_SYSTEM_PROMPT = """You are a lead chemical analysis expert. You have received STRUCTURED DATA outputs from several specialist agents focusing on different aspects of a chemical (physical properties, environmental fate, profiling/reactivity, experimental data).
Your task is to:
1. Interpret the structured data provided by each specialist agent.
2. Synthesize these findings into a single, cohesive, and comprehensive textual report.
3. Ensure the report directly addresses the user's original analysis context.
4. Structure the final report logically (e.g., Executive Summary, Detailed Findings by Aspect, Context-Specific Implications, Conclusion).
5. Maintain a consistent, clear, and scientific tone.
6. Avoid simply listing the data; interpret and integrate the findings meaningfully.
7. Highlight the most critical information relevant to the user's context.
8. If data from different specialists seems contradictory, note the discrepancy."""

# This template needs significant change to accept structured data (e.g., JSON strings)
SYNTHESIZER_USER_TEMPLATE = """
User's Analysis Context: {context}

Structured Data from Specialist Agents:

--- Physical Properties Data (JSON) ---
{physical_properties_analysis}
--- End of Physical Properties Data ---

--- Environmental Fate Data (JSON) ---
{environmental_fate_analysis}
--- End of Environmental Fate Data ---

--- Profiling/Reactivity Data (JSON) ---
{profiling_reactivity_analysis}
--- End of Profiling/Reactivity Data ---

--- Experimental Data Summary (JSON) ---
{experimental_data_analysis}
--- End of Experimental Data Summary ---

Please synthesize these structured data outputs into a single, comprehensive textual report addressing the user's context: '{context}'."""


# --- Specialist Agent Functions (Reverted to Text Analysis Output) ---

async def analyze_physical_properties(data: Dict[str, Any], context: str) -> str:
    """Analyzes physical properties using an LLM agent, returning text."""
    try:
        # Use safe_json for serialization
        data_json = safe_json(data)
        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", PHYS_PROPS_SYSTEM_PROMPT),
            ("human", PHYS_PROPS_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        # Return error message string
        error_msg = f"Error in Physical Properties Agent: {str(e)}"
        print(error_msg)
        return error_msg

async def analyze_environmental_fate(data: Dict[str, Any], context: str) -> str:
    """Analyzes environmental fate properties using an LLM agent, returning text."""
    try:
        # Use safe_json for serialization
        data_json = safe_json(data)
        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", ENV_FATE_SYSTEM_PROMPT),
            ("human", ENV_FATE_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        # Return error message string
        error_msg = f"Error in Environmental Fate Agent: {str(e)}"
        print(error_msg)
        return error_msg

async def analyze_profiling_reactivity(data: Dict[str, Any], context: str) -> str:
    """Analyzes profiling and reactivity using an LLM agent, returning text."""
    try:
        # Use safe_json for serialization
        data_json = safe_json(data)
        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", PROFILING_REACTIVITY_SYSTEM_PROMPT),
            ("human", PROFILING_REACTIVITY_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        # Return error message string
        error_msg = f"Error in Profiling/Reactivity Agent: {str(e)}"
        print(error_msg)
        return error_msg

async def analyze_experimental_data(data: Dict[str, Any], context: str) -> str:
    """Analyzes experimental data using an LLM agent, returning text."""
    try:
        # Use safe_json for serialization
        # Input 'data' is expected to be {"experimental_results": [list_of_dicts]}
        data_json = safe_json(data)
        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", EXPERIMENTAL_DATA_SYSTEM_PROMPT),
            ("human", EXPERIMENTAL_DATA_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        # Return error message string
        error_msg = f"Error in Experimental Data Agent: {str(e)}"
        print(error_msg)
        return error_msg


# --- Synthesizer Agent Function (Original - Expecting Text Inputs) ---

async def synthesize_report(specialist_outputs: List[str], context: str) -> str:
    """Synthesizes text outputs from specialist agents into a final report."""
    try:
        # Ensure we have 4 string outputs, padding with default message if necessary
        if len(specialist_outputs) != 4:
             print(f"Warning: Expected 4 specialist outputs, received {len(specialist_outputs)}")
             outputs = specialist_outputs + ["No analysis available."] * (4 - len(specialist_outputs))
        else:
             outputs = specialist_outputs

        # Ensure all inputs are strings, handling potential errors from specialists
        analyses = [out if isinstance(out, str) else "Error: Invalid analysis format received." for out in outputs]
        phys_analysis = analyses[0]
        env_analysis = analyses[1]
        prof_analysis = analyses[2]
        exp_analysis = analyses[3]

        # Use the original prompts that expect text analysis strings
        SYNTHESIZER_SYSTEM_PROMPT_ORIGINAL = """You are a lead chemical analysis expert. You have received analyses from several specialist agents focusing on different aspects of a chemical (physical properties, environmental fate, profiling/reactivity, experimental data).
Your task is to:
1. Synthesize the key findings from ALL specialist analyses into a single, cohesive, and comprehensive report.
2. Ensure the report directly addresses the user's original analysis context.
3. Structure the final report logically (e.g., Executive Summary, Detailed Findings by Aspect, Context-Specific Implications, Conclusion).
4. Maintain a consistent, clear, and scientific tone.
5. Avoid simple concatenation; integrate the findings meaningfully.
6. Highlight the most critical information relevant to the user's context.
7. If specialist analyses conflict, note the discrepancy."""

        SYNTHESIZER_USER_TEMPLATE_ORIGINAL = """
User's Analysis Context: {context}

Specialist Agent Analyses:

--- Analysis from Physical Properties Agent ---
{physical_properties_analysis}
--- End of Physical Properties Analysis ---

--- Analysis from Environmental Fate Agent ---
{environmental_fate_analysis}
--- End of Environmental Fate Analysis ---

--- Analysis from Profiling/Reactivity Agent ---
{profiling_reactivity_analysis}
--- End of Profiling/Reactivity Analysis ---

--- Analysis from Experimental Data Agent ---
{experimental_data_analysis}
--- End of Experimental Data Analysis ---

Please synthesize these analyses into a single, comprehensive report addressing the user's context: '{context}'."""

        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
             ("system", SYNTHESIZER_SYSTEM_PROMPT_ORIGINAL),
             ("human", SYNTHESIZER_USER_TEMPLATE_ORIGINAL)
        ])
        # Synthesizer outputs a string
        chain = prompt_template | llm | output_parser
        input_dict = {
            "context": context,
            "physical_properties_analysis": phys_analysis,
            "environmental_fate_analysis": env_analysis,
            "profiling_reactivity_analysis": prof_analysis,
            "experimental_data_analysis": exp_analysis
        }
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Synthesizer Agent: {str(e)}"
        print(error_msg)
        return error_msg
