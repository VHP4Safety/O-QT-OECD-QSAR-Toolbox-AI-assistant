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
        model="gpt-4.1", # Changed to user-specified model
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

# --- NEW: Chemical Context Agent Prompts ---
CHEM_CONTEXT_SYSTEM_PROMPT = """You are a specialist agent focused on identifying the primary chemical being analyzed.
Your task is to extract the key identifiers (Chemical Name, SMILES, CAS Number if available) from the provided 'basic_info' JSON data.
1. Parse the JSON data under 'Raw Basic Chemical Info (JSON Format)'.
2. Identify the most likely primary chemical name, its SMILES string, and its CAS number (if present).
3. Format the output concisely as: "Confirmed Chemical: [Name] (CAS: [CAS Number or N/A], SMILES: [SMILES String])".
4. Ignore other data or context provided. Focus solely on extracting these identifiers.
Use clear, factual language."""

CHEM_CONTEXT_USER_TEMPLATE = """
Analysis Context (Ignore for this task): {context}

Raw Basic Chemical Info (JSON Format):
```json
{data_json}
```

Please extract the primary chemical identifiers from the JSON data above."""


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

# --- NEW: Read Across Agent Prompts ---
READ_ACROSS_SYSTEM_PROMPT = """You are a specialist agent focused on Read-Across strategy based on available data.
Your task is to analyze the provided full results data (including experimental data) and the combined analyses from other specialist agents.
1. Identify key data gaps based on the available experimental data within the 'Full Results Data (JSON Format)'.
2. Based on the chemical's properties and reactivity profile (from 'Other Specialist Analyses'), suggest a suitable read-across strategy in EXTENSIVE DETAIL. This should include:
   a) Elaborate on whether an analogue approach or category approach is more appropriate and WHY
   b) Explain the specific structural, mechanistic, or metabolic features that should guide read-across selection
   c) Discuss how each relevant physicochemical property (e.g., LogKow, water solubility) informs which analogues would be suitable
   d) Detail which predictive endpoints (e.g., toxicity types, mechanisms) are most critical for the read-across
   e) Address how the data gaps identified should influence the selection strategy
   f) Explain how the user's context (e.g., Parkinson's Disease) should be considered in the read-across approach
3. AFTER providing this detailed strategy explanation, propose 3-4 specific chemical **names** (e.g., "Bisphenol A", "Atrazine", "Permethrin") that could serve as suitable analogues or category members for read-across. Justify each suggestion based on inferred structural similarity, shared relevant functional groups or profiling alerts, and likelihood of having rich experimental data available in public databases (e.g., common pesticides, industrial chemicals, pharmaceuticals). Avoid suggesting only close structural analogues if diverse, mechanistically relevant chemicals with better data coverage exist. Do NOT invent data; base suggestions on patterns seen in the provided information and general chemical knowledge.
4. Relate the read-across strategy to the user's provided analysis context.
5. Structure your analysis clearly with distinct sections for "Read-Across Strategy", "Strategy Rationale", and "Suggested Chemical Analogues".
Use clear, scientific language."""

READ_ACROSS_USER_TEMPLATE = """
Analysis Context: {context}

Full Results Data (JSON Format):
```json
{results_json}
```

Other Specialist Analyses:
--- Combined Text ---
{specialist_outputs_text}
--- End Combined Text ---

Please provide a read-across analysis, including data gaps, strategy, and 3-4 potential analogue suggestions based *only* on the provided data and analyses."""


# --- Specialist Agent Functions (Reverted to Text Analysis Output) ---

# --- NEW: Chemical Context Agent Function ---
async def analyze_chemical_context(chemical_data: Dict[str, Any], context: str) -> str:
    """Extracts key chemical identifiers using an LLM agent."""
    try:
        # Extract only the basic_info part for this agent
        basic_info = chemical_data.get('basic_info', {})
        data_json = safe_json(basic_info)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", CHEM_CONTEXT_SYSTEM_PROMPT),
            ("human", CHEM_CONTEXT_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        # Basic validation/cleanup
        if "Confirmed Chemical:" not in response:
            # Fallback if LLM fails to follow format
            name = basic_info.get('Name', 'Unknown')
            cas = basic_info.get('CAS', 'N/A')
            smiles = basic_info.get('SMILES', 'N/A')
            return f"Confirmed Chemical: {name} (CAS: {cas}, SMILES: {smiles})"
        return response.strip()
    except Exception as e:
        error_msg = f"Error in Chemical Context Agent: {str(e)}"
        print(error_msg)
        # Fallback on error
        name = chemical_data.get('basic_info', {}).get('Name', 'Unknown')
        cas = chemical_data.get('basic_info', {}).get('CAS', 'N/A')
        smiles = chemical_data.get('basic_info', {}).get('SMILES', 'N/A')
        return f"Confirmed Chemical: {name} (CAS: {cas}, SMILES: {smiles}) [Error during analysis: {e}]"


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

# --- NEW: Read Across Agent Function ---
async def analyze_read_across(results: Dict[str, Any], specialist_outputs: List[str], context: str) -> str:
    """Analyzes data gaps and suggests read-across strategy using an LLM agent."""
    try:
        results_json = safe_json(results)
        # Combine specialist outputs into a single string for context
        specialist_outputs_text = "\n\n---\n\n".join(specialist_outputs)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", READ_ACROSS_SYSTEM_PROMPT),
            ("human", READ_ACROSS_USER_TEMPLATE)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {
            "context": context,
            "results_json": results_json,
            "specialist_outputs_text": specialist_outputs_text
        }
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Read Across Agent: {str(e)}"
        print(error_msg)
        return error_msg


# --- Synthesizer Agent Function (MODIFIED) ---

async def synthesize_report(
    chemical_identifier: str,
    specialist_outputs: List[str],
    read_across_report: str, # Added read_across_report
    context: str
) -> str:
    """Synthesizes text outputs from specialist agents into a final report, including read-across."""
    try:
        # Expecting 4 outputs in specialist_outputs list (original specialists)
        if len(specialist_outputs) != 4:
             print(f"Warning: Expected 4 original specialist outputs for synthesis, received {len(specialist_outputs)}")
             # Pad only the original 4 if needed
             outputs = specialist_outputs + ["No analysis available."] * (4 - len(specialist_outputs))
        else:
             outputs = specialist_outputs

        # Ensure all inputs are strings, handling potential errors from specialists
        analyses = [out if isinstance(out, str) else "Error: Invalid analysis format received." for out in outputs]
        phys_analysis = analyses[0]
        env_analysis = analyses[1]
        prof_analysis = analyses[2]
        exp_analysis = analyses[3]
        # Ensure read_across_report is also a string
        read_across_analysis = read_across_report if isinstance(read_across_report, str) else "Error: Invalid read-across analysis format received."


        # MODIFIED Prompts to include chemical_identifier and read_across_report
        SYNTHESIZER_SYSTEM_PROMPT_MODIFIED = """You are a lead chemical analysis expert. You have received analyses from several specialist agents focusing on different aspects of a chemical (physical properties, environmental fate, profiling/reactivity, experimental data) and a read-across analysis.
The chemical being analyzed is explicitly identified as: {chemical_identifier}.
Your task is to:

1. Synthesize the findings from the FOUR core specialist analyses (Physical Properties, Environmental Fate, Profiling/Reactivity, Experimental Data) into a single, cohesive, and comprehensive report about **{chemical_identifier}**.

2. CRITICAL: Preserve the EXACT numerical values and units from each specialist report - DO NOT round or approximate values (e.g., use "166.48 °C" not "~166 °C" if that's how it appears in the specialist report). When a specialist provides a precise value like "0.37857 mg/L", use EXACTLY that value, not an approximation like "0.38 mg/L".

3. Ensure the report directly addresses the user's original analysis context: '{context}'. Explicitly connect chemical properties to the specified context (e.g., how certain properties relate to Parkinson's Disease if that's the context).

4. Structure the final report logically (e.g., Executive Summary for {chemical_identifier}, Detailed Findings by Aspect, Context-Specific Implications, Conclusion).

5. **Crucially**, add a final section titled 'Read-Across Strategy and Suggestions' containing the complete analysis provided by the Read Across Agent, preserving both the detailed strategy explanation and the specific chemical suggestions.

6. Maintain a consistent, clear, and scientific tone, always referring to the chemical as **{chemical_identifier}**.

7. When presenting a value from a specialist report, always include its significance and relevance as described by that specialist. Do not lose these contextual connections.

8. If specialist analyses conflict, note the discrepancy and mention both perspectives."""

        SYNTHESIZER_USER_TEMPLATE_MODIFIED = """
Chemical Identifier: {chemical_identifier}
User's Analysis Context: {context}

Core Specialist Agent Analyses for {chemical_identifier}:

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

Read-Across Agent Analysis:
--- Analysis from Read Across Agent ---
{read_across_analysis}
--- End of Read Across Analysis ---

Please synthesize these analyses into a single, comprehensive report about **{chemical_identifier}**, addressing the user's context '{context}', and including the read-across section at the end."""

        # Use tuple format for prompt messages
        prompt_template = ChatPromptTemplate.from_messages([
             ("system", SYNTHESIZER_SYSTEM_PROMPT_MODIFIED),
             ("human", SYNTHESIZER_USER_TEMPLATE_MODIFIED)
        ])
        # Synthesizer outputs a string
        chain = prompt_template | llm | output_parser
        input_dict = {
            "chemical_identifier": chemical_identifier, # Pass the actual identifier
            "context": context,
            "physical_properties_analysis": phys_analysis,
            "environmental_fate_analysis": env_analysis,
            "profiling_reactivity_analysis": prof_analysis,
            "experimental_data_analysis": exp_analysis,
            "read_across_analysis": read_across_analysis # Pass the read-across report
        }
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Synthesizer Agent: {str(e)}"
        print(error_msg)
        # Include identifier in error message if possible
        return f"Error synthesizing report for {chemical_identifier}: {error_msg}"
