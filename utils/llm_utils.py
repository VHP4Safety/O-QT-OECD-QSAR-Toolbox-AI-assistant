"""
Utility functions for LLM-based report generation using a multi-agent approach.
"""
import os
import asyncio
import json
from functools import lru_cache # Make sure this is imported
from functools import wraps
from typing import Dict, Any, List, Optional
import yaml # Added
from pathlib import Path # Added for robust path handling

def async_lru_cache(maxsize=128):
    """Decorator to create an LRU cache for async functions."""
    cache = {}
    order = []

    def decorator(async_func):
        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            key = str((args, sorted(kwargs.items())))

            if key in cache:
                order.remove(key)
                order.append(key)
                return cache[key]

            result = await async_func(*args, **kwargs)

            if len(cache) >= maxsize:
                oldest_key = order.pop(0)
                cache.pop(oldest_key)

            cache[key] = result
            order.append(key)
            return result
        return wrapper
    return decorator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .data_formatter import safe_json

load_dotenv()

# --- Prompt Loading ---
@lru_cache() # Cache results to avoid re-reading the file constantly
def load_prompts(prompt_file: str = "utils/prompts.yaml") -> Dict[str, Any]:
    """Loads prompts from the specified YAML file."""
    prompts = {}
    try:
        # Use Path for better cross-platform compatibility
        file_path = Path(__file__).parent / Path(prompt_file).name
        with open(file_path, 'r') as f:
            prompts = yaml.safe_load(f)
        if not prompts: # Handle empty file case
                 print(f"Warning: Prompt file '{file_path}' is empty.")
                 return {} # Return empty dict, fallback logic will be needed later
        print(f"Successfully loaded prompts from {file_path}") # Add log for confirmation
        return prompts
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {file_path}. Check the path.")
        # No fallback here, raise or return empty dict depending on desired strictness
        # Returning empty dict to allow agent functions to handle missing prompts
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML from {file_path}: {e}")
        # Returning empty dict to allow agent functions to handle errors
        return {}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading prompts: {e}")
        return {} # General fallback

# Load prompts ONCE at module level
ALL_PROMPTS = load_prompts()

# --- Helper to get specific prompt safely ---
def get_prompt(key: str, prompt_type: str = "system") -> str:
    """Safely retrieves a specific prompt, providing a default if not found."""
    try:
        # Drill down into the structure: ALL_PROMPTS -> key -> prompt_type
        prompt = ALL_PROMPTS.get(key, {}).get(prompt_type)
        if prompt is None:
            print(f"Warning: Prompt '{key}.{prompt_type}' not found in YAML. Using default.")
            # Define a generic fallback prompt
            return f"Default {prompt_type} prompt for {key}. YAML load failed or key missing."
        return prompt
    except Exception as e:
         print(f"Error retrieving prompt '{key}.{prompt_type}': {e}. Using default.")
         return f"Default {prompt_type} prompt for {key}. Error during retrieval."


# --- LLM Initialization (keep as is) ---
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

llm = get_llm()
output_parser = StrOutputParser()

# --- Specialist Agent Functions (MODIFIED) ---

# --- Chemical Context Agent ---
@async_lru_cache(maxsize=128)
async def analyze_chemical_context(chemical_data: Dict[str, Any], context: str) -> str:
    """Extracts key chemical identifiers using an LLM agent."""
    try:
        basic_info = chemical_data.get('basic_info', {}) # [cite: 282]
        data_json = safe_json(basic_info)

        # Get prompts from loaded YAML data
        system_prompt = get_prompt("chem_context", "system")
        user_template = get_prompt("chem_context", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser # [cite: 283]
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        # ... (keep existing fallback logic if needed) [cite: 284]
        return response.strip()
    except Exception as e:
        # ... (keep existing error handling) [cite: 285]
         error_msg = f"Error in Chemical Context Agent: {str(e)}"
         print(error_msg)
         name = chemical_data.get('basic_info', {}).get('Name', 'Unknown')
         cas = chemical_data.get('basic_info', {}).get('CAS', 'N/A')
         smiles = chemical_data.get('basic_info', {}).get('SMILES', 'N/A')
         return f"Confirmed Chemical: {name} (CAS: {cas}, SMILES: {smiles}) [Error during analysis: {e}]"


# --- Physical Properties Agent ---
@async_lru_cache(maxsize=128)
async def analyze_physical_properties(data: Dict[str, Any], context: str) -> str:
    """Analyzes physical properties using an LLM agent, returning text."""
    try:
        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("phys_props", "system")
        user_template = get_prompt("phys_props", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ]) # [cite: 286]
        chain = prompt_template | llm | output_parser # [cite: 287]
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Physical Properties Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Environmental Fate Agent ---
@async_lru_cache(maxsize=128)
async def analyze_environmental_fate(data: Dict[str, Any], context: str) -> str:
    """Analyzes environmental fate properties using an LLM agent, returning text.""" # [cite: 288]
    try:
        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("env_fate", "system")
        user_template = get_prompt("env_fate", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser # [cite: 289]
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Environmental Fate Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Profiling/Reactivity Agent ---
@async_lru_cache(maxsize=128)
async def analyze_profiling_reactivity(data: Dict[str, Any], context: str) -> str:
    """Analyzes profiling and reactivity using an LLM agent, returning text.""" # [cite: 290]
    try:
        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("profiling_reactivity", "system")
        user_template = get_prompt("profiling_reactivity", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser # [cite: 291]
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Profiling/Reactivity Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Experimental Data Agent ---
@async_lru_cache(maxsize=128)
async def analyze_experimental_data(data: Dict[str, Any], context: str) -> str:
    """Analyzes experimental data using an LLM agent, returning text.""" # [cite: 292]
    try:
        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("experimental_data", "system")
        user_template = get_prompt("experimental_data", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template) # [cite: 293]
        ])
        chain = prompt_template | llm | output_parser # [cite: 294]
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Experimental Data Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Read Across Agent ---
@async_lru_cache(maxsize=128)
async def analyze_read_across(results: Dict[str, Any], specialist_outputs: List[str], context: str) -> str:
    """Analyzes data gaps and suggests read-across strategy using an LLM agent.""" # [cite: 295]
    try:
        results_json = safe_json(results)
        specialist_outputs_text = "\n\n---\n\n".join(specialist_outputs)

        # Get prompts from loaded YAML data
        system_prompt = get_prompt("read_across", "system")
        user_template = get_prompt("read_across", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template) # [cite: 296]
        ])
        chain = prompt_template | llm | output_parser # [cite: 297]
        input_dict = {
            "context": context,
            "results_json": results_json,
            "specialist_outputs_text": specialist_outputs_text
        }
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Read Across Agent: {str(e)}"
        print(error_msg) # [cite: 298]
        return error_msg # Return error message string

# --- Synthesizer Agent ---
@async_lru_cache(maxsize=128)
async def synthesize_report(
    chemical_identifier: str,
    specialist_outputs: List[str],
    read_across_report: str,
    context: str
) -> str:
    """Synthesizes text outputs from specialist agents into a final report, including read-across."""
    try:
        # ... (keep existing logic for handling specialist_outputs list size) [cite: 299, 300, 301]
        if len(specialist_outputs) != 4:
             print(f"Warning: Expected 4 original specialist outputs for synthesis, received {len(specialist_outputs)}")
             outputs = specialist_outputs + ["No analysis available."] * (4 - len(specialist_outputs))
        else:
             outputs = specialist_outputs

        analyses = [out if isinstance(out, str) else "Error: Invalid analysis format received." for out in outputs]
        phys_analysis = analyses[0]
        env_analysis = analyses[1]
        prof_analysis = analyses[2]
        exp_analysis = analyses[3]
        read_across_analysis = read_across_report if isinstance(read_across_report, str) else "Error: Invalid read-across analysis format received."

        # Get prompts from loaded YAML data
        system_prompt = get_prompt("synthesizer", "system")
        user_template = get_prompt("synthesizer", "user")

        # Check if prompts were loaded successfully before creating template
        if "Default" in system_prompt or "Default" in user_template:
             # Handle the case where prompts failed to load
             print("ERROR: Synthesizer prompts failed to load from YAML. Cannot proceed.")
             return f"Error synthesizing report for {chemical_identifier}: Could not load prompts."


        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser # [cite: 312]
        input_dict = {
            "chemical_identifier": chemical_identifier,
            "context": context,
            "physical_properties_analysis": phys_analysis,
            "environmental_fate_analysis": env_analysis,
            "profiling_reactivity_analysis": prof_analysis,
            "experimental_data_analysis": exp_analysis,
            "read_across_analysis": read_across_analysis # [cite: 313]
        }
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Synthesizer Agent: {str(e)}"
        print(error_msg)
        return f"Error synthesizing report for {chemical_identifier}: {error_msg}" # [cite: 314]

# Rest of the file...
