# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

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

# NOTE: We intentionally DO NOT import streamlit here. Configuration must be passed explicitly.

# UPDATED: More robust handling of unhashable arguments for caching
def async_lru_cache(maxsize=128):
    """Decorator to create an LRU cache for async functions."""
    cache = {}
    order = []

    def decorator(async_func):
        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            # Create a hashable key from arguments
            try:
                # Improved handling for complex types like lists of dicts
                hashable_args = []
                for arg in args:
                    if isinstance(arg, list):
                        # Try converting list elements (if dicts) to frozensets, otherwise serialize
                        try:
                            hashable_list = tuple(frozenset(item.items()) if isinstance(item, dict) else item for item in arg)
                            hashable_args.append(hashable_list)
                        except TypeError:
                            # Fallback if list contains complex unhashable types
                            hashable_args.append(json.dumps(arg, sort_keys=True, default=str))
                    elif isinstance(arg, dict):
                        # Handle nested dictionaries by serializing unhashable values
                        try:
                            hashable_items = []
                            for k, v in arg.items():
                                try:
                                    hash(v)
                                    hashable_items.append((k, v))
                                except TypeError:
                                    # If value is unhashable (like a list or dict), serialize it
                                    hashable_items.append((k, json.dumps(v, sort_keys=True, default=str)))
                            hashable_args.append(frozenset(hashable_items))
                        except TypeError:
                            # Fallback if dict contains values that cannot be handled even by serialization
                            hashable_args.append(json.dumps(arg, sort_keys=True, default=str))
                    else:
                        hashable_args.append(arg)
                args_key = tuple(hashable_args)
                
                # Handle kwargs similarly
                hashable_kwargs = []
                for k, v in kwargs.items():
                    try:
                        hash(v)
                        hashable_kwargs.append((k, v))
                    except TypeError:
                        hashable_kwargs.append((k, json.dumps(v, sort_keys=True, default=str)))
                kwargs_key = frozenset(hashable_kwargs)

                key = (args_key, kwargs_key)

            except (TypeError, json.JSONDecodeError) as e:
                # If arguments contain unhashable types that cannot be handled, bypass cache
                # print(f"Bypassing cache for {async_func.__name__} due to unhashable arguments: {e}")
                return await async_func(*args, **kwargs)

            if key in cache:
                if key in order:
                   order.remove(key)
                order.append(key)
                return cache[key]

            result = await async_func(*args, **kwargs)

            if len(cache) >= maxsize:
                if order:
                    oldest_key = order.pop(0)
                    if oldest_key in cache:
                       cache.pop(oldest_key)

            cache[key] = result
            order.append(key)
            return result

        def cache_clear():
            """Clear the cache."""
            nonlocal cache, order
            cache.clear()
            order.clear()
            # print(f"Cache cleared for {async_func.__name__}")

        wrapper.cache_clear = cache_clear
        return wrapper
    return decorator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .data_formatter import safe_json

# Load .env from project root (adjust path for src layout)
# This is kept as a fallback for configuration
try:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))
except Exception:
    pass # Ignore if dotenv is not installed or .env file is missing

# --- Prompt Loading ---
@lru_cache() # Cache results
def load_prompts(prompt_file: str = "prompts.yaml") -> Dict[str, Any]:
    """Loads prompts from the YAML file within the same directory."""
    prompts = {}
    try:
        # Construct path relative to the current file (__file__)
        file_path = Path(__file__).parent / prompt_file
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            prompts = yaml.safe_load(f)
        if not prompts:
            print(f"Warning: Prompt file '{file_path}' is empty or invalid.")
            return {}
        return prompts
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {file_path}. Ensure '{prompt_file}' is in the 'utils' directory.")
        return {} # Return empty dict, agent functions need to handle this
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML from {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading prompts: {e}")
        return {}

# Load prompts ONCE at module level
ALL_PROMPTS = load_prompts()

# Helper to get prompts safely (keep or adapt the existing one)
def get_prompt(key: str, prompt_type: str = "system") -> str:
    """Safely retrieves a specific prompt, providing a default if not found."""
    try:
        prompt = ALL_PROMPTS.get(key, {}).get(prompt_type)
        if prompt is None:
            print(f"Warning: Prompt '{key}.{prompt_type}' not found in YAML. Using default.")
            # Provide a more informative default
            return f"DEFAULT PROMPT: Task details for {key} ({prompt_type}) were expected but not found in prompts.yaml."
        return prompt
    except Exception as e:
         print(f"Error retrieving prompt '{key}.{prompt_type}': {e}. Using default.")
         return f"DEFAULT PROMPT: Error retrieving task details for {key} ({prompt_type})."


# --- LLM Initialization (FIXED: Takes configuration explicitly) ---

def initialize_llm(llm_config: Dict[str, Any], timeout: float = 120.0) -> ChatOpenAI:
    """Initializes and returns the LangChain ChatOpenAI instance based on provided config."""

    if not llm_config:
        raise ValueError("LLM configuration dictionary is missing.")

    api_key = llm_config.get('api_key')
    # Use the model_id which is mapped from the display name in app.py
    model_id = llm_config.get('model_id') 
    api_base = llm_config.get('api_base') # Used for OpenRouter or other proxies
    provider = llm_config.get('provider')

    if not api_key:
        # Fallback logic for pytest environment (if needed)
        if "PYTEST_CURRENT_TEST" in os.environ:
             print("Warning: API Key not set, using dummy key for pytest collection.")
             api_key = "DUMMY_API_KEY_FOR_PYTEST"
             model_id = model_id or "gpt-4.1"
        else:
             raise ValueError("API Key (OpenAI or OpenRouter) is not set in configuration.")

    if not model_id:
        raise ValueError("LLM model ID is not specified in the configuration.")

    # Using langchain's integration for async capabilities
    # ChatOpenAI is compatible with both OpenAI and OpenRouter (which mimics the OpenAI API)
    llm = ChatOpenAI(
        model=model_id,
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=api_base if api_base else None, # Only set if provided (e.g., OpenRouter)
        max_retries=3,
        request_timeout=timeout
    )

    # Specific headers required by OpenRouter (if using LangChain >= 0.1.17)
    if provider == 'OpenRouter' and api_base and "openrouter.ai" in api_base:
        # Check if default_headers attribute exists and initialize if needed
        if hasattr(llm, 'default_headers'):
            if llm.default_headers is None:
                llm.default_headers = {}
            llm.default_headers.update({
                "HTTP-Referer": "https://oqt-assistant.com", # Replace if hosted elsewhere
                "X-Title": "O'QT Assistant"
            })
        # For older LangChain versions, initialization might look different or require custom clients.

    return llm

output_parser = StrOutputParser()

# --- Specialist Agent Functions (MODIFIED to accept llm_config) ---

# --- Chemical Context Agent ---
@async_lru_cache(maxsize=128)
async def analyze_chemical_context(chemical_data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Extracts key chemical identifiers using an LLM agent."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        basic_info = chemical_data.get('basic_info', {})
        data_json = safe_json(basic_info)

        # Get prompts from loaded YAML data
        system_prompt = get_prompt("chem_context", "system")
        user_template = get_prompt("chem_context", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response.strip()
    except Exception as e:
         error_msg = f"Error in Chemical Context Agent: {str(e)}"
         print(error_msg)
         # Fallback if LLM fails
         name = chemical_data.get('basic_info', {}).get('Name', 'Unknown')
         cas = chemical_data.get('basic_info', {}).get('Cas', 'N/A') # Use 'Cas' as per API response
         smiles = chemical_data.get('basic_info', {}).get('Smiles', 'N/A') # Use 'Smiles' as per API response
         return f"Confirmed Chemical: {name} (CAS: {cas}, SMILES: {smiles}) [Error during analysis: {e}]"


# --- Physical Properties Agent ---
@async_lru_cache(maxsize=128)
async def analyze_physical_properties(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes physical properties using an LLM agent, returning text."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("phys_props", "system")
        user_template = get_prompt("phys_props", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Physical Properties Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Environmental Fate Agent ---
@async_lru_cache(maxsize=128)
async def analyze_environmental_fate(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes environmental fate properties using an LLM agent, returning text."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("env_fate", "system")
        user_template = get_prompt("env_fate", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Environmental Fate Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Profiling/Reactivity Agent ---
@async_lru_cache(maxsize=128)
async def analyze_profiling_reactivity(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes profiling and reactivity using an LLM agent, returning text."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("profiling_reactivity", "system")
        user_template = get_prompt("profiling_reactivity", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Profiling/Reactivity Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Experimental Data Agent ---
@async_lru_cache(maxsize=128)
async def analyze_experimental_data(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes experimental data using an LLM agent, returning text."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("experimental_data", "system")
        user_template = get_prompt("experimental_data", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Experimental Data Agent: {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- UPDATED: Metabolism Agent (Fixed logic for multi-simulator) ---
@async_lru_cache(maxsize=128)
async def analyze_metabolism(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes metabolism simulation results using an LLM agent, returning text."""
    # Handle cases where data might be missing
    if not data:
        return "Metabolism data was not available for analysis."

    status = data.get("status", "Unknown")
    
    if status == "Skipped":
        return "Metabolism simulation was skipped by the user."
    
    # Check if there are any simulations recorded
    simulations = data.get("simulations", {})
    if not simulations:
        # Handle cases where the structure exists but no runs occurred (or failed before runs started)
        if status == "Failed" or status == "Error":
             return f"Metabolism simulation failed or encountered an error. Details: {data.get('note', 'N/A')}"
        return "Metabolism data structure is present, but no simulation runs were recorded."

    # FIX: Check if any metabolites were actually generated across all simulations
    total_metabolites = 0
    for sim_result in simulations.values():
        # Ensure metabolites entry is a list before calculating length, and handle potential non-list entries
        metabolites_list = sim_result.get("metabolites", [])
        if isinstance(metabolites_list, list):
            total_metabolites += len(metabolites_list)

    # If no metabolites were found across all simulations
    if total_metabolites == 0:
         # Check if the overall status indicates failure/error even if individual simulations didn't record metabolites
        if status == "Failed" or status == "Error":
             return f"Metabolism simulation failed or encountered an error. No metabolites were generated. Details: {data.get('note', 'N/A')}"
        
        # This message is correctly triggered only if total_metabolites is 0 and status is Success/Partial Success
        return "Metabolism simulation(s) completed, but no metabolites were generated across any selected simulators."

    # If we reach here, we have data to analyze, proceed to LLM call.
    try:
        llm = initialize_llm(llm_config)

        data_json = safe_json(data)
        # Get prompts from loaded YAML data
        system_prompt = get_prompt("metabolism", "system")
        user_template = get_prompt("metabolism", "user")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        chain = prompt_template | llm | output_parser
        input_dict = {"context": context, "data_json": data_json}
        response = await chain.ainvoke(input_dict)
        return response
    except Exception as e:
        error_msg = f"Error in Metabolism Agent (LLM processing): {str(e)}"
        print(error_msg)
        return error_msg # Return error message string

# --- Read Across Agent ---
@async_lru_cache(maxsize=128)
async def analyze_read_across(results: Dict[str, Any], specialist_outputs: List[str], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes data gaps and suggests read-across strategy using an LLM agent with robust timeout handling."""

    # Implement progressive timeout strategy with fallback
    timeout_strategies = [
        {"timeout": 180.0, "description": "Extended timeout (3 minutes)"},
        {"timeout": 300.0, "description": "Long timeout (5 minutes)"},
    ]
    
    error_msg = "N/A"

    for strategy in timeout_strategies:
        try:
            # print(f"Attempting Read Across analysis with {strategy['description']}")

            # FIXED: Initialize LLM with passed config AND custom timeout
            llm = initialize_llm(llm_config, timeout=strategy["timeout"])

            # Prepare data - truncate if too large to reduce processing time
            results_json = safe_json(results)
            specialist_outputs_text = "\n\n---\n\n".join(specialist_outputs)

            # Limit input size to prevent timeouts - truncate if necessary
            max_chars = 30000  # Increased limit slightly due to 5 specialists now
            if len(results_json) > max_chars:
                results_json = results_json[:max_chars] + "\n\n[NOTE: Results data truncated to prevent timeout]"
                # print(f"Truncated results_json to {max_chars} characters to prevent timeout")

            if len(specialist_outputs_text) > max_chars:
                specialist_outputs_text = specialist_outputs_text[:max_chars] + "\n\n[NOTE: Specialist outputs truncated to prevent timeout]"
                # print(f"Truncated specialist_outputs_text to {max_chars} characters to prevent timeout")

            # Get prompts from loaded YAML data
            system_prompt = get_prompt("read_across", "system")
            user_template = get_prompt("read_across", "user")

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_template)
            ])

            chain = prompt_template | llm | output_parser

            input_dict = {
                "context": context,
                "results_json": results_json,
                "specialist_outputs_text": specialist_outputs_text
            }

            # Add asyncio timeout as additional protection
            response = await asyncio.wait_for(
                chain.ainvoke(input_dict),
                timeout=strategy["timeout"] + 30  # Extra buffer
            )

            # print(f"Read Across analysis completed successfully with {strategy['description']}")
            return response

        except asyncio.TimeoutError:
            error_msg = f"Read Across Agent timed out after {strategy['timeout']} seconds"
            print(f"Warning: {error_msg}")
            continue  # Try next strategy

        except Exception as e:
            error_msg = f"Read Across Agent failed with {strategy['description']}: {str(e)}"
            print(f"Warning: {error_msg}")

            # If this is a timeout-related error, try next strategy
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                continue
            else:
                # For non-timeout errors (e.g., AuthenticationError), don't continue trying
                break

    # If all strategies failed, return a meaningful fallback response
    fallback_response = f"""
    Error in Read Across Agent: Analysis failed or timed out.

    No read-across analysis could be provided due to an error or timeout. If read-across or analog data are required for this assessment, it is recommended to rerun the analysis or consult the QSAR Toolbox desktop application.

    **Last Error Message (if available):** {error_msg}

    **Potential Solutions:**
    - Ensure your LLM API configuration is correct (API Key, Model Access).
    - Try running the analysis again, potentially with a smaller dataset or a faster LLM model.
    - Manually identify similar chemicals for read-across analysis.
    """

    print("All timeout strategies exhausted or analysis failed, returning fallback response")
    return fallback_response

# --- Synthesizer Agent ---
@async_lru_cache(maxsize=128)
async def synthesize_report(
    chemical_identifier: str,
    specialist_outputs: List[str],
    read_across_report: str,
    context: str,
    llm_config: Dict[str, Any]
) -> str:
    """Synthesizes text outputs from specialist agents into a final report, including read-across."""
    try:
        llm = initialize_llm(llm_config) # FIXED: Initialize LLM with passed config

        # Updated expected number of outputs from 4 to 5
        EXPECTED_OUTPUTS = 5
        if len(specialist_outputs) != EXPECTED_OUTPUTS:
             print(f"Warning: Expected {EXPECTED_OUTPUTS} original specialist outputs for synthesis, received {len(specialist_outputs)}")
             outputs = specialist_outputs + ["No analysis available."] * (EXPECTED_OUTPUTS - len(specialist_outputs))
        else:
             outputs = specialist_outputs

        analyses = [out if isinstance(out, str) else "Error: Invalid analysis format received." for out in outputs]
        phys_analysis = analyses[0]
        env_analysis = analyses[1]
        prof_analysis = analyses[2]
        exp_analysis = analyses[3]
        metabolism_analysis = analyses[4] # NEW
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
        chain = prompt_template | llm | output_parser
        input_dict = {
            "chemical_identifier": chemical_identifier,
            "context": context,
            "physical_properties_analysis": phys_analysis,
            "environmental_fate_analysis": env_analysis,
            "profiling_reactivity_analysis": prof_analysis,
            "experimental_data_analysis": exp_analysis,
            "metabolism_analysis": metabolism_analysis, # NEW
            "read_across_analysis": read_across_analysis
        }
        response = await chain.ainvoke(input_dict)
        return response
    
    except Exception as e:
        error_msg = f"Error in Synthesizer Agent: {str(e)}"
        print(error_msg)
        
        # Provide specific guidance for common OpenRouter errors
        if "404" in str(e) and "data policy" in str(e).lower():
            return f"""Error synthesizing report for {chemical_identifier}: 

OpenRouter Privacy Settings Issue Detected:
The error suggests your OpenRouter privacy settings are blocking access to this model.

To resolve:
1. Go to https://openrouter.ai/settings/privacy
2. Enable "Providers that may train on inputs"
3. Retry the analysis

Alternatively, try switching to a different model or use direct OpenAI API instead of OpenRouter.

Technical error: {error_msg}"""
        
        return f"Error synthesizing report for {chemical_identifier}: {error_msg}"