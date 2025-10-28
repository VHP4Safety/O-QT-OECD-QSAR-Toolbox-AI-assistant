# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for LLM-based report generation using a multi-agent approach.
"""
import os
import asyncio
import json
from functools import lru_cache
from functools import wraps
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# NEW: Global semaphore for TPM rate-limit control (lazy-initialized per event loop)
_MAX_CONC = int(os.getenv("OQT_MAX_LLM_CONCURRENCY", "2"))
_LLM_SEMAPHORE = None
_SEMAPHORE_LOOP_ID = None

def _get_semaphore():
    """Get or create semaphore for the current event loop."""
    global _LLM_SEMAPHORE, _SEMAPHORE_LOOP_ID
    
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        
        # Create new semaphore if we're in a different event loop
        if _SEMAPHORE_LOOP_ID != loop_id:
            _LLM_SEMAPHORE = asyncio.Semaphore(_MAX_CONC)
            _SEMAPHORE_LOOP_ID = loop_id
            logger.debug(f"Created new semaphore for event loop {loop_id} (concurrency={_MAX_CONC})")
        
        return _LLM_SEMAPHORE
    except RuntimeError:
        # No event loop running, return None
        return None

logger.info(f"LLM concurrency limit set to {_MAX_CONC} (via OQT_MAX_LLM_CONCURRENCY)")

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
                # logger.debug(f"Bypassing cache for {async_func.__name__} due to unhashable arguments: {e}")
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
            # logger.info(f"Cache cleared for {async_func.__name__}")

        wrapper.cache_clear = cache_clear
        return wrapper
    return decorator

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .data_formatter import safe_json

# Load .env from project root (adjust path for src layout)
# This is kept as a fallback for configuration
try:
    # Adjust the pathfinding logic to be more robust
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if 'src' is in the path (common development structure)
    if 'src' in current_dir.split(os.sep):
        # Assuming llm_utils.py is in src/oqt_assistant/utils, the root is three levels up
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    else:
        # Assuming package installation structure
        project_root = os.path.dirname(os.path.dirname(current_dir))

    dotenv_path = os.path.join(project_root, '.env')
    
    # Fallback to the original highly nested logic
    if not os.path.exists(dotenv_path):
         fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
         if os.path.exists(fallback_path):
             dotenv_path = fallback_path
             
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

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
            logger.warning(f"Prompt file '{file_path}' is empty or invalid.")
            return {}
        return prompts
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {file_path}. Ensure '{prompt_file}' is in the 'utils' directory.")
        return {} # Return empty dict, agent functions need to handle this
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML from {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred loading prompts: {e}")
        return {}

# Load prompts ONCE at module level
ALL_PROMPTS = load_prompts()

# Helper to get prompts safely
def get_prompt(key: str, prompt_type: str = "system") -> str:
    """Safely retrieves a specific prompt, providing a default if not found."""
    try:
        prompt = ALL_PROMPTS.get(key, {}).get(prompt_type)
        if prompt is None:
            logger.warning(f"Prompt '{key}.{prompt_type}' not found in YAML. Using default.")
            # Provide a more informative default
            return f"DEFAULT PROMPT: Task details for {key} ({prompt_type}) were expected but not found in prompts.yaml."
        return prompt
    except Exception as e:
         logger.error(f"Error retrieving prompt '{key}.{prompt_type}': {e}. Using default.")
         return f"DEFAULT PROMPT: Error retrieving task details for {key} ({prompt_type})."


# --- LLM Initialization (UPDATED: Uses get_llm for GPT-5 support) ---

def initialize_llm(llm_config: Dict[str, Any], timeout: float = 120.0) -> BaseChatModel:
    """Initializes and returns the LangChain chat model based on provided config."""
    
    # Import get_llm function
    from .llm_provider import get_llm

    if not llm_config:
        raise ValueError("LLM configuration dictionary is missing.")

    api_key = llm_config.get('api_key')
    # Use the model_id which is mapped from the display name in app.py
    model_id = llm_config.get('model_id')
    api_base = llm_config.get('api_base') # Used for OpenRouter or other proxies
    provider = llm_config.get('provider')

    # NEW: Handle overrides (set in app.py based on wizard config)
    # Check for overrides first, otherwise use defaults
    temperature = llm_config.get('temperature_override')
    if temperature is None:
        temperature = llm_config.get('temperature', 0.15) # Default temperature changed to 0.15

    max_tokens = llm_config.get('max_tokens_override')
    if max_tokens is None:
        max_tokens = llm_config.get('max_tokens', 10000) # Default max_tokens

    # Get reasoning effort for GPT-5 models
    reasoning_effort = llm_config.get('reasoning_effort')

    if not api_key:
        # Fallback logic for pytest environment (if needed)
        if "PYTEST_CURRENT_TEST" in os.environ:
             logger.warning("API Key not set, using dummy key for pytest collection.")
             api_key = "DUMMY_API_KEY_FOR_PYTEST"
             model_id = model_id or "gpt-4.1"
        else:
             raise ValueError("API Key (OpenAI or OpenRouter) is not set in configuration.")

    if not model_id:
        raise ValueError("LLM model ID is not specified in the configuration.")

    # Determine provider for get_llm function
    if provider == 'OpenRouter' or (api_base and "openrouter.ai" in api_base):
        llm_provider = "openai-compatible"
        # Set environment variables for get_llm to use
        os.environ["OPENAI_BASE_URL"] = api_base or "https://openrouter.ai/api/v1"
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        llm_provider = "openai"
        # Set environment variable for OpenAI
        os.environ["OPENAI_API_KEY"] = api_key

    # Use the updated get_llm function that handles GPT-5 parameters correctly
    try:
        llm = get_llm(
            provider=llm_provider,
            model=model_id,
            temperature=temperature,
            max_output_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Primary LLM init failed, switching to fallback generator: {e}")
        llm = _FallbackChat()

    # Set additional properties that get_llm doesn't handle
    if hasattr(llm, 'max_retries'):
        llm.max_retries = 3
    if hasattr(llm, 'request_timeout'):
        llm.request_timeout = timeout

    # Specific headers required by OpenRouter (if using LangChain >= 0.1.17)
    if provider == 'OpenRouter' and api_base and "openrouter.ai" in api_base:
        # Check if default_headers attribute exists and initialize if needed
        if hasattr(llm, 'default_headers'):
            if llm.default_headers is None:
                llm.default_headers = {}
            llm.default_headers.update({
                "HTTP-Referer": "https://oqt-assistant.com", # Replace if hosted elsewhere
                "X-Title": "O-QT Assistant"
            })

    return llm

output_parser = StrOutputParser()

# NEW: Helper function for robust chain invocation
async def _invoke_chain(chain, input_dict, agent_name: str) -> str:
    """Invokes the LangChain chain with TPM throttling and ensures a non-empty string response."""
    try:
        # NEW: Throttle concurrent LLM calls to prevent TPM bursts
        semaphore = _get_semaphore()
        if semaphore:
            async with semaphore:
                response = await chain.ainvoke(input_dict)
        else:
            # No event loop, invoke directly (shouldn't happen in async context)
            response = await chain.ainvoke(input_dict)
        
        # Check if response is valid (it should be a string from StrOutputParser)
        if not isinstance(response, str) or not response.strip():
            logger.warning(f"{agent_name} returned an empty or invalid response.")
            # Return a placeholder message to ensure the pipeline continues
            return f"[{agent_name} Analysis: No response generated. This section could not be completed.]"
            
        return response.strip()
    
    except Exception as e:
        # Catch exceptions during invocation (e.g., API errors, timeouts within LangChain)
        logger.error(f"Error during invocation of {agent_name}: {e}", exc_info=True)
        # Fallback: try direct OpenAI call to generate some content
        try:
            prompt = _build_fallback_prompt(agent_name, input_dict)
            text = _fallback_openai_generate(prompt)
            if text:
                return text.strip()
        except Exception as inner:
            logger.error(f"Fallback OpenAI generation failed for {agent_name}: {inner}")
        # Final minimal placeholder
        return f"[{agent_name}: Content not available due to LLM error.]"

# --- Lightweight Fallback Utilities ---
def _build_fallback_prompt(agent_name: str, input_dict: Dict[str, Any]) -> str:
    ctx = input_dict.get("context") or ""
    # Find any JSON-like field
    payload = None
    for k in ("data_json", "experimental_results", "qsar_processed", "profiling", "metabolism"):
        if k in input_dict and input_dict[k]:
            payload = input_dict[k]
            break
    if isinstance(payload, (dict, list)):
        try:
            payload = json.dumps(payload, ensure_ascii=False)[:4000]
        except Exception:
            payload = str(payload)[:4000]
    else:
        payload = str(payload)[:4000] if payload else ""
    return (
        f"You are the {agent_name} for a QSAR/QPRF report.\n"
        f"Context: {ctx}\n"
        f"Input data (JSON or text, truncated):\n{payload}\n\n"
        f"Write a concise, factual section suitable for the report. Use bullets where natural."
    )


def _fallback_openai_generate(prompt: str, model: str = None, max_output_tokens: int = 800) -> str:
    """Use OpenAI Responses API directly to get a text generation, bypassing LangChain.

    Respects OPENAI_API_KEY in the environment. Chooses a stable default model if not provided.
    """
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for fallback generation")
    client = OpenAI(api_key=api_key)
    mdl = (model or os.environ.get("OQT_FALLBACK_OPENAI_MODEL") or "gpt-4o-mini")
    try:
        resp = client.responses.create(model=mdl, input=prompt, max_output_tokens=max_output_tokens)
        out = getattr(resp, 'output_text', None)
        if not out and getattr(resp, 'choices', None):
            # Defensive for older SDKs
            choice0 = resp.choices[0] if resp.choices else None
            out = getattr(getattr(choice0, 'message', None), 'content', None) if choice0 else None
        return out or ""
    except Exception as e:
        raise RuntimeError(f"OpenAI fallback call failed: {e}")


# --- Specialist Agent Functions (MODIFIED to use _invoke_chain) ---

# Note: Exceptions raised by _invoke_chain are propagated up to app.py for central handling.

# --- Chemical Context Agent ---
@async_lru_cache(maxsize=128)
async def analyze_chemical_context(chemical_data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Extracts key chemical identifiers using an LLM agent."""
    llm = initialize_llm(llm_config)

    basic_info = chemical_data.get('basic_info', {})
    data_json = safe_json(basic_info)

    # Get prompts
    system_prompt = get_prompt("chem_context", "system")
    user_template = get_prompt("chem_context", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Chemical Context Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Chemical Context Agent")
    return response
        

# --- Physical Properties Agent ---
@async_lru_cache(maxsize=128)
async def analyze_physical_properties(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes physical properties using an LLM agent, returning text."""
    llm = initialize_llm(llm_config)

    data_json = safe_json(data)
    # Get prompts
    system_prompt = get_prompt("phys_props", "system")
    user_template = get_prompt("phys_props", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Physical Properties Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Physical Properties Agent")
    return response

# --- Environmental Fate Agent ---
@async_lru_cache(maxsize=128)
async def analyze_environmental_fate(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes environmental fate properties using an LLM agent, returning text."""
    llm = initialize_llm(llm_config)

    data_json = safe_json(data)
    # Get prompts
    system_prompt = get_prompt("env_fate", "system")
    user_template = get_prompt("env_fate", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Environmental Fate Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Environmental Fate Agent")
    return response

# --- Profiling/Reactivity Agent ---
@async_lru_cache(maxsize=128)
async def analyze_profiling_reactivity(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes profiling and reactivity using an LLM agent, returning text."""
    llm = initialize_llm(llm_config)

    data_json = safe_json(data)
    # Get prompts
    system_prompt = get_prompt("profiling_reactivity", "system")
    user_template = get_prompt("profiling_reactivity", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Profiling/Reactivity Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Profiling/Reactivity Agent")
    return response

# --- Experimental Data Agent ---
@async_lru_cache(maxsize=128)
async def analyze_experimental_data(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes experimental data using an LLM agent with smart sampling for large payloads."""
    llm = initialize_llm(llm_config)
    
    exp_results = data.get("experimental_results", [])
    
    # Smart sampling - prioritize most recent records
    max_records = int(os.getenv("OQT_EXP_MAX_RECORDS", "500"))
    
    if len(exp_results) > max_records:
        logger.info(f"Experimental data has {len(exp_results)} records, selecting top {max_records} most relevant")
        
        # Prioritize by publication year (most recent first)
        def sort_key(record):
            year = record.get("Publication_Year", 0)
            try:
                year = int(year) if year else 0
            except (ValueError, TypeError):
                year = 0
            return -year
        
        sorted_results = sorted(exp_results, key=sort_key)
        exp_results = sorted_results[:max_records]
        logger.info(f"Retained {len(exp_results)} experimental records after year-based prioritization")
    
    # Prepare data for LLM
    processed_data = {"experimental_results": exp_results}
    data_json = safe_json(processed_data)
    
    # Get prompts
    system_prompt = get_prompt("experimental_data", "system")
    user_template = get_prompt("experimental_data", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Experimental Data Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    response = await _invoke_chain(chain, input_dict, "Experimental Data Agent")
    return response

# --- UPDATED: Metabolism Agent ---
@async_lru_cache(maxsize=128)
async def analyze_metabolism(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes metabolism simulation results using an LLM agent, returning text."""
    # Handle cases where data might be missing or skipped (pre-LLM checks)
    if not data:
        return "Metabolism data was not available for analysis."

    status = data.get("status", "Unknown")
    
    if status == "Skipped":
        return "Metabolism simulation was skipped by the user."
    
    # Check if there are any simulations recorded
    simulations = data.get("simulations", {})
    if not simulations:
        # Handle cases where the structure exists but simulations dictionary is empty
        if status in ["Failed", "Error"]:
            return f"Metabolism analysis cannot proceed. Overall Status: {status}. Note: {data.get('note', 'No details provided.')}"
        else:
            return "Metabolism data structure found, but no individual simulation results were available for analysis."

    # Proceed with LLM analysis if simulations exist
    llm = initialize_llm(llm_config)

    # Prepare data for LLM
    data_json = safe_json(data)

    # Get prompts
    system_prompt = get_prompt("metabolism", "system")
    user_template = get_prompt("metabolism", "user")

    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Metabolism Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Metabolism Agent")
    return response

# --- QSAR Predictions Agent ---
@async_lru_cache(maxsize=128)
async def analyze_qsar_predictions(data: Dict[str, Any], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes QSAR predictions (filtered to in-domain models) using an LLM agent."""
    predictions = data.get("in_domain") if isinstance(data, dict) else None
    if not predictions:
        total_models = data.get("summary", {}).get("total", 0) if isinstance(data, dict) else 0
        if total_models:
            return f"No QSAR models reported the chemical within their applicability domain (evaluated {total_models} models)."
        return "QSAR predictions were not available for analysis."

    llm = initialize_llm(llm_config)
    data_json = safe_json(data)

    system_prompt = get_prompt("qsar_predictions", "system")
    user_template = get_prompt("qsar_predictions", "user")
    input_dict = {"context": context, "data_json": data_json}
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("QSAR Predictions Agent", input_dict))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser

    response = await _invoke_chain(chain, input_dict, "QSAR Predictions Agent")
    return response


# --- Read Across Agent (UPDATED: Robust invocation within timeout loop) ---
@async_lru_cache(maxsize=128)
async def analyze_read_across(results: Dict[str, Any], specialist_outputs: List[str], context: str, llm_config: Dict[str, Any]) -> str:
    """Analyzes data gaps and suggests read-across strategy using an LLM agent with robust timeout handling."""

    # Implement progressive timeout strategy
    timeout_strategies = [
        {"timeout": 180.0, "description": "Extended timeout (3 minutes)"},
        {"timeout": 300.0, "description": "Long timeout (5 minutes)"},
    ]
    
    error_msg = "N/A"
    
    # Extract scope configuration if available
    scope_config = results.get('scope_config', {})
    rax_strategy = scope_config.get('rax_strategy', 'Default/Hybrid')
    rax_similarity_basis = scope_config.get('rax_similarity_basis', 'Default/Combined')
    
    # Prepare context enhancement based on scope
    scope_context = f"\n\n[ANALYSIS SCOPE CONFIGURATION]\nRequested Read-Across Approach: {rax_strategy}\nRequested Similarity Basis: {rax_similarity_basis}\n"

    # Prepare data - truncate if too large
    results_json = safe_json(results)
    specialist_outputs_text = "\n\n---\n\n".join(specialist_outputs)

    # Limit input size
    max_chars = 30000
    if len(results_json) > max_chars:
        results_json = results_json[:max_chars] + "\n\n[NOTE: Results data truncated to prevent timeout]"

    if len(specialist_outputs_text) > max_chars:
        specialist_outputs_text = specialist_outputs_text[:max_chars] + "\n\n[NOTE: Specialist outputs truncated to prevent timeout]"

    # Get prompts
    system_prompt = get_prompt("read_across", "system")
    user_template = get_prompt("read_across", "user")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])

    input_dict = {
        "context": context,
        "scope_context": scope_context,
        "results_json": results_json,
        "specialist_outputs_text": specialist_outputs_text
    }

    for strategy in timeout_strategies:
        try:
            # Initialize LLM with custom timeout
            llm = initialize_llm(llm_config, timeout=strategy["timeout"])
            # If fallback LLM is active, skip LCEL chain and generate directly
            if isinstance(llm, _FallbackChat):
                return _fallback_openai_generate(_build_fallback_prompt("Read Across Agent", input_dict))
            chain = prompt_template | llm | output_parser

            # Add asyncio timeout AND use robust helper
            async def invoke_with_timeout():
                # We use _invoke_chain here. If it raises an exception, asyncio.wait_for will propagate it.
                return await _invoke_chain(chain, input_dict, "Read Across Agent")

            response = await asyncio.wait_for(
                invoke_with_timeout(),
                timeout=strategy["timeout"] + 30  # Extra buffer
            )

            return response

        except asyncio.TimeoutError:
            error_msg = f"Read Across Agent timed out after {strategy['timeout']} seconds"
            logger.warning(error_msg)
            continue  # Try next strategy

        except Exception as e:
            # This catches exceptions raised by _invoke_chain (e.g., API errors)
            error_msg = f"Read Across Agent failed with {strategy['description']}: {str(e)}"
            logger.warning(error_msg, exc_info=True)

            # If this is a timeout-related error (caught internally by LangChain and raised), try next strategy
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                continue
            else:
                # For non-timeout errors (e.g., AuthenticationError), propagate the error immediately
                # so app.py can handle it globally.
                raise e

    # If all strategies failed (only due to timeouts, as other errors are raised), return fallback response
    fallback_response = f"""
    # Read-Across Analysis (Fallback Response)

    **Status:** Analysis could not be completed due to processing limitations (timeouts or errors).

    **Error Details:** {error_msg}

    **Data Gap Identification:**
    Due to the analysis failure, a comprehensive identification of data gaps based on the provided results could not be performed by the AI agent. Users should manually review the 'Experimental Data' tab to identify missing endpoints.

    **Read-Across Strategy Suggestion:**
    A detailed, context-specific read-across strategy could not be generated. However, a general approach is recommended:
    1. **Identify Key Endpoints:** Determine which endpoints are critical based on the analysis context.
    2. **Assess Similarity:** Look for chemicals with high structural similarity (using SMILES comparison tools) AND similar reactivity profiles (review the 'Profiling' tab).
    3. **Consider Metabolism:** Review the predicted metabolites in the 'Metabolism' tab. Analogues should ideally share similar metabolic pathways.
    4. **Data Availability:** Prioritize analogues known to have rich experimental data.

    **Suggested Chemical Analogues:**
    Specific analogue suggestions cannot be provided at this time.
    """

    logger.error("All timeout strategies exhausted or analysis failed, returning fallback response")
    return fallback_response

# --- Synthesizer Agent (UPDATED: Uses _invoke_chain) ---
@async_lru_cache(maxsize=128)
async def synthesize_report(chemical_identifier: str, specialist_outputs: List[str], read_across_report: str, context: str, llm_config: Dict[str, Any]) -> str:
    """Synthesizes the final report from specialist agent outputs."""
    
    llm = initialize_llm(llm_config)

    # Combine specialist outputs
    combined_specialist_text = "\n\n---\n\n".join(specialist_outputs)

    # Get prompts
    system_prompt = get_prompt("synthesizer", "system")
    user_template = get_prompt("synthesizer", "user")

    input_dict = {
        "chemical_identifier": chemical_identifier,
        "context": context,
        "combined_specialist_text": combined_specialist_text,
        "read_across_report": read_across_report
    }
    if isinstance(llm, _FallbackChat):
        return _fallback_openai_generate(_build_fallback_prompt("Synthesizer Agent", input_dict))
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template)
    ])
    chain = prompt_template | llm | output_parser
    
    # Use the robust helper
    response = await _invoke_chain(chain, input_dict, "Synthesizer Agent")
    return response
class _FallbackChat:
    """Minimal async-compatible LLM for LCEL chains using direct OpenAI calls."""
    max_retries = 1
    request_timeout = 60

    async def ainvoke(self, prompt_value, **kwargs):  # noqa: D401
        # Extract text from LangChain prompt value or list of messages
        try:
            if hasattr(prompt_value, 'to_messages'):
                messages = prompt_value.to_messages()
            elif isinstance(prompt_value, list):
                messages = prompt_value
            else:
                messages = [prompt_value]
            parts = []
            for m in messages:
                content = getattr(m, 'content', None)
                if isinstance(content, list):
                    content = ' '.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
                if not content and isinstance(m, str):
                    content = m
                if content:
                    parts.append(str(content))
            prompt = '\n\n'.join(parts)
        except Exception:
            prompt = str(prompt_value)

        # Use the same fallback generator
        return _fallback_openai_generate(prompt)
