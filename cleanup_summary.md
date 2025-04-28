# Streamlined QSAR App Cleanup Summary

This document summarizes the proposed changes to clean up the `streamlined_qsar_app` project, removing unnecessary files and code identified during the review process.

## Files to Remove

The following files appear to be unused, redundant, or legacy code and are recommended for removal:

1.  **`streamlined_qsar_app/cleanup.py`**
    *   **Reason:** This script's functionality (removing temporary files, build artifacts) is largely covered by the `.gitignore` file, making it redundant. Standard practice is to rely on `.gitignore`.
2.  **`streamlined_qsar_app/setup.py`**
    *   **Reason:** The project's `README.md` instructs users to run the app via `streamlit run app.py` after installing dependencies from `requirements.txt`. It does not mention installing the project as a package using `setup.py`, making this file unnecessary for the documented usage pattern.
3.  **`streamlined_qsar_app/utils/api_client.py`**
    *   **Reason:** This file defines an asynchronous API client using `aiohttp`. However, the main application (`app.py`) exclusively uses the synchronous client defined in `utils/qsar_api.py` (using `requests`). This `api_client.py` appears to be an unused alternative or older version.
4.  **`streamlined_qsar_app/utils/report_generator.py`**
    *   **Reason:** This module defines functions for report generation using the `openai` library directly. This functionality seems entirely superseded by the multi-agent LangChain implementation in `utils/llm_utils.py`, which is what `app.py` actually uses. This file appears to be unused legacy code.

## Files to Modify

The following files require modifications for cleanup and consistency:

1.  **`streamlined_qsar_app/.gitignore`**
    *   **Change:** Add patterns for common testing artifacts and log files.
    *   **Lines to Add:**
        ```gitignore
        # Testing
        .pytest_cache/
        .coverage

        # Logs
        *.log
        ```

2.  **`streamlined_qsar_app/app.py`**
    *   **Change:** Remove commented-out, unused imports.
    *   **Lines to Remove:**
        ```python
        # import json # No longer needed for specialist output formatting
        # from pydantic import BaseModel # No longer needed
        ```

3.  **`streamlined_qsar_app/utils/data_formatter.py`**
    *   **Change 1:** Move imports currently inside the `safe_json` function to the top of the file for standard practice.
        *   **Imports to Move:** `json`, `decimal`, `numpy as np`
    *   **Change 2:** Replace the `print()` call in the `except` block within `clean_response_data` with a proper logging call (e.g., `logging.warning()`) for consistency.
        *   **Line to Modify:** `print(f"Error cleaning profiling data: {e}")` -> `logging.warning(f"Error cleaning profiling data: {e}")` (requires `import logging` at the top).

4.  **`streamlined_qsar_app/utils/llm_utils.py`**
    *   **Change 1:** Remove the unused top-level `import json`.
    *   **Change 2:** Remove the redundant module-level synthesizer prompts (`SYNTHESIZER_SYSTEM_PROMPT`, `SYNTHESIZER_USER_TEMPLATE`) as the function uses different prompts defined internally.
    *   **Change 3:** Replace `print()` calls within the `except` blocks of the agent functions (e.g., `analyze_physical_properties`) with `logging.error()` for consistency (requires `import logging` at the top).

## Next Steps

1.  Review this summary.
2.  If approved, switch to **Act Mode**.
3.  Execute the file removals and modifications outlined above.
