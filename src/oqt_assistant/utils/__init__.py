# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit UI components for O-QT Assistant.

⚠️ Important:
Do not import submodules here (no eager imports). Eager imports can trigger
side effects and circular/partial initialization during package import,
e.g., importing legacy 'wizard' which pulls utility functions too early.

Import submodules explicitly where needed, for example:
    from oqt_assistant.components.search import render_search_section
    from oqt_assistant.components.results import render_results_section
    from oqt_assistant.components.guided_wizard import run_guided_wizard
"""
