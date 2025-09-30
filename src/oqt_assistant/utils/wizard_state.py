# File: src/qsar_assistant/utils/wizard_state.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import time

WIZ_VERSION = "1.0.0"

READ_ACROSS_DEFINITION = (
    "Read‑across potential is the degree to which one or more source chemicals can "
    "scientifically justify filling data gaps for a target chemical. It integrates "
    "structural similarity (substructures/functional groups), mechanistic congruence "
    "(profiler alerts/modes of action, metabolites), and property comparability "
    "(e.g., logKow, Koc, solubility), weighted per endpoint and data availability."
)

@dataclass
class LLMConfig:
    provider: str = "openai"       # "openai" | "openai-compatible" | "none"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 1200
    cost_per_million_in: float = 3.0    # editable UI hint, not a source of truth
    cost_per_million_out: float = 15.0  # editable UI hint, not a source of truth

@dataclass
class ScopeConfig:
    include_properties: bool = True
    include_experimental: bool = True
    include_profilers: bool = True
    include_metabolism: bool = False
    # Reviewer request: allow explicit exclusion
    exclude_adme_tk: bool = True
    exclude_mammalian_endpoints: bool = True
    # Optional advanced picks (names or GUIDs if you wire them later)
    calculators_selected: List[str] = field(default_factory=list)
    profilers_selected: List[str] = field(default_factory=list)
    metabolism_sims_selected: List[str] = field(default_factory=list)

@dataclass
class ReadAcrossConfig:
    strategy: str = "both"  # "analogue" | "category" | "both"
    endpoints_of_interest: List[str] = field(default_factory=lambda: [
        "Ecotoxicity", "Persistence/Bioaccumulation", "Genotoxicity/Mutagenicity",
        "Carcinogenicity", "Reproductive/Developmental Toxicity", "Neurotoxicity",
        "Endocrine Disruption", "Irritation/Corrosion", "Sensitization"
    ])
    include_metabolites: bool = True
    # Simple weights; agents can consume these as hints
    weight_structure: float = 0.4
    weight_mechanism: float = 0.4
    weight_properties: float = 0.2
    # Free‑text adds traceability in the final report
    justification_notes: str = READ_ACROSS_DEFINITION

@dataclass
class SetupConfig:
    api_url: str = ""      # auto‑filled from env/your sidebar
    context: str = ""      # user goal for analysis
    analysis_label: str = ""  # free text (“Project / batch name”)
    export_pdf: bool = True
    seed: int = 42         # for near‑determinism in LLMs when supported

@dataclass
class ChemicalConfig:
    search_type: str = "name"  # "name" | "smiles" | "cas"
    identifier: str = ""
    resolved: bool = False
    selected_chemical_id: Optional[str] = None
    selected_display_name: Optional[str] = None

@dataclass
class WizardState:
    version: str = WIZ_VERSION
    created_at: float = field(default_factory=time.time)
    setup: SetupConfig = field(default_factory=SetupConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    readacross: ReadAcrossConfig = field(default_factory=ReadAcrossConfig)
    chemical: ChemicalConfig = field(default_factory=ChemicalConfig)
    # Runtime flags (not part of recipe)
    current_step: int = 0
    last_cost_estimate_usd: float = 0.0
    last_token_budget_in: int = 0
    last_token_budget_out: int = 0

    def to_recipe(self) -> Dict[str, Any]:
        """Export a reproducible recipe JSON (sans transient runtime fields)."""
        d = asdict(self)
        d.pop("current_step", None)
        d.pop("last_cost_estimate_usd", None)
        d.pop("last_token_budget_in", None)
        d.pop("last_token_budget_out", None)
        return d

    def to_recipe_json(self) -> str:
        return json.dumps(self.to_recipe(), indent=2)
