import os
from typing import Any

import yaml
from rich.console import Console

console = Console()

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {"provider": "openai"},
    "sources": {
        "congress": {"enabled": True, "limit": 3},
        "regulations": {"enabled": True, "limit": 3},
        "legiscan_federal": {"enabled": True, "limit": 3},
        "texas_bills": {"enabled": False, "bills": []}
    },
    "change_tracking": {"enabled": False}
}


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    From config.yaml

    Args:
        config_path: Path to config file (default: config.yaml)

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print("[red]Error: config.yaml not found. Using default config.[/red]")
        return DEFAULT_CONFIG


def load_nrg_context(context_file: str = "nrg_business_context.txt") -> str:
    """
    Args:
        context_file: Path to context file

    Returns:
        NRG business context string
    """
    try:
        with open(context_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        console.print(f"[red]Warning: {context_file} not found. Using minimal context.[/red]")
        return "NRG Energy is a major US electricity generator with natural gas and oil-fired power plants."


def get_api_keys() -> dict[str, str]:
    return {
        "congress": os.getenv("CONGRESS_API_KEY", ""),
        "regulations": os.getenv("REGULATIONS_API_KEY", ""),
        "legiscan": os.getenv("LEGISCAN_API_KEY", ""),
        "openstates": os.getenv("OPENSTATES_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "google": os.getenv("GOOGLE_API_KEY", ""),
    }
