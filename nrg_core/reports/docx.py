"""
Word document generation for legislative analysis reports.

Uses Pandoc to convert Markdown to DOCX format.
"""

import subprocess
from typing import Optional

from rich.console import Console

console = Console()


def convert_markdown_to_word(md_file: str) -> Optional[str]:
    try:
        docx_file = md_file.replace('.md', '.docx')

        # Check if pandoc is available
        result = subprocess.run(
            ['which', 'pandoc'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            console.print("[yellow]⚠ Pandoc not found - skipping Word document generation[/yellow]")
            console.print("[dim]  Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)[/dim]")
            return None

        # Convert markdown to Word
        result = subprocess.run(
            ['pandoc', md_file, '-o', docx_file],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return docx_file
        else:
            console.print(f"[yellow]⚠ Pandoc conversion failed: {result.stderr}[/yellow]")
            return None

    except Exception as e:
        console.print(f"[yellow]⚠ Error converting to Word: {e}[/yellow]")
        return None
