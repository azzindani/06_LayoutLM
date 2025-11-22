#!/usr/bin/env python
"""
Convert original Jupyter notebook to Python modules.
This script helps extract and organize code from the original notebook.
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_notebook_code(notebook_path: str) -> str:
    """
    Extract all code cells from a Jupyter notebook.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Concatenated code from all cells
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    code_blocks = []

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            code_blocks.append(source)

    return '\n\n'.join(code_blocks)


def analyze_notebook(notebook_path: str) -> dict:
    """
    Analyze notebook structure.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Analysis results
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    analysis = {
        'total_cells': len(notebook.get('cells', [])),
        'code_cells': 0,
        'markdown_cells': 0,
        'imports': [],
        'functions': [],
        'classes': []
    }

    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')

        if cell_type == 'code':
            analysis['code_cells'] += 1
            source = ''.join(cell.get('source', []))

            # Find imports
            for line in source.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    analysis['imports'].append(line.strip())

            # Find function definitions
            for line in source.split('\n'):
                if line.strip().startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    analysis['functions'].append(func_name)

            # Find class definitions
            for line in source.split('\n'):
                if line.strip().startswith('class '):
                    class_name = line.split('(')[0].split(':')[0].replace('class ', '').strip()
                    analysis['classes'].append(class_name)

        elif cell_type == 'markdown':
            analysis['markdown_cells'] += 1

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and convert Jupyter notebook"
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default="ocr-layoutlmv3-base-v2.ipynb",
        help="Path to notebook file"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract code to stdout"
    )

    args = parser.parse_args()

    if not os.path.exists(args.notebook):
        print(f"Error: Notebook not found: {args.notebook}")
        sys.exit(1)

    if args.extract:
        code = extract_notebook_code(args.notebook)
        print(code)
    else:
        print("=" * 60)
        print("Notebook Analysis")
        print("=" * 60)

        analysis = analyze_notebook(args.notebook)

        print(f"\nFile: {args.notebook}")
        print(f"\nCells:")
        print(f"  Total: {analysis['total_cells']}")
        print(f"  Code: {analysis['code_cells']}")
        print(f"  Markdown: {analysis['markdown_cells']}")

        print(f"\nImports ({len(analysis['imports'])}):")
        for imp in analysis['imports'][:10]:
            print(f"  - {imp}")
        if len(analysis['imports']) > 10:
            print(f"  ... and {len(analysis['imports']) - 10} more")

        print(f"\nFunctions ({len(analysis['functions'])}):")
        for func in analysis['functions']:
            print(f"  - {func}")

        print(f"\nClasses ({len(analysis['classes'])}):")
        for cls in analysis['classes']:
            print(f"  - {cls}")

        print("\n" + "=" * 60)
