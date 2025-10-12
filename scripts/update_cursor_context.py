#!/usr/bin/env python3
"""
Automatically update .cursorrules file with latest project context.

Run this script after major code changes to keep Cursor context up-to-date.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_info():
    """Get current git branch and latest commit."""
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                        stderr=subprocess.DEVNULL).decode().strip()
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                        stderr=subprocess.DEVNULL).decode().strip()
        return branch, commit
    except:
        return "main", "unknown"


def count_lines_in_file(filepath):
    """Count lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0


def get_project_stats():
    """Get project statistics."""
    root = Path(__file__).parent.parent
    
    stats = {
        'transaction_web_app_lines': count_lines_in_file(root / 'transaction_web_app.py'),
        'data_store_lines': count_lines_in_file(root / 'data_store.py'),
        'test_files': len(list((root / 'tests').glob('test_*.py'))) if (root / 'tests').exists() else 0,
        'total_tests': 0,
    }
    
    # Count total tests
    if (root / 'tests').exists():
        for test_file in (root / 'tests').glob('test_*.py'):
            with open(test_file, 'r') as f:
                content = f.read()
                stats['total_tests'] += content.count('def test_')
    
    return stats


def update_cursorrules():
    """Update the .cursorrules file with latest information."""
    root = Path(__file__).parent.parent
    cursorrules_path = root / '.cursorrules'
    
    # Read current file
    with open(cursorrules_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get dynamic information
    branch, commit = get_git_info()
    stats = get_project_stats()
    today = datetime.now().strftime('%B %d, %Y')
    
    # Update last updated date
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('**Last Updated**:'):
            lines[i] = f'**Last Updated**: {today}'
        elif '(Main UI -' in line and 'lines)' in line:
            lines[i] = f"   - Streamlit web interface ({stats['transaction_web_app_lines']}+ lines)"
        elif '(Database Layer -' in line and 'lines)' in line:
            lines[i] = f"2. **data_store.py** (Database Layer - {stats['data_store_lines']} lines)"
    
    # Update test count if mentioned
    updated_content = '\n'.join(lines)
    
    # Write back
    with open(cursorrules_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ Updated .cursorrules")
    print(f"   - Date: {today}")
    print(f"   - Branch: {branch}")
    print(f"   - Commit: {commit}")
    print(f"   - Tests: {stats['total_tests']}")


def main():
    """Main entry point."""
    print("🔄 Updating Cursor context...")
    update_cursorrules()
    print("✨ Done!")


if __name__ == '__main__':
    main()

