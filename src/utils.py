"""
Utility functions for the adversarial grading system.
"""

from typing import Dict, Any


def format_rubric(rubric: Dict[str, Any]) -> str:
    """
    Format rubric dictionary into readable text for prompts.
    
    Args:
        rubric: Rubric dictionary with 'description' and 'items'
        
    Returns:
        Formatted string representation of the rubric
    """
    items_text = []
    for item in rubric['items']:
        item_text = f"\nItem: {item['label']} (id: {item['id']})\n"
        for option in item['options']:
            item_text += f"  {option['optionId']}: {option['label']}\n"
        items_text.append(item_text)
    
    return f"Description: {rubric['description']}\n\nRubric Items:\n" + "\n".join(items_text)

