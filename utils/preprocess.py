# utils/preprocess.py
import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters and lowercasing it.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()
