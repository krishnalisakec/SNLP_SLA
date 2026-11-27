
import re
import json
from pathlib import Path


class ClinicalPreprocessor:
    """
    Preprocesses clinical notes by cleaning text,
    expanding common medical abbreviations,
    splitting text into sentences and sections.
    """

    def __init__(self):
        self.abbreviations = {
            "pt": "patient",
            "hx": "history",
            "dx": "diagnosis",
            "tx": "treatment",
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "bp": "blood pressure",
            "hr": "heart rate",
            "rr": "respiratory rate"
        }

    def clean_text(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def expand_abbreviations(self, text):
        for short_form, full_form in self.abbreviations.items():
            pattern = r"\b" + re.escape(short_form) + r"\b"
            text = re.sub(
                pattern,
                full_form,
                text,
                flags=re.IGNORECASE
            )
        return text

    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def process(self, text):
        cleaned_text = self.clean_text(text)
        expanded_text = self.expand_abbreviations(cleaned_text)
        sentences = self.split_sentences(expanded_text)

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "expanded_text": expanded_text,
            "sentences": sentences
        }


if __name__ == "__main__":
    sample_note = """
    Pt has hx of HTN and DM.
    Pt reports SOB and CP.
    BP and HR monitored.
    """

    processor = ClinicalPreprocessor()
    result = processor.process(sample_note)

    print("\n===== CLINICAL DOCUMENT PREPROCESSING =====")
    print(json.dumps(result, indent=4))
