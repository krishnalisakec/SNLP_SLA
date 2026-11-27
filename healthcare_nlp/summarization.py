
import re
import json


class ClinicalSummarizer:

    def __init__(self):

        self.medical_categories = {
            "DISEASE": [
                "diabetes mellitus", "diabetes",
                "hypertension", "asthma", "pneumonia",
                "respiratory infection"
            ],

            "SYMPTOM": [
                "shortness of breath", "persistent cough",
                "chest pain", "headache", "fatigue",
                "fever", "cough"
            ],

            "MEDICATION": [
                "metformin", "insulin", "aspirin",
                "paracetamol", "medication"
            ],

            "INVESTIGATION": [
                "blood tests", "blood test",
                "blood glucose", "blood pressure",
                "heart rate", "chest x-ray", "x-ray"
            ],

            "TREATMENT": [
                "treatment", "prescribed", "monitor",
                "investigation", "follow-up", "follow up"
            ]
        }

        self.important_phrases = [
            "possible", "possibly", "suspected",
            "may have", "might have", "denies",
            "denied", "no history of", "negative for",
            "without", "elevated", "abnormal"
        ]

        self.section_keywords = {
            "Chief Complaints": ["chief complaints", "symptoms"],
            "Medical History": ["medical history", "history"],
            "Clinical Examination": ["clinical examination", "examination"],
            "Investigation and Findings": ["investigation", "findings"],
            "Assessment": ["assessment", "diagnosis"],
            "Treatment and Plan": ["treatment", "plan"],
            "Follow-up": ["follow-up", "follow up"]
        }

    def calculate_sentence_score(self, sentence):

        sentence_lower = sentence.lower()
        score = 0

        for category, keywords in self.medical_categories.items():

            if any(keyword in sentence_lower for keyword in keywords):

                if category == "DISEASE":
                    score += 5
                elif category == "SYMPTOM":
                    score += 4
                elif category == "MEDICATION":
                    score += 4
                elif category == "INVESTIGATION":
                    score += 3
                elif category == "TREATMENT":
                    score += 2

        for phrase in self.important_phrases:

            if re.search(
                r'(?<!\w)' + re.escape(phrase) + r'(?!\w)',
                sentence_lower
            ):
                score += 3

        return score

    def get_section(self, sentence):

        sentence_lower = sentence.lower()

        for section, keywords in self.section_keywords.items():

            if any(
                re.search(r'\b' + re.escape(keyword) + r'\b',
                          sentence_lower)
                for keyword in keywords
            ):
                return section

        return None

    def summarize(self, text, max_sentences=12):

        # Remove Markdown formatting
        text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
        text = re.sub(r'\*\*', '', text)

        # Split into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+|\n+',
            text.strip()
        )

        sentence_scores = []
        seen = set()

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Ignore metadata
            if re.match(
                r'^(document id|department|document type|patient name|age|gender|visit type)\s*:',
                sentence,
                re.IGNORECASE
            ):
                continue

            # Ignore section headings
            if re.match(r'^\d+\.?\s+', sentence):
                continue

            if sentence.endswith(":") and len(sentence.split()) <= 6:
                continue

            sentence_key = sentence.lower()

            if sentence_key in seen:
                continue

            seen.add(sentence_key)

            score = self.calculate_sentence_score(sentence)

            section = self.get_section(sentence)

            sentence_scores.append({
                "sentence": sentence,
                "score": score,
                "section": section
            })

        # Select important sentences
        ranked = sorted(
            sentence_scores,
            key=lambda item: item["score"],
            reverse=True
        )

        selected = ranked[:max_sentences]

        # Restore original document order
        selected_sentences = {
            item["sentence"] for item in selected
        }

        summary_sentences = [
            item["sentence"]
            for item in sentence_scores
            if item["sentence"] in selected_sentences
        ]

        return {
            "summary": " ".join(summary_sentences),
            "sentence_scores": sentence_scores
        }


if __name__ == "__main__":

    sample_text = """
    Patient has diabetes and hypertension.
    Patient takes metformin for diabetes.
    Patient reports fever, persistent cough, headache and fatigue.
    Patient reports shortness of breath.
    Patient denies chest pain.
    No history of asthma was reported.
    Possible respiratory infection is suspected.
    Blood glucose levels were elevated.
    Blood tests were performed.
    Chest X-ray was suggested.
    Blood pressure was monitored.
    Continue the prescribed diabetes medication.
    Patient should return for follow-up after three days.
    """

    summarizer = ClinicalSummarizer()

    result = summarizer.summarize(sample_text)

    print("\n===== FINAL CLINICAL DOCUMENT SUMMARY =====")
    print(json.dumps(result, indent=4))
