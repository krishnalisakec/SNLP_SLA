
import re
from transformers import pipeline


class ClinicalNER:

    def __init__(self):

        print("Loading Pretrained Biomedical NER Model...")

        self.model_name = "d4data/biomedical-ner-all"

        self.ner_pipeline = pipeline(
            task="token-classification",
            model=self.model_name,
            aggregation_strategy="simple",
            device=-1
        )

        print("Biomedical NER Model Loaded Successfully!")

        self.clinical_vocabulary = {

            "DISEASE": [
                "diabetes mellitus",
                "diabetes",
                "hypertension",
                "respiratory infection",
                "asthma"
            ],

            "SYMPTOM": [
                "shortness of breath",
                "persistent cough",
                "chest pain",
                "headache",
                "fatigue",
                "fever",
                "cough"
            ],

            "MEDICATION": [
                "metformin"
            ],

            "TEST": [
                "blood tests",
                "blood glucose",
                "blood pressure",
                "chest x-ray",
                "heart rate"
            ]
        }

    # -----------------------------------
    # Model Label Mapping
    # -----------------------------------

    def map_entity_type(self, model_label):

        label = model_label.replace("B-", "").replace("I-", "")
        label = label.lower()

        entity_mapping = {

            "disease_disorder": "DISEASE",
            "sign_symptom": "SYMPTOM",
            "medication": "MEDICATION",
            "diagnostic_procedure": "TEST",
            "therapeutic_procedure": "TEST",
            "lab_value": "TEST"

        }

        return entity_mapping.get(label, label.upper())

    # -----------------------------------
    # Biomedical Model Extraction
    # -----------------------------------

    def extract_model_entities(self, text):

        entities = []

        if not text or not text.strip():
            return entities

        sentences = list(
            re.finditer(r'[^.!?\n]+[.!?]?', text)
        )

        for sentence_match in sentences:

            original_sentence = sentence_match.group()
            sentence = original_sentence.strip()

            if not sentence:
                continue

            sentence_start = (
                sentence_match.start()
                + len(original_sentence)
                - len(original_sentence.lstrip())
            )

            try:

                predictions = self.ner_pipeline(sentence)

                for item in predictions:

                    entity_text = item["word"].strip()

                    if not entity_text:
                        continue

                    entity_type = self.map_entity_type(
                        item["entity_group"]
                    )

                    if entity_type not in [
                        "DISEASE",
                        "SYMPTOM",
                        "MEDICATION",
                        "TEST"
                    ]:
                        continue

                    if entity_text.lower() == "met":
                        continue

                    start = sentence_start + item["start"]
                    end = sentence_start + item["end"]

                    entities.append({

                        "entity": entity_text,
                        "type": entity_type,
                        "start": start,
                        "end": end,
                        "score": round(float(item["score"]), 4),
                        "source": "Biomedical Model"

                    })

            except Exception as error:

                print("Biomedical NER Error:", error)

        return entities

    # -----------------------------------
    # Clinical Vocabulary Extraction
    # -----------------------------------

    def extract_vocabulary_entities(self, text):

        entities = []

        if not text:
            return entities

        for entity_type, terms in self.clinical_vocabulary.items():

            for term in terms:

                pattern = (
                    r'(?<!\w)' + re.escape(term) + r'(?!\w)'
                )

                for match in re.finditer(
                    pattern,
                    text,
                    re.IGNORECASE
                ):

                    entities.append({

                        "entity": match.group(),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "score": None,
                        "source": "Clinical Vocabulary"

                    })

        return entities

    # -----------------------------------
    # Hybrid Entity Combination
    # -----------------------------------

    def extract_entities(self, text):

        if not text or not text.strip():
            return []

        print("\nRunning Biomedical Model...")

        model_entities = self.extract_model_entities(text)

        print(
            "Biomedical Model Entities:",
            len(model_entities)
        )

        print("\nRunning Clinical Vocabulary Matching...")

        vocabulary_entities = self.extract_vocabulary_entities(text)

        print(
            "Clinical Vocabulary Entities:",
            len(vocabulary_entities)
        )

        combined_entities = []

        # Combine exact matching predictions
        for vocab in vocabulary_entities:

            matching_model = next(
                (
                    model for model in model_entities
                    if model["start"] == vocab["start"]
                    and model["end"] == vocab["end"]
                    and model["type"] == vocab["type"]
                ),
                None
            )

            if matching_model:

                combined_entities.append({

                    "entity": vocab["entity"],
                    "type": vocab["type"],
                    "start": vocab["start"],
                    "end": vocab["end"],
                    "score": matching_model["score"],
                    "source": "Biomedical Model + Clinical Vocabulary"

                })

            else:

                combined_entities.append(vocab.copy())

        # Add model predictions not already covered by vocabulary
        for model in model_entities:

            matching = any(
                model["start"] == vocab["start"]
                and model["end"] == vocab["end"]
                and model["type"] == vocab["type"]
                for vocab in vocabulary_entities
            )

            if not matching:
                combined_entities.append(model.copy())

        # -----------------------------------
        # Remove Duplicate and Overlapping Entities
        # Prefer longer phrases
        # -----------------------------------

        combined_entities.sort(
            key=lambda item: (
                -(item["end"] - item["start"]),
                item["start"]
            )
        )

        final_entities = []

        for candidate in combined_entities:

            overlaps = any(
                candidate["start"] < existing["end"]
                and candidate["end"] > existing["start"]
                for existing in final_entities
            )

            if not overlaps:
                final_entities.append(candidate)

        # Restore original text order
        final_entities.sort(
            key=lambda item: item["start"]
        )

        return final_entities

    # -----------------------------------
    # Display Results
    # -----------------------------------

    def display_results(self, entities):

        print("\nHYBRID CLINICAL NER RESULTS")
        print("=" * 80)

        for item in entities:
            print(item)

        print("=" * 80)

        print("Total Entities Detected:", len(entities))

        model_count = sum(
            1 for item in entities
            if item["source"] == "Biomedical Model"
        )

        vocabulary_count = sum(
            1 for item in entities
            if item["source"] == "Clinical Vocabulary"
        )

        combined_count = sum(
            1 for item in entities
            if item["source"] ==
            "Biomedical Model + Clinical Vocabulary"
        )

        print("Biomedical Model Only:", model_count)
        print("Clinical Vocabulary Only:", vocabulary_count)
        print("Detected by Both Methods:", combined_count)


# -----------------------------------
# Standalone Testing
# -----------------------------------

if __name__ == "__main__":

    ner = ClinicalNER()

    sample_text = """
    The patient has diabetes mellitus and hypertension.
    The patient reports fever, persistent cough, headache and fatigue.
    The patient is taking metformin.
    Blood tests were performed.
    Blood pressure was monitored.
    Chest X-ray was suggested.
    The patient denies chest pain.
    No history of asthma was reported.
    """

    results = ner.extract_entities(sample_text)

    ner.display_results(results)
