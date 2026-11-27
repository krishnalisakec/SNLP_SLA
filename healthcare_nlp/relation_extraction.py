
import re
import json

from healthcare_nlp.assertion import ClinicalAssertion


class ClinicalRelationExtractor:

    def __init__(self):

        self.assertion_detector = ClinicalAssertion()

        self.diseases = [
            "diabetes mellitus",
            "diabetes",
            "hypertension",
            "asthma",
            "pneumonia",
            "respiratory infection"
        ]

        self.symptoms = [
            "shortness of breath",
            "chest pain",
            "fever",
            "cough",
            "headache",
            "fatigue"
        ]

        self.medications = [
            "metformin",
            "insulin",
            "aspirin",
            "paracetamol"
        ]

        self.tests = [
            "blood pressure",
            "blood glucose",
            "ecg",
            "blood test",
            "blood tests",
            "chest x-ray",
            "x-ray"
        ]

    def find_entities(self, sentence, entity_list):

        found = []

        # Check longer entity names first
        for entity in sorted(
            entity_list,
            key=len,
            reverse=True
        ):

            pattern = (
                r"(?<!\w)" +
                re.escape(entity) +
                r"(?!\w)"
            )

            if re.search(pattern, sentence, re.IGNORECASE):

                # Avoid duplicate or overlapping entity matches
                overlaps = False

                for existing in found:
                    if (
                        entity.lower() in existing.lower()
                        or existing.lower() in entity.lower()
                    ):
                        overlaps = True
                        break

                if not overlaps:
                    found.append(entity)

        return found

    def get_assertion(self, sentence, entity):

        return self.assertion_detector.detect_assertion(
            sentence,
            entity
        )

    def extract_relations(self, text):

        relations = []

        sentences = re.split(
            r'(?<=[.!?])\s+|\n+',
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_lower = sentence.lower()

            diseases = self.find_entities(
                sentence,
                self.diseases
            )

            symptoms = self.find_entities(
                sentence,
                self.symptoms
            )

            medications = self.find_entities(
                sentence,
                self.medications
            )

            tests = self.find_entities(
                sentence,
                self.tests
            )

            # Disease relations with assertion status
            for disease in diseases:

                assertion = self.get_assertion(
                    sentence,
                    disease
                )

                if assertion == "PRESENT":
                    relation = "HAS_DIAGNOSIS"

                elif assertion == "NEGATED":
                    relation = "NEGATED_DIAGNOSIS"

                else:
                    relation = "POSSIBLE_DIAGNOSIS"

                relations.append({
                    "subject": "Patient",
                    "relation": relation,
                    "object": disease,
                    "assertion": assertion,
                    "sentence": sentence
                })

            # Symptom relations with assertion status
            for symptom in symptoms:

                assertion = self.get_assertion(
                    sentence,
                    symptom
                )

                if assertion == "PRESENT":
                    relation = "HAS_SYMPTOM"

                elif assertion == "NEGATED":
                    relation = "NEGATED_SYMPTOM"

                else:
                    relation = "POSSIBLE_SYMPTOM"

                relations.append({
                    "subject": "Patient",
                    "relation": relation,
                    "object": symptom,
                    "assertion": assertion,
                    "sentence": sentence
                })

            # Medication relations
            if re.search(
                r"\b(takes|taking|prescribed|given|started)\b",
                sentence_lower
            ):

                for medication in medications:

                    assertion = self.get_assertion(
                        sentence,
                        medication
                    )

                    if assertion == "PRESENT":
                        relation = "TAKES_MEDICATION"
                    elif assertion == "NEGATED":
                        relation = "NOT_TAKING_MEDICATION"
                    else:
                        relation = "POSSIBLY_TAKES_MEDICATION"

                    relations.append({
                        "subject": "Patient",
                        "relation": relation,
                        "object": medication,
                        "assertion": assertion,
                        "sentence": sentence
                    })

            # Diagnostic test relations
            if re.search(
                r"\b(performed|conducted|underwent|done|shows|monitored|ordered)\b",
                sentence_lower
            ):

                for test in tests:

                    relations.append({
                        "subject": "Patient",
                        "relation": "UNDERWENT_TEST",
                        "object": test,
                        "assertion": "PRESENT",
                        "sentence": sentence
                    })

        return relations


if __name__ == "__main__":

    sample_text = """
    Patient has diabetes and hypertension.
    Patient reports fever and cough.
    Patient denies chest pain.
    Patient may have pneumonia.
    Patient takes metformin.
    Blood pressure test was performed.
    """

    extractor = ClinicalRelationExtractor()

    result = extractor.extract_relations(sample_text)

    print("\n===== ASSERTION-AWARE CLINICAL RELATION EXTRACTION =====")

    print(json.dumps(result, indent=4))
