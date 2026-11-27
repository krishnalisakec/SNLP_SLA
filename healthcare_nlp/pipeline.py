
import json
import re

from healthcare_nlp.preprocessing import ClinicalPreprocessor
from healthcare_nlp.ner import ClinicalNER
from healthcare_nlp.assertion import ClinicalAssertion
from healthcare_nlp.coreference import ClinicalCoreference
from healthcare_nlp.relation_extraction import ClinicalRelationExtractor
from healthcare_nlp.salience import ClinicalSalience
from healthcare_nlp.summarization import ClinicalSummarizer


class HealthcareNLPPipeline:

    def __init__(self):
        self.preprocessor = ClinicalPreprocessor()
        self.ner = ClinicalNER()
        self.assertion = ClinicalAssertion()
        self.coreference = ClinicalCoreference()
        self.relation_extractor = ClinicalRelationExtractor()
        self.salience = ClinicalSalience()
        self.summarizer = ClinicalSummarizer()

    def get_entity_sentence(self, text, entity):

        start = entity.get("start")
        end = entity.get("end")

        if start is None or end is None:
            return text

        sentences = re.split(
            r'(?<=[.!?])\s+|\n+',
            text
        )

        current_position = 0

        for sentence in sentences:

            sentence_start = text.find(
                sentence,
                current_position
            )

            if sentence_start == -1:
                continue

            sentence_end = sentence_start + len(sentence)

            if sentence_start <= start < sentence_end:
                return sentence

            current_position = sentence_end

        return text

    def process_document(self, clinical_text):

        # Tier 1: Clinical Preprocessing
        preprocessing_result = self.preprocessor.process(
            clinical_text
        )

        processed_text = preprocessing_result["expanded_text"]

        # Tier 2: Hybrid Clinical NER
        entities = self.ner.extract_entities(processed_text)

        # Tier 2: Clinical Assertion Detection
        assertion_results = []

        for entity in entities:

            entity_sentence = self.get_entity_sentence(
                processed_text,
                entity
            )

            status = self.assertion.detect_assertion(
                entity_sentence,
                entity["entity"]
            )

            assertion_results.append({
                "sentence": entity_sentence,
                "entity": entity["entity"],
                "type": entity["type"],
                "assertion": status
            })

        # Tier 3: Clinical Coreference Resolution
        coreference_result = self.coreference.resolve(
            processed_text
        )

        # Tier 3: Assertion-Aware Relation Extraction
        relations = self.relation_extractor.extract_relations(
            processed_text
        )

        # Tier 4: Clinical Salience Scoring
        salience_results = self.salience.calculate_score(
            entities
        )

        # Tier 4: Clinical Summarization
        summary_result = self.summarizer.summarize(
            processed_text
        )

        return {
            "preprocessing": preprocessing_result,
            "entities": entities,
            "assertions": assertion_results,
            "coreference": coreference_result,
            "relations": relations,
            "salience": salience_results,
            "clinical_summary": summary_result
        }


if __name__ == "__main__":

    sample_note = """
    Patient has diabetes and hypertension.
    He takes metformin.
    Patient reports fever and cough.
    She denies chest pain.
    Blood pressure was monitored.
    """

    pipeline = HealthcareNLPPipeline()

    result = pipeline.process_document(sample_note)

    print("\n===== COMPLETE HEALTHCARE NLP PIPELINE =====")
    print(json.dumps(result, indent=4))
