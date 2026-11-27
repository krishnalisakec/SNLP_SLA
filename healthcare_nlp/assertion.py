
import re
import json


class ClinicalAssertion:

    def __init__(self):

        # Negation indicators
        self.negation_words = [
            "no",
            "not",
            "denies",
            "deny",
            "denied",
            "without",
            "negative for",
            "absent",
            "never",
            "no history of",
            "no evidence of",
            "free of",
            "lack of",
            "lacks"
        ]

        # Uncertainty indicators
        self.uncertainty_words = [
            "possible",
            "possibly",
            "suspected",
            "suspect",
            "probable",
            "probably",
            "may have",
            "might have",
            "could have",
            "rule out",
            "suggestive of",
            "potential"
        ]

    # -----------------------------------
    # Find Entity Position
    # -----------------------------------

    def find_entity(self, text, entity):

        pattern = (
            r"(?<!\w)" + re.escape(entity) + r"(?!\w)"
        )

        return re.search(pattern, text, re.IGNORECASE)

    # -----------------------------------
    # Extract Entity Context
    # -----------------------------------

    def extract_context(self, text, position):

        # Find the beginning of the current sentence
        sentence_start = max(
            text.rfind(".", 0, position),
            text.rfind("!", 0, position),
            text.rfind("?", 0, position),
            text.rfind("\n", 0, position)
        ) + 1

        # Extract text before the entity in the same sentence
        preceding_text = text[sentence_start:position].lower()

        # Keep the last 10 words for assertion analysis
        preceding_words = preceding_text.split()

        context = " ".join(preceding_words[-10:])

        return context

    # -----------------------------------
    # Check Assertion Indicators
    # -----------------------------------

    def contains_indicator(self, context, indicators):

        for word in indicators:

            pattern = (
                r"(?<!\w)" + re.escape(word) + r"(?!\w)"
            )

            if re.search(pattern, context, re.IGNORECASE):
                return True

        return False

    # -----------------------------------
    # Main Assertion Detection
    # -----------------------------------

    def detect_assertion(self, text, entity):

        if not text or not entity:
            return "NOT_FOUND"

        match = self.find_entity(text, entity)

        if not match:
            return "NOT_FOUND"

        context = self.extract_context(
            text,
            match.start()
        )

        # Negation has priority over uncertainty
        if self.contains_indicator(
            context,
            self.negation_words
        ):
            return "NEGATED"

        # Check uncertainty
        if self.contains_indicator(
            context,
            self.uncertainty_words
        ):
            return "POSSIBLE"

        # No negation or uncertainty found
        return "PRESENT"


# -----------------------------------
# Standalone Testing
# -----------------------------------

if __name__ == "__main__":

    clinical_sentences = [

        ("Patient has diabetes mellitus.", "diabetes mellitus"),

        ("Patient denies fever.", "fever"),

        ("Patient may have pneumonia.", "pneumonia"),

        ("Patient has hypertension.", "hypertension"),

        ("Patient has no cough.", "cough"),

        ("Patient has possible asthma.", "asthma"),

        ("Patient is without chest pain.", "chest pain"),

        ("Patient reports headache.", "headache"),

        ("Patient denies chest pain.", "chest pain"),

        ("No history of asthma was reported.", "asthma"),

        ("Possible respiratory infection was suspected.",
         "respiratory infection"),

        ("Patient has diabetes. Patient denies fever.", "diabetes"),

        ("Patient has diabetes. Patient denies fever.", "fever")
    ]

    assertion = ClinicalAssertion()

    results = []

    correct_predictions = 0

    for sentence, entity in clinical_sentences:

        status = assertion.detect_assertion(
            sentence,
            entity
        )

        results.append({

            "sentence": sentence,
            "entity": entity,
            "assertion": status

        })

        # Expected results for testing
        expected = {

            "diabetes mellitus": "PRESENT",
            "fever": "NEGATED",
            "pneumonia": "POSSIBLE",
            "hypertension": "PRESENT",
            "cough": "NEGATED",
            "asthma": "POSSIBLE",
            "chest pain": "NEGATED",
            "headache": "PRESENT",
            "respiratory infection": "POSSIBLE",
            "diabetes": "PRESENT"

        }

        # The second asthma example is explicitly negated
        if sentence.startswith("No history"):
            expected_status = "NEGATED"
        elif sentence.startswith("Patient has possible"):
            expected_status = "POSSIBLE"
        else:
            expected_status = expected.get(entity, "NOT_FOUND")

        if status == expected_status:
            correct_predictions += 1

    print("\n===== CLINICAL ASSERTION DETECTION =====")

    print(json.dumps(results, indent=4))

    print("\n===== TEST SUMMARY =====")

    print("Total Test Cases:", len(clinical_sentences))

    print("Correct Predictions:", correct_predictions)

    print(
        "Incorrect Predictions:",
        len(clinical_sentences) - correct_predictions
    )

    print(
        "Test Accuracy:",
        round(
            correct_predictions / len(clinical_sentences) * 100,
            2
        ),
        "%"
    )
