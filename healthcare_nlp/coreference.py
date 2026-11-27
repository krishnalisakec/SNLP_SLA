
import re
import json


class ClinicalCoreference:

    def __init__(self):

        self.patient_references = [
            "the patient",
            "he",
            "she",
            "they",
            "patient"
        ]

    def resolve(self, text):

        sentences = re.split(
            r'(?<=[.!?])\s+|\n+',
            text.strip()
        )

        resolved_sentences = []
        references = []

        for sentence in sentences:

            updated_sentence = sentence

            # Find longer expressions first to avoid
            # duplicate matches such as "the patient" and "patient".
            matches = []

            for reference in self.patient_references:

                pattern = (
                    r'(?<!\w)' +
                    re.escape(reference) +
                    r'(?!\w)'
                )

                for match in re.finditer(
                    pattern,
                    sentence,
                    re.IGNORECASE
                ):

                    matches.append({
                        "mention": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "referent": "Patient",
                        "sentence": sentence
                    })

            # Remove overlapping matches
            matches.sort(
                key=lambda item: (
                    item["start"],
                    -(item["end"] - item["start"])
                )
            )

            selected_matches = []
            last_end = -1

            for match in matches:

                if match["start"] >= last_end:

                    selected_matches.append(match)
                    last_end = match["end"]

            # Replace matches from right to left so that
            # character positions remain valid.
            for match in reversed(selected_matches):

                start = match["start"]
                end = match["end"]

                updated_sentence = (
                    updated_sentence[:start]
                    + "Patient"
                    + updated_sentence[end:]
                )

            for match in selected_matches:

                references.append({
                    "mention": match["mention"],
                    "referent": "Patient",
                    "sentence": sentence
                })

            resolved_sentences.append(updated_sentence)

        return {
            "original_text": text,
            "resolved_text": " ".join(resolved_sentences),
            "coreference_links": references
        }


if __name__ == "__main__":

    sample_text = (
        "Patient has diabetes. "
        "He takes metformin. "
        "The patient is under observation. "
        "She visits the hospital regularly."
    )

    coreference = ClinicalCoreference()

    result = coreference.resolve(sample_text)

    print("\n===== CLINICAL COREFERENCE RESOLUTION =====")

    print(json.dumps(result, indent=4))
