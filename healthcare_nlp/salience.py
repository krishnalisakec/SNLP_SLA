
import json


class ClinicalSalience:

    def __init__(self):

        self.importance_weights = {
            "DISEASE": 5,
            "SYMPTOM": 4,
            "MEDICATION": 4,
            "TEST": 3
        }

    def calculate_score(self, entities):

        scored_entities = []

        for entity in entities:

            entity_type = entity.get("type", "OTHER")

            score = self.importance_weights.get(
                entity_type, 1
            )

            scored_entities.append({
                "entity": entity.get("entity"),
                "type": entity_type,
                "salience_score": score
            })

        scored_entities.sort(
            key=lambda item: item["salience_score"],
            reverse=True
        )

        return scored_entities


if __name__ == "__main__":

    sample_entities = [
        {"entity": "diabetes", "type": "DISEASE"},
        {"entity": "fever", "type": "SYMPTOM"},
        {"entity": "metformin", "type": "MEDICATION"},
        {"entity": "blood pressure", "type": "TEST"},
        {"entity": "hypertension", "type": "DISEASE"}
    ]

    scorer = ClinicalSalience()

    results = scorer.calculate_score(sample_entities)

    print("\n===== CLINICAL SALIENCE SCORING =====")
    print(json.dumps(results, indent=4))
