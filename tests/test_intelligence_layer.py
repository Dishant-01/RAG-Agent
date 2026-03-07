import json
import unittest

from intelligence_layer import IntelligenceLayer


class FakeVectorStore:
    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self, include=None):
        docs = [""] * len(self._metadatas)
        return {"documents": docs, "metadatas": self._metadatas}


def _row_meta(machine, job, fields):
    return {
        "doc_kind": "categorical_row",
        "source": "sample.pdf",
        "row_machine": machine,
        "row_job": job,
        "row_job_type": fields.get("operation_type", ""),
        "row_payload": f"Machine: {machine} | Job: {job}",
        "row_fields_json": json.dumps(fields),
    }


class IntelligenceLayerTests(unittest.TestCase):
    def test_generates_analytics_context(self):
        store = FakeVectorStore(
            [
                _row_meta("M01", "J001", {"energy_consumption": "10.0"}),
                _row_meta("M01", "J002", {"energy_consumption": "11.0"}),
                _row_meta("M02", "J003", {"energy_consumption": "12.0"}),
            ]
        )
        layer = IntelligenceLayer(store)
        answer = layer.answer("Give me production summary")
        self.assertIsNotNone(answer)
        self.assertIn("ANALYTICS_CONTEXT_JSON", answer[0])
        self.assertIn("\"distinct_jobs\": 3", answer[0])
        self.assertIn("\"distinct_machines\": 2", answer[0])

    def test_range_scope_is_applied(self):
        store = FakeVectorStore(
            [
                _row_meta("M01", "J001", {"energy_consumption": "10.0"}),
                _row_meta("M01", "J002", {"energy_consumption": "20.0"}),
                _row_meta("M01", "J003", {"energy_consumption": "30.0"}),
            ]
        )
        layer = IntelligenceLayer(store)
        answer = layer.answer("What is the average energy consumption of J001 to J003?")
        self.assertIsNotNone(answer)
        self.assertIn("\"job_range\": [", answer[0])
        self.assertIn("\"rows\": 3", answer[0])
        self.assertIn("\"sum\": 60.0", answer[0])

    def test_ignores_non_machine_codes(self):
        store = FakeVectorStore(
            [
                _row_meta("80", "J001", {"energy_consumption": "10.0"}),
                _row_meta("M01", "J002", {"energy_consumption": "20.0"}),
            ]
        )
        layer = IntelligenceLayer(store)
        counts = layer.machine_row_counts()
        self.assertEqual(counts, {"M01": 1})


if __name__ == "__main__":
    unittest.main()
