from __future__ import annotations

import unittest

from paper_v1.data.label_space import DEFAULT_LABEL_SPACE


class LabelMappingTest(unittest.TestCase):
    def test_negative_one_maps_to_zero_under_default_policy(self) -> None:
        row = {
            "label_atelectasis": "-1",
            "label_cardiomegaly": "1",
            "label_consolidation": "0",
            "label_edema": "",
            "label_pleural_effusion": "0",
            "label_pneumonia": "0",
            "label_pneumothorax": "0",
        }
        vector = DEFAULT_LABEL_SPACE.vector_from_row(row, negative_one_policy="zero")
        self.assertEqual(vector.tolist()[0], 0.0)
        self.assertEqual(vector.tolist()[1], 1.0)


if __name__ == "__main__":
    unittest.main()
