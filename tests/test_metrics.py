import unittest

# Append directory to system path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.metrics import RecallAtK


class TestRecallAtK(unittest.TestCase):
    """Tests for the RecallAtK metric class focusing on its ability to accurately calculate recall at different recalls at k."""

    def test_calculate_single_k_single_groundtruth_all_k(self):
        """Test that recall is calculated correctly when all ground truth documents are retrieved within k."""
        metric = RecallAtK(k=3, compute_all_k=True)
        ground_truths = ["doc3"]
        retrievals = ["doc2", "doc1", "doc3"]
        expected = {"Recall@1": False, "Recall@2": False, "Recall@3": True}
        self.assertEqual(metric.calculate(ground_truths, retrievals), expected)

    def test_calculate_single_k_single_groundtruth_true(self):
        """Test that recall is calculated correctly when all ground truth documents are retrieved within k."""
        metric = RecallAtK(k=2)
        ground_truths = ["doc1"]
        retrievals = ["doc2", "doc1", "doc3"]
        expected = {"Recall@2": True}
        self.assertEqual(metric.calculate(ground_truths, retrievals), expected)

    def test_calculate_single_k_true(self):
        """Test that recall is calculated correctly when all ground truth documents are retrieved within k."""
        metric = RecallAtK(k=2)
        ground_truths = ["doc1", "doc2"]
        retrievals = ["doc2", "doc1", "doc3"]
        expected = {"Recall@2": True}
        self.assertEqual(metric.calculate(ground_truths, retrievals), expected)

    def test_calculate_single_k_false(self):
        """Test that recall correctly identifies when not all ground truth documents are retrieved within k."""
        metric = RecallAtK(k=2)
        ground_truths = ["doc1", "doc2"]
        retrievals = ["doc2", "doc3", "doc1"]
        expected = {"Recall@2": False}
        self.assertEqual(metric.calculate(ground_truths, retrievals), expected)

    def test_calculate_all_k_mixed_results(self):
        """Test recall at all values up to k, verifying that results vary based on recalls at k."""
        metric = RecallAtK(k=3, compute_all_k=True)
        ground_truths = ["doc1", "doc3"]
        retrievals = ["doc3", "doc2", "doc1", "doc4"]
        expected = {"Recall@1": False, "Recall@2": False, "Recall@3": True}
        self.assertEqual(metric.calculate(ground_truths, retrievals), expected)
