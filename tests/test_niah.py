"""
Unit tests for NIAH prompt generator and memory tracker utilities.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.niah_generator import NeedleInHaystackGenerator
from src.evaluation.memory_tracker import LatencyTimer, MemoryTracker


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # 1 word ~ 1 token for mock testing
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class TestNIAHGenerator(unittest.TestCase):
    def test_prompt_generation(self):
        generator = NeedleInHaystackGenerator()
        tokenizer = MockTokenizer()

        prompt, answer = generator.generate_prompt(
            tokenizer=tokenizer,
            target_token_length=200,
            depth_percent=0.5,
        )

        self.assertIn(generator.needle, prompt)
        self.assertIn(generator.query, prompt)
        self.assertEqual(answer, generator.expected_answer)

    def test_latency_timer(self):
        timer = LatencyTimer()
        timer.start()
        timer.mark_first_token()
        metrics = timer.stop(num_generated_tokens=10)

        self.assertIn("ttft_ms", metrics)
        self.assertIn("throughput_tokens_per_sec", metrics)
        self.assertEqual(metrics["num_generated_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
