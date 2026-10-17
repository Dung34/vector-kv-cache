"""
Needle-In-A-Haystack (NIAH) Synthetic Prompt Generator.
Constructs long-context evaluation prompts by embedding a precise 'needle' statement
into a large 'haystack' corpus at controlled depth intervals (0% to 100%).
"""

from typing import Tuple, Optional, Any


DEFAULT_NEEDLE = "The secret passkey to access the encrypted research archive is 739284."
DEFAULT_QUERY = "What is the secret passkey to access the encrypted research archive? Give only the passkey."
DEFAULT_ANSWER = "739284"

# Neutral background essay paragraph (repetitive background context)
DEFAULT_HAYSTACK_TEXT = (
    "In the development of large-scale computational systems, efficiency and scalability are paramount. "
    "Modern language models utilize multi-head self-attention mechanisms to capture complex contextual representations. "
    "As sequences scale to thousands of tokens, intermediate representations such as keys and values must be managed effectively. "
    "Careful caching strategies and algorithmic optimizations ensure stable inference throughput across diverse hardware configurations. "
    "Researchers continue to explore new frontiers in modular compression, tensor decomposition, and distributed serving. "
)


class NeedleInHaystackGenerator:
    """Generates synthetic long-context prompts with a needle placed at a specific depth percentage."""

    def __init__(
        self,
        needle: str = DEFAULT_NEEDLE,
        query: str = DEFAULT_QUERY,
        expected_answer: str = DEFAULT_ANSWER,
        haystack_text: str = DEFAULT_HAYSTACK_TEXT,
    ):
        self.needle = needle
        self.query = query
        self.expected_answer = expected_answer
        self.haystack_text = haystack_text

    def generate_prompt(
        self,
        tokenizer: Any,
        target_token_length: int,
        depth_percent: float, # from 0.0 (top) to 1.0 (bottom)
    ) -> Tuple[str, str]:
        """
        Generates prompt with approx target_token_length tokens, placing the needle at depth_percent.
        Returns: (full_prompt, expected_answer)
        """
        assert 0.0 <= depth_percent <= 1.0, "depth_percent must be between 0.0 and 1.0"

        needle_tokens = tokenizer.encode(self.needle, add_special_tokens=False)
        query_prompt = f"\n\nBased on the text above, answer the following question:\n{self.query}\nAnswer:"
        query_tokens = tokenizer.encode(query_prompt, add_special_tokens=False)

        available_tokens = target_token_length - len(needle_tokens) - len(query_tokens) - 20
        if available_tokens <= 0:
            available_tokens = 50

        # Calculate tokens before and after the needle
        tokens_before_needle = int(available_tokens * depth_percent)
        tokens_after_needle = available_tokens - tokens_before_needle

        # Generate haystack text before needle
        text_before = self._repeat_text_to_token_count(tokenizer, tokens_before_needle)
        text_after = self._repeat_text_to_token_count(tokenizer, tokens_after_needle)

        full_prompt = (
            f"{text_before}\n\n"
            f"{self.needle}\n\n"
            f"{text_after}"
            f"{query_prompt}"
        )

        return full_prompt, self.expected_answer

    def _repeat_text_to_token_count(self, tokenizer: Any, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        repeated = ""
        current_tokens = 0
        while current_tokens < target_tokens:
            repeated += self.haystack_text + " "
            current_tokens = len(tokenizer.encode(repeated, add_special_tokens=False))
        # Trim back down
        token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True)
