"""
Needle-In-A-Haystack (NIAH) Synthetic Prompt Generator.
Constructs single-needle and multi-needle long-context evaluation prompts.
Multi-Needle benchmark tests the model's ability to recall multiple facts scattered
across different depths (e.g., 3-5 needles), exposing the binary eviction flaw of H2O.
"""

from typing import Tuple, Optional, Any, List, Dict


DEFAULT_NEEDLE = "The secret passkey to access the encrypted research archive is 739284."
DEFAULT_QUERY = "What is the secret passkey to access the encrypted research archive? Give only the passkey."
DEFAULT_ANSWER = "739284"

# 5 distinct needles for Multi-Needle evaluation
DEFAULT_MULTI_NEEDLES = [
    {"key": "Alpha vault", "passkey": "739284", "sentence": "The secret passkey for Alpha vault is 739284."},
    {"key": "Beta vault", "passkey": "918273", "sentence": "The secret passkey for Beta vault is 918273."},
    {"key": "Gamma vault", "passkey": "582194", "sentence": "The secret passkey for Gamma vault is 582194."},
    {"key": "Delta vault", "passkey": "341908", "sentence": "The secret passkey for Delta vault is 341908."},
    {"key": "Epsilon vault", "passkey": "627415", "sentence": "The secret passkey for Epsilon vault is 627415."},
]

DEFAULT_HAYSTACK_TEXT = (
    "In the development of large-scale computational systems, efficiency and scalability are paramount. "
    "Modern language models utilize multi-head self-attention mechanisms to capture complex contextual representations. "
    "As sequences scale to thousands of tokens, intermediate representations such as keys and values must be managed effectively. "
    "Careful caching strategies and algorithmic optimizations ensure stable inference throughput across diverse hardware configurations. "
    "Researchers continue to explore new frontiers in modular compression, tensor decomposition, and distributed serving. "
)


class NeedleInHaystackGenerator:
    """Generates synthetic long-context prompts with a single needle placed at a specific depth percentage."""

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
        assert 0.0 <= depth_percent <= 1.0, "depth_percent must be between 0.0 and 1.0"

        needle_tokens = tokenizer.encode(self.needle, add_special_tokens=False)
        query_prompt = f"\n\nBased on the text above, answer the following question:\n{self.query}\nAnswer:"
        query_tokens = tokenizer.encode(query_prompt, add_special_tokens=False)

        available_tokens = target_token_length - len(needle_tokens) - len(query_tokens) - 20
        if available_tokens <= 0:
            available_tokens = 50

        tokens_before_needle = int(available_tokens * depth_percent)
        tokens_after_needle = available_tokens - tokens_before_needle

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
        token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True)


class MultiNeedleGenerator:
    """
    Generates synthetic long-context prompts with multiple needles (3 to 5 keys)
    distributed across distinct depths of the haystack.
    Evaluates multi-point retrieval accuracy and recall across the entire context.
    """

    def __init__(
        self,
        needles: Optional[List[Dict[str, str]]] = None,
        haystack_text: str = DEFAULT_HAYSTACK_TEXT,
    ):
        self.needles = needles or DEFAULT_MULTI_NEEDLES
        self.haystack_text = haystack_text

    def generate_prompt(
        self,
        tokenizer: Any,
        target_token_length: int,
        num_needles: int = 3,
        custom_depths: Optional[List[float]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Embeds num_needles evenly (or at custom_depths) across target_token_length.
        Returns: (full_prompt, selected_needles)
        """
        num_needles = min(num_needles, len(self.needles))
        selected_needles = self.needles[:num_needles]

        # Calculate insertion depths
        if custom_depths is not None:
            assert len(custom_depths) == num_needles, "Length of custom_depths must match num_needles"
            depths = sorted(custom_depths)
        else:
            # e.g., for 3 needles: [0.25, 0.50, 0.75]
            step = 1.0 / (num_needles + 1)
            depths = [round(step * (i + 1), 2) for i in range(num_needles)]

        # Prepare question query asking for all needles
        vault_names = ", ".join([n["key"] for n in selected_needles])
        query_prompt = (
            f"\n\nBased on the text above, what are the secret passkeys for {vault_names}?\n"
            f"List each vault and its corresponding passkey:\nAnswer:"
        )

        all_needle_text = " ".join([n["sentence"] for n in selected_needles])
        needle_tokens_len = len(tokenizer.encode(all_needle_text, add_special_tokens=False))
        query_tokens_len = len(tokenizer.encode(query_prompt, add_special_tokens=False))

        available_tokens = max(100, target_token_length - needle_tokens_len - query_tokens_len - 30)

        # Distribute available tokens across N+1 segments
        # Depths represent positions in the overall available tokens
        # [0.0, d1, d2, ..., dn, 1.0]
        boundaries = [0.0] + depths + [1.0]
        segment_lengths = [
            int(available_tokens * (boundaries[i + 1] - boundaries[i]))
            for i in range(len(boundaries) - 1)
        ]

        # Build composite document
        doc_parts = []
        for i in range(num_needles):
            doc_parts.append(self._repeat_text_to_token_count(tokenizer, segment_lengths[i]))
            doc_parts.append(f"\n\n{selected_needles[i]['sentence']}\n\n")

        # Final segment after last needle
        doc_parts.append(self._repeat_text_to_token_count(tokenizer, segment_lengths[-1]))
        doc_parts.append(query_prompt)

        full_prompt = "".join(doc_parts)
        return full_prompt, selected_needles

    def _repeat_text_to_token_count(self, tokenizer: Any, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        repeated = ""
        current_tokens = 0
        while current_tokens < target_tokens:
            repeated += self.haystack_text + " "
            current_tokens = len(tokenizer.encode(repeated, add_special_tokens=False))
        token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True)
