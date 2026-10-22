import unittest
import torch
import torch.nn as nn
from src.utils.mask_patch import patch_hf_eager_attention_for_pruned_cache

class TestMaskPatch(unittest.TestCase):
    def test_eager_attention_accepts_aligned_mask(self):
        patch_hf_eager_attention_for_pruned_cache()
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
            fn = qwen2_mod.eager_attention_forward
            self.assertTrue(getattr(fn, "_is_pruned_cache_patched", False))

            class MockModule:
                num_key_value_groups = 1
                training = False
            mod = MockModule()
            query = torch.randn(1, 4, 1, 32)
            # Pruned key has 50 tokens
            key = torch.randn(1, 4, 50, 32)
            value = torch.randn(1, 4, 50, 32)
            attention_mask = torch.zeros(1, 1, 1, 50)

            out, weights = fn(mod, query, key, value, attention_mask, scaling=1.0)
            self.assertEqual(weights.shape, (1, 4, 1, 50))
            self.assertEqual(out.shape, (1, 1, 4, 32))
        except ImportError:
            pass

    def test_eager_attention_rejects_misaligned_mask(self):
        patch_hf_eager_attention_for_pruned_cache()
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
            fn = qwen2_mod.eager_attention_forward

            class MockModule:
                num_key_value_groups = 1
                training = False

            mod = MockModule()
            query = torch.randn(1, 4, 1, 32)
            key = torch.randn(1, 4, 50, 32)
            value = torch.randn(1, 4, 50, 32)
            misaligned_mask = torch.zeros(1, 1, 1, 100)

            with self.assertRaisesRegex(AssertionError, "does not match key length"):
                fn(mod, query, key, value, misaligned_mask, scaling=1.0)
        except ImportError:
            pass

if __name__ == "__main__":
    unittest.main()
