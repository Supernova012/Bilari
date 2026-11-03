import unittest

try:
    import torch
    from chess_intuition.model import ChessResNet
except ImportError:  # base rules-only installation can still run engine tests
    torch = None
    ChessResNet = None


@unittest.skipIf(torch is None, "PyTorch training extra is not installed")
class ModelTests(unittest.TestCase):
    def test_policy_and_value_heads_have_expected_shapes(self):
        model = ChessResNet(blocks=2, channels=16).eval()
        with torch.inference_mode():
            policy, value = model(torch.zeros(3, 19, 8, 8))
        self.assertEqual(tuple(policy.shape), (3, 4672))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertTrue(bool(torch.isfinite(policy).all()))
        self.assertTrue(bool(torch.isfinite(value).all()))


if __name__ == "__main__":
    unittest.main()

