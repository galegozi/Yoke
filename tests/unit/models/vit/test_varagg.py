"""Test for `models/vit/variable_aggregation.py`."""

import unittest
import torch

from yoke.models.vit.aggregate_variables import AggVars


class Test_AggVars(unittest.TestCase):
    """A class to test AggVars."""

    def setUp(self) -> None:
        """Set up common variables for the tests."""
        self.embed_dim = 32
        self.num_heads = 4
        self.model = AggVars(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def test_initialization(self) -> None:
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model.var_query, torch.nn.Parameter)
        self.assertEqual(self.model.var_query.shape, (1, 1, self.embed_dim))
        self.assertIsInstance(self.model.var_agg, torch.nn.MultiheadAttention)

    def test_forward_shape(self) -> None:
        """Test if the forward method returns the expected shape."""
        # Input tensor of shape (B, NumVars, NumTokens, embed_dim)
        B, V, L, D = 3, 15, 128, self.embed_dim
        x = torch.rand(B, V, L, D).type(torch.FloatTensor).to(self.device)

        output = self.model(x)
        expected_output_shape = (B, L, D)

        self.assertEqual(output.shape, expected_output_shape)

    def test_forward_values(self) -> None:
        """Test if the forward method runs without errors and the output is a tensor."""
        B, V, L, D = 3, 15, 128, self.embed_dim
        x = torch.rand(B, V, L, D).type(torch.FloatTensor).to(self.device)

        output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
