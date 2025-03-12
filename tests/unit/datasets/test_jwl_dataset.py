"""Unit tests for `datasets/jwl_dataset.py`."""

import unittest
from io import StringIO
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd
import yoke.datasets.jwl_dataset as JWL


# Define a sample CSV data string.
SAMPLE_CSV_DATA = (
    "a,b,c,w,r1,r2,V0,dcj,pcj,vcj,edet,e1,e2,e3,e4,e5,e6,e7,t0.1,t0.15,t0.25,t0.35,t0.5,"
    "t0.75,t1,t1.5,t2,t2.5,t3.5,t4.5\n"
    "1095.9397,46.632,2.050178,0.516286,5.224688,1.954136,0.544662,8.821064436467122,"
    "36.067109881699004,0.40715541171779507,5.349386344511012,0.7300730633948351,"
    "3.572946964437067,4.085814985537526,4.2868605086925005,4.4064077916057025,"
    "4.49169120427878,4.557388170680887,0.723111,0.728153,1.025569,1.204946,1.408454,"
    "1.694065,1.906253,2.182175,2.347675,2.471929,2.615156,2.684149\n"
    "1268.5589,77.8092,3.050357,0.617791,5.719151,2.376748,0.544662,8.77493146230528,"
    "35.403223777259875,0.40826353056205544,5.462162418213971,0.720678445281516,"
    "3.5546127545639985,4.083693743012503,4.318777011511218,4.467049180770517,"
    "4.573151872979772,4.6539178840834,0.709519,0.714558,1.011041,1.185748,1.389338,"
    "1.677436,1.892087,2.171521,2.337622,2.461379,2.607677,2.682875"
)


class TestJWL_Dataset(unittest.TestCase):
    """Testing JWL_CYLEX_pdv2jwl_Dataset using a mocked CSV."""

    def _loadDF(self) -> None:
        # Instead of loading from disk, we create the DataFrame from SAMPLE_CSV_DATA.
        self.data_df = pd.read_csv(StringIO(SAMPLE_CSV_DATA), sep=",", header=0)
        # For simplicity, select the first row for testing.
        self.itest = 0
        self.jwl = self.data_df.loc[self.itest, "a":"V0"].values
        self.CJ = self.data_df.loc[self.itest, "dcj":"edet"].values
        self.e = self.data_df.loc[self.itest, "e1":"e7"].values
        self.pdv = self.data_df.loc[self.itest, "t0.1":"t4.5"].values
        self.Nsamples = len(self.data_df)

    def setUp(self) -> None:
        """Set up common variables for the tests."""
        self.rng = slice(0, None)
        # Patch pandas.read_csv so that when the dataset attempts to load a
        # file, it receives our DataFrame.
        patcher = patch(
            "pandas.read_csv",
            return_value=pd.read_csv(StringIO(SAMPLE_CSV_DATA), sep=",", header=0),
        )
        self.addCleanup(patcher.stop)
        self.mock_read_csv = patcher.start()

        # The file argument is now a dummy value because read_csv is patched.
        self.file = "dummy.csv"

        # Initialize the model with the dummy file argument.
        self.model = JWL.CYLEX_pdv2jwl_Dataset(rng=self.rng, file=self.file)
        self._loadDF()

    def test_initialization(self) -> None:
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, torch.utils.data.Dataset)

    def test_resetCJ(self) -> None:
        """Test if model returns correct CJ values."""
        self.model.resetCJ()
        CJ = self.model.df.loc[self.itest, "dcj":"edet"].values
        np.testing.assert_allclose(CJ, self.CJ)

    def test_len(self) -> None:
        """Test if model returns the correct Dataset length."""
        self.assertEqual(len(self.model), self.Nsamples)

    def test_getitem(self) -> None:
        """Test if model returns correct indexed items."""
        input, output = self.model[self.itest]
        np.testing.assert_allclose(input, self.pdv)
        np.testing.assert_allclose(output, self.jwl[0:-1])

    def test_computeCJ(self) -> None:
        """Test if compute_CJ produces the correct CJ state."""
        CJ = JWL.compute_CJ(*self.jwl)
        # Render volume dimensional.
        CJ[2] *= self.jwl[-1]
        print("test_computeCJ:", self.CJ, CJ)
        np.testing.assert_allclose(self.CJ, CJ)

    def test_compute_e_release(self) -> None:
        """Test if compute_e_release produces the correct e release values."""
        vs = np.linspace(1, 7, 7)
        ecalc = JWL.compute_e_release(*self.jwl, self.CJ[-1], vs)
        np.testing.assert_allclose(ecalc, self.e)


class TestJWLnorm_Dataset(TestJWL_Dataset):
    """Normalized-testing for CYCLEXnorm_pdv2jwl_Dataset using a mocked CSV."""

    def setUp(self) -> None:
        """Set up common variables for the normalized dataset tests."""
        self.rng = slice(0, None)
        patcher = patch(
            "pandas.read_csv",
            return_value=pd.read_csv(StringIO(SAMPLE_CSV_DATA), sep=",", header=0),
        )
        self.addCleanup(patcher.stop)
        self.mock_read_csv = patcher.start()
        self.file = "dummy.csv"
        self.model = JWL.CYLEXnorm_pdv2jwl_Dataset(rng=self.rng, file=self.file)
        self._loadDF()

    def test_getitem(self) -> None:
        """Test if normalized dataset returns correct indexed items."""
        input, output = self.model[self.itest]
        np.testing.assert_allclose(
            input,
            (self.pdv - self.model.pdvmin) / (self.model.pdvmax - self.model.pdvmin),
        )
        np.testing.assert_allclose(
            output,
            (self.jwl[0:-1] - self.model.jwlmins)
            / (self.model.jwlmaxs - self.model.jwlmins),
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
