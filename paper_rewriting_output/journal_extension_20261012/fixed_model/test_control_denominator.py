import unittest
import numpy as np
from add_control_denominator import joint_control_bootstrap


class ControlDenominatorTests(unittest.TestCase):
    def arrays(self):
        return {"sample_ids": np.arange(6), "entity": np.array(["a"] * 3 + ["b"] * 3),
                "start_time": np.array([0, 1, 2, 0, 1, 2]),
                "squared": np.ones((6, 2)), "counts": np.ones((6, 2))}

    def test_control_denominator_is_not_aligned_denominator(self):
        a = self.arrays()
        c = dict(a, squared=2 * a["squared"])
        ci = joint_control_bootstrap(a, {"double": c}, repetitions=100, block_length=2)
        np.testing.assert_allclose(ci["double"], [50.0, 50.0])

    def test_identical_pair_and_improvement(self):
        a = self.arrays()
        improved = dict(a, squared=0.5 * a["squared"])
        ci = joint_control_bootstrap(a, {"same": a, "improved": improved}, repetitions=100, block_length=4)
        np.testing.assert_allclose(ci["same"], [0.0, 0.0])
        np.testing.assert_allclose(ci["improved"], [-100.0, -100.0])

    def test_zero_control_loss_is_undefined(self):
        a = self.arrays()
        c = dict(a, squared=np.zeros((6, 2)))
        with self.assertRaises(ValueError):
            joint_control_bootstrap(a, {"zero": c}, repetitions=2, block_length=2)


if __name__ == "__main__":
    unittest.main()
