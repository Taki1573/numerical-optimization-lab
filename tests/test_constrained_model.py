import unittest

from constrained_model import BOUNDS, equality, inequalities, objective

class TestConstrainedModel(unittest.TestCase):
    def test_objective_at_feasible_point(self):
        self.assertAlmostEqual(objective((0.5, 0)), 6.5)

    def test_constraints_at_feasible_point(self):
        x = (0.5, 0)
        self.assertEqual(inequalities(x), (0.5, 0.75, 0.75))
        self.assertAlmostEqual(equality(x), 0.0)

    def test_constraints_at_vilotion_point(self):
        x = (0.25, 0.5)
        self.assertEqual(inequalities(x), (-0.25, 0.4375, 1.4375))

    def test_sign_equality(self):
        x = (0.5, 0.1)
        self.assertAlmostEqual(equality(x), +0.1)

    def test_BOUNDS(self):
        self.assertEqual(BOUNDS, ((0.0, 1.0), (-0.5, 2.0)))