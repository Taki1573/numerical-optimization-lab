import json
import math
from pathlib import Path
import unittest

from benchmarks import rastrigin, rastrigin_grad, rosenbrock, rosenbrock_grad
from verify_stage1 import verify


class TestBenchmarks(unittest.TestCase):
    def test_rosenbrock_two_dimensional_reference(self):
        self.assertAlmostEqual(rosenbrock([-1.2, 1.0]), 24.2)
        g1, g2 = rosenbrock_grad([-1.2, 1.0])
        self.assertAlmostEqual(g1, -215.6)
        self.assertAlmostEqual(g2, -88.0)

    def test_rosenbrock_first_interior_last(self):
        self.assertEqual(rosenbrock([0.0] * 10), 9)
        self.assertEqual(rosenbrock_grad([0.0] * 10), [-2.0] * 9 + [0.0])

    def test_rastrigin_half_integer(self):
        self.assertAlmostEqual(rastrigin([0.5] * 10), 202.5)
        for g in rastrigin_grad([0.5] * 10):
            self.assertAlmostEqual(g, 1)

    def test_rastrigin_one_coordinate(self):
        self.assertAlmostEqual(rastrigin([-0.25]), 10.0625)
        self.assertAlmostEqual(rastrigin_grad([-0.25])[0], -0.5 - 20 * math.pi)

    def test_invalid_inputs(self):
        for fn in (rosenbrock, rosenbrock_grad, rastrigin, rastrigin_grad):
            for x in ([], [0.0, math.nan], [math.inf, 1]):
                with self.subTest(fn=fn.__name__, x=x), self.assertRaises(ValueError):
                    fn(x)
        for fn in (rosenbrock, rosenbrock_grad):
            with self.assertRaises(ValueError):
                fn([1.0])

    def test_input_unchanged(self):
        x = [0.3, 0.7]
        for fn in (rosenbrock, rosenbrock_grad, rastrigin, rastrigin_grad):
            fn(x)
            self.assertEqual(x, [0.3, 0.7])

    def test_config_and_all_directional_checks(self):
        config_file = Path(__file__).resolve().parents[1] / "configs" / "experiments.json"
        config = json.loads(config_file.read_text(encoding="utf-8"))
        self.assertGreaterEqual(config["dimension"], 10)
        for starts in config["initial_points"].values():
            for point in starts.values():
                self.assertEqual(len(point), config["dimension"])
        results = verify(config)
        self.assertEqual(results["summary"]["minima_passed"], 2)
        self.assertEqual(results["summary"]["directions_total"], 6 * (config["dimension"] + 2))
        self.assertTrue(results["summary"]["passed"], results["summary"])


if __name__ == "__main__":
    unittest.main()
