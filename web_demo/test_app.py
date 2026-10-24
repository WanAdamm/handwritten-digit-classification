import unittest

from app import predict


class PredictTests(unittest.TestCase):
    def test_prediction_is_a_digit(self):
        prediction = predict([0] * 784)
        self.assertIn(prediction, range(10))

    def test_rejects_invalid_pixels(self):
        for pixels in ([0] * 783, [0] * 783 + [256]):
            with self.subTest(length=len(pixels)):
                with self.assertRaises(ValueError):
                    predict(pixels)


if __name__ == "__main__":
    unittest.main()
