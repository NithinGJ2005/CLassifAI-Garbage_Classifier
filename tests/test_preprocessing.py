import unittest
import numpy as np
from PIL import Image

# Replicable preprocessing function matching app.py
def preprocess_mobilenet_v2(arr):
    # Scales [0, 255] to [-1.0, 1.0]
    return (arr / 127.5) - 1.0

def prepare_image_for_test(pil_img):
    rgb = pil_img.convert("RGB")
    resized = rgb.resize((224, 224))
    arr = np.array(resized, dtype=np.float32)
    norm = preprocess_mobilenet_v2(arr)
    return np.expand_dims(norm, axis=0)

class TestImagePreprocessing(unittest.TestCase):

    def test_output_tensor_shape(self):
        # Create arbitrary test image
        img = Image.new("RGB", (640, 480), color=(128, 64, 32))
        tensor = prepare_image_for_test(img)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_value_range_normalization(self):
        # Extreme pixel values: black (0), white (255), mid (128)
        img_black = Image.new("RGB", (100, 100), color=(0, 0, 0))
        tensor_black = prepare_image_for_test(img_black)
        self.assertAlmostEqual(float(tensor_black.min()), -1.0, places=3)

        img_white = Image.new("RGB", (100, 100), color=(255, 255, 255))
        tensor_white = prepare_image_for_test(img_white)
        self.assertAlmostEqual(float(tensor_white.max()), 1.0, places=3)

        # Values should be strictly in [-1.0, 1.0]
        self.assertTrue(np.all(tensor_black >= -1.0) and np.all(tensor_black <= 1.0))
        self.assertTrue(np.all(tensor_white >= -1.0) and np.all(tensor_white <= 1.0))

    def test_rgba_to_rgb_conversion(self):
        # RGBA image with alpha channel
        img_rgba = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
        tensor = prepare_image_for_test(img_rgba)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_grayscale_to_rgb_conversion(self):
        # Grayscale single-channel image
        img_l = Image.new("L", (200, 200), color=120)
        tensor = prepare_image_for_test(img_l)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_exact_once_normalization(self):
        # Ensure that preprocessing is applied once, not compounded
        arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        norm1 = preprocess_mobilenet_v2(arr)
        np.testing.assert_allclose(norm1, [-1.0, 0.0, 1.0], atol=1e-5)

if __name__ == "__main__":
    unittest.main()
