import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.settings = {
            "brightness": 0,
            "contrast": 1.0,
            "saturation": 1.0,
            "hue": 0,
            "white_balance": [1.0, 1.0, 1.0],
            "gamma": 1.0,
            "sharpness": 0,
            "color_temp": 0,
            "exposure": 0,
        }

    def apply_settings(self, image: np.ndarray, settings: dict = None) -> np.ndarray:
        # print(settings)
        if settings:
            self.settings.update(settings)
        img = image.astype(np.float32) / 255.0
        img += self.settings["exposure"]
        img = (
            (img - 0.5) * self.settings["contrast"]
            + 0.5
            + (self.settings["brightness"] / 255.0)
        )
        img = self._apply_color_balance(img)
        img = self._apply_hsv_adjustments(img)
        if self.settings["color_temp"] != 0:
            img = self._apply_color_temperature(img)
        img = np.clip(img, 0, 1)
        if self.settings["gamma"] != 1.0:
            img = self._apply_gamma(img)
        img_8u = (img * 255).astype(np.uint8)
        if self.settings["sharpness"] != 0:
            img_8u = self._apply_sharpness(img_8u)
        return img_8u

    def _apply_gamma(self, img: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        img = np.maximum(img, epsilon)
        return np.power(img, 1.0 / self.settings["gamma"])

    def _apply_color_balance(self, img: np.ndarray) -> np.ndarray:
        img[:, :, 0] *= self.settings["white_balance"][0]
        img[:, :, 1] *= self.settings["white_balance"][1]
        img[:, :, 2] *= self.settings["white_balance"][2]
        return img

    def _apply_hsv_adjustments(self, img: np.ndarray) -> np.ndarray:
        if self.settings["saturation"] != 1.0 or self.settings["hue"] != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] *= self.settings["saturation"]  # Насыщенность
            hsv[:, :, 0] = (hsv[:, :, 0] + self.settings["hue"]) % 360  # Оттенок
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def _apply_color_temperature(self, img: np.ndarray) -> np.ndarray:
        temperature = self.settings["color_temp"] / 100.0
        if temperature > 0:
            img[:, :, 0] *= 1.0 + temperature * 0.4
            img[:, :, 1] *= 1.0 + temperature * 0.1
        else:
            img[:, :, 2] *= 1.0 - temperature * 0.4
        return img

    def _apply_sharpness(self, img: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        strength = 1.0 + self.settings["sharpness"] / 100.0
        kernel = kernel * strength
        kernel[1, 1] = kernel[1, 1] - (8 * (strength - 1.0))
        return cv2.filter2D(img, -1, kernel)
