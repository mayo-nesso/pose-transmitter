from inference_hub.inference_models import Models, _get_infer_method, _get_input_size
from inference_hub.inference_utils import (
    determine_crop_region,
    init_crop_region,
    keypoints_and_edges_for_display,
    run_inference,
)


def init_lightning():
    return Hub(Models.Movenet_Lightning)


def init_lightning_lite8():
    return Hub(Models.Movenet_Lightning_Litle_8)


def init_lightning_lite16():
    return Hub(Models.Movenet_Lightning_Litle_16)


def init_thunder():
    return Hub(Models.Movenet_Thunder)


def init_thunder_lite8():
    return Hub(Models.Movenet_Thunder_Litle_8)


def init_thunder_lite16():
    return Hub(Models.Movenet_Thunder_Litle_16)


class Hub:
    def __init__(self, model_name):
        self.infer_method = _get_infer_method(model_name)
        self.model_input_size = _get_input_size(model_name)
        self.crop_region = None

    def run_inference(self, image, image_height, image_width):
        """
        Run inference on the input image.

        Args:
            image (numpy.ndarray): Input image.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.

        Returns:
            numpy.ndarray: Keypoints with scores.
        """
        if self.crop_region is None:
            self.crop_region = init_crop_region(image_height, image_width)

        keypoints_with_scores = run_inference(
            self.infer_method, image, self.crop_region, crop_size=[self.model_input_size, self.model_input_size]
        )

        self.crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

        return keypoints_with_scores

    def keypoints_and_edges_for_display(self, keypoints_with_scores, image_height, image_width):
        """
        Get keypoints and edges for display.

        Args:
            keypoints_with_scores (numpy.ndarray): Keypoints with scores.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.

        Returns:
            tuple: Tuple containing keypoints and edges.
        """
        return keypoints_and_edges_for_display(keypoints_with_scores, image_height, image_width)
