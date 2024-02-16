import tensorflow_hub as hub

from video_pose import VideoPose

MODELS_MovenetLight = "movenet_lightning"
MODELS_MovenetThunder = "movenet_thunder"


def load_model(model_name):
    # if model_name == "":
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return module.signatures["serving_default"], input_size


model, model_input_size = load_model(MODELS_MovenetLight)

vp = VideoPose(model, model_input_size)

vp.infer()
