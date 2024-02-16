import tf_hub_models as th_models
from video_pose import VideoPose

MODELS_Movenet_Light = "movenet_lightning"
MODELS_Movenet_Light_Litle_8 = "movenet_lightning_int8.tflite"
MODELS_Movenet_Light_Litle_16 = "movenet_lightning_f16.tflite"
MODELS_Movenet_Thunder = "movenet_thunder"
MODELS_Movenet_Thunder_Litle_8 = "movenet_thunder_int8.tflite"
MODELS_Movenet_Thunder_Litle_16 = "movenet_thunder_f16.tflite"

infer_method, model_input_size = th_models.get_infer_and_input_size(MODELS_Movenet_Thunder_Litle_16)
vp = VideoPose(infer_method, model_input_size)

vp.infer()
