# Copyright 2021 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import requests
import tensorflow as tf
import tensorflow_hub as hub


def _get_file(file_name, url):
    if os.path.exists(file_name):
        return

    print(f"Downloading {file_name}...")
    response = requests.get(url)
    if response.status_code == 200:
        # Escribir el contenido de la respuesta en el archivo
        with open(file_name, "wb") as file:
            file.write(response.content)
    else:
        print(f"Error: {response.status_code} - downloading file: {file_name}")


def _get_infer_method(model_name):
    if "tflite" in model_name:
        if "movenet_lightning_f16" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
        elif "movenet_thunder_f16" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
        elif "movenet_lightning_int8" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
        elif "movenet_thunder_int8" in model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        model_path = f"{model_name}.tflite"
        _get_file(model_path, url)

        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
            input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
            """
            # TF Lite format expects tensor type of uint8.
            input_image = tf.cast(input_image, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
            return keypoints_with_scores

        return movenet

    else:
        if "movenet_lightning" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        elif "movenet_thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
            input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
            """
            model = module.signatures["serving_default"]

            # SavedModel format expects tensor type of int32.
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference.
            outputs = model(input_image)
            # Output is a [1, 1, 17, 3] tensor.
            keypoints_with_scores = outputs["output_0"].numpy()
            return keypoints_with_scores

        return movenet


def get_infer_and_input_size(model_name):
    # Infer method...
    infer_method = _get_infer_method(model_name)
    # Model input size...
    input_size = 192 if "lightning" in model_name else 256

    return infer_method, input_size
