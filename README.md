# Pose Transmitter

This Python project uses OpenCV to read images or videos and then uses TensorFlow and MoveNet models to estimate the pose of a human.</br>
Heavily inspired by [TensorFlow Lite Pose Estimation Example](https://www.tensorflow.org/lite/examples/pose_estimation/overview) actually, this is where I took the code provided in the examples to interact with TF. </br> >
The input can be a video file or webcam input.</br>
Finally, the information inferred by the TensorFlow model is transmitted via UDP packets to a user-defined server.

## Usage

You can run the code from the console using the following format:

`python ./pose_transmitter/main.py --source 0 --host_ip 127.0.0.1 --host_port 4900 --debug`

or:

`python ./pose_transmitter/main.py --source some_video.mp4`

or simply:

`python ./pose_transmitter/main.py`

The default value for the host is `127.0.0.1`, and for the port `4900`.

## UDP Message Structure

The messages sent via UDP have the following structure:

```
    [
        [989.79443359375, 635.7391967773438], 
        [1091.6107177734375, 512.1315307617188], 
        [889.1296997070312, 520.48828125], 
        [1204.3812255859375, 533.4179077148438], 
        ...
    ]
```

Which is a list of positions of the points found by the inference model.
It is worth noting that not all points could be inferred and sent to the list. For example, if the camera captures half a body, it will send the information of the points found on that half body.

The list of points follows the logic described in [TensorFlow Lite Pose Estimation Example](https://www.tensorflow.org/lite/examples/pose_estimation/overview), i.e.:

|Id| Part|
|-|-|
|0 |nose|
|1 |leftEye|
|2 |rightEye|
|3 |leftEar|
|4 |rightEar|
|5 |leftShoulder|
|6 |rightShoulder|
|7 |leftElbow|
|8 |rightElbow|
|9 |leftWrist|
|10| rightWrist|
|11| leftHip|
|12| rightHip|
|13| leftKnee|
|14| rightKnee|
|15| leftAnkle|
|16| rightAnkle|

For more information, refer to the TensorFlow Lite Pose Estimation documentation.

## Notes

- The performance can be improved. Since *It works on my machine ¯\_(ツ)_/¯* at a decent framerate, this is left as an exercise to the adventurers (which probably is me in the future).

## Author

- [Mayo Nesso](https://github.com/mayo-nesso)

## License

This project is open source and available under the [MIT License](LICENSE.md).
