import threading
from time import time

import cv2 as cv
import tensorflow as tf
from matplotlib import colors


class VideoPoseProcessor:

    def __init__(self, inference_hub, source=0, debug=False) -> None:
        """
        Initialize VideoPose object.

        Args:
            inference_hub: Inference hub object.
            source: Source for video input.
            debug: Flag to enable debug mode.
        """
        self.inference_hub = inference_hub
        self.source = 0 if source == "0" else source
        self.stop_event = threading.Event()
        self.debug = debug
        self.start_time = time()

    def _start_processing_loop(self, callback=None, display_results=True):
        """
        Start processing loop.

        Args:
            callback: Callback function to send results.
            display_results: Flag to display results.
        """
        vcap = cv.VideoCapture(self.source)
        if not vcap.isOpened():
            raise RuntimeError("VideoCapture failed to start!")

        # Get video dimensions
        image_width = vcap.get(cv.CAP_PROP_FRAME_WIDTH)
        image_height = vcap.get(cv.CAP_PROP_FRAME_HEIGHT)

        while not self.stop_event.is_set():
            success, frame = vcap.read()
            if not success:
                break

            # We need to convert from OpenCV BRG to RGB encoding...
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Run inference !
            keypoints_with_scores = self.inference_hub.run_inference(image, image_height, image_width)

            # Get points, edges and colors from keypoints...:
            (keypoint_locs, keypoint_edges, edge_colors) = self.inference_hub.keypoints_and_edges_for_display(
                keypoints_with_scores, image_height, image_width
            )

            # Send results through the callback function
            callback and callback(keypoint_locs, keypoint_edges, edge_colors)

            # Debug & Drawing edges and points!
            if display_results:
                self._display_frame(frame, keypoint_locs, keypoint_edges, edge_colors)

        # Release resources
        vcap.release()
        cv.destroyAllWindows()

    def _display_frame(self, cv_frame, keypoint_locs, keypoint_edges, edge_colors):
        """
        Display keypoints and edges on frame.

        Args:
            cv_frame: OpenCV frame.
            keypoint_locs: Keypoint locations.
            keypoint_edges: Keypoint edges.
            edge_colors: Edge colors.
        """
        for i, lp in enumerate(tf.cast(keypoint_edges, dtype=tf.int32).numpy()):
            col = colors.to_rgb(edge_colors[i])
            col = tuple(c * 255 for c in col)
            cv.line(cv_frame, lp[0], lp[1], col, thickness=1)
        for p in tf.cast(keypoint_locs, dtype=tf.int32).numpy():
            cv.drawMarker(
                cv_frame, p, (255, 0, 0), markerType=cv.MARKER_TRIANGLE_UP, markerSize=20, thickness=1, line_type=cv.LINE_AA
            )

        # FPS
        end_time = time()
        fps = 1 / (end_time - self.start_time)
        self.start_time = end_time
        cv.putText(
            cv_frame,
            f"FPS: {fps:.1f}",
            org=(20, 100),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 200, 0),
            thickness=3,
        )

        # Display image
        cv.imshow("Image", cv_frame)
        # wait time in millisecs...(desired...)
        F60 = 1000 // 60
        cv.waitKey(F60)

    def stop_processing(self):
        """Stop video processing."""
        self.stop_event.set()

    def start_processing(self, callback):
        """
        Start video processing.

        Args:
            callback: Callback function to send results.
        """
        if self.debug:
            self._start_processing_loop(callback=callback, display_results=True)
        else:
            video_thread = threading.Thread(target=self._start_processing_loop, args=(callback, False))
            video_thread.start()
            video_thread.join()
