import threading
from time import time

import cv2 as cv
import tensorflow as tf
from matplotlib import colors


class VideoPose():

    def __init__(self, thub, source=0, debug=False) -> None:
        self.thub = thub
        self.source = source
        self._stop_event = threading.Event()
        self.DEBUG = debug
        self.start_time = time()

    def _start_processing_loop(self, callback=None, display_results=True):
        vcap = cv.VideoCapture(self.source)
        # vcap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        # vcap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        # vcap.set(cv.CAP_PROP_FPS, 30)
        if not vcap.isOpened():
            print("E: VideoCapture failed to start! aborting...")
            return None

        # Initial crop region
        image_width = vcap.get(cv.CAP_PROP_FRAME_WIDTH)
        image_height = vcap.get(cv.CAP_PROP_FRAME_HEIGHT)

        while not self._stop_event.is_set():
            # Read next frame from the video
            success, frame = vcap.read()
            if not success:
                break

            # We need to convert from cv BRG to RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # #
            # Actual inference !
            keypoints_with_scores = self.thub.run_inference(image, image_height, image_width)

            # Get points, edges and colors from keypoints...:
            (keypoint_locs, keypoint_edges, edge_colors) = self.thub.keypoints_and_edges_for_display(
                keypoints_with_scores, image_height, image_width
            )

            # #
            # Send results through pipeline.... -->
            callback and callback(keypoint_locs, keypoint_edges, edge_colors)
            # Debug & Drawing edges and points!
            display_results and self._display_pipe(frame, keypoint_locs, keypoint_edges, edge_colors)

        # Let it goooooo....
        vcap.release()
        del(vcap)
        cv.destroyAllWindows()

    def _display_pipe(self, cv_frame, keypoint_locs, keypoint_edges, edge_colors):
        # #
        # Debug & Drawing edges and points!
        for i, lp in enumerate(tf.cast(keypoint_edges, dtype=tf.int32).numpy()):
            col = colors.to_rgb(edge_colors[i]) 
            col = tuple(c * 255 for c in col)
            cv.line(cv_frame, lp[0], lp[1], col, thickness=1)
        for p in tf.cast(keypoint_locs, dtype=tf.int32).numpy():
            cv.drawMarker(cv_frame, p, (255,0,0), markerType=cv.MARKER_TRIANGLE_UP, markerSize=20, thickness=1, line_type=cv.LINE_AA)

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
        # #
        # Display image
        cv.imshow("Image", cv_frame)
        # wait time in millisecs...(desired...)
        F60 = 1000 // 60
        cv.waitKey(F60)

    def stop_processing(self):
        self._stop_event.set()

    def start_processing(self, callback):
        if self.DEBUG:
            self._start_processing_loop(callback=callback, display_results=True)
        else:
            video_thread = threading.Thread(target=self._start_processing_loop, args=(callback, False))
            video_thread.start()
            #    # Esperar a que los threads terminen
            video_thread.join()
