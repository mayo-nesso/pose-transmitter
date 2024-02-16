import cv2 as cv
import tensorflow as tf
from matplotlib import colors

import tf_hub_utils as th_utils


class VideoPose():

    def __init__(self, infer_method, input_size) -> None:
        self._infer = infer_method
        self.model_input_size = input_size

    def infer(self, source=0, process_pipleline=None, display_results=True):
        vcap = cv.VideoCapture(source)

        if not vcap.isOpened():
            print("E: VideoCapture failed to start! aborting...")
            return None

        # Initial crop region
        image_width  = vcap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
        image_height = vcap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
        crop_region = th_utils.init_crop_region(image_height, image_width)

        while True:
            # Read next frame from the video
            success, frame = vcap.read()
            if not success:
                break

            # We need to convert from cv BRG to RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # #
            # Actual inference !
            keypoints_with_scores = th_utils.run_inference(self._infer, image, crop_region, 
                                        crop_size=[self.model_input_size, self.model_input_size])

            # Get points, edges and colors from keypoints...:
            (keypoint_locs, keypoint_edges, edge_colors) = \
                th_utils.keypoints_and_edges_for_display(keypoints_with_scores, image_height, image_width)

            # Send results through pipeline.... -->
            process_pipleline and process_pipleline(keypoint_locs, keypoint_edges, edge_colors)
            # Debug & Drawing edges and points!
            display_results and self.display_pipe(frame, keypoint_locs, keypoint_edges, edge_colors)

            # #
            # Get new crop region!
            crop_region = th_utils.determine_crop_region(keypoints_with_scores, image_height, image_width)

        # Let it goooooo....
        vcap.release()
        del(vcap)
        cv.destroyAllWindows()

    def display_pipe(self, cv_frame, keypoint_locs, keypoint_edges, edge_colors):
        # #
        # Debug & Drawing edges and points!
        for i, lp in enumerate(tf.cast(keypoint_edges, dtype=tf.int32).numpy()):
            col = colors.to_rgb(edge_colors[i]) 
            col = tuple(c * 255 for c in col)
            cv.line(cv_frame, lp[0], lp[1], col, thickness=1)
        for p in tf.cast(keypoint_locs, dtype=tf.int32).numpy():
            cv.drawMarker(cv_frame, p, (255,0,0), markerType=cv.MARKER_TRIANGLE_UP, markerSize=20, thickness=1, line_type=cv.LINE_AA)

        # #
        # Display image
        cv.imshow("Image", cv_frame)
        # wait time in millisecs...
        F60 = 1000 // 60
        cv.waitKey(F60)
