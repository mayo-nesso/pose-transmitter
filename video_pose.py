import cv2 as cv
import tensorflow as tf
from matplotlib import colors

import tf_hub_utils as th_utils


class VideoPose():
    def __init__(self, model, input_size) -> None:
        self.model = model
        self.model_input_size = input_size

    def _infer(self, input_image):
        """Runs detection on an input image.

        Args:
        model: Model used to infer keypoints on input_image. Retrieved from
            module. ie `model = module['serving_default']`
            https://www.kaggle.com/models/google/movenet/frameworks/tensorFlow2/variations/singlepose-lightning
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = self.model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    
    def infer(self, source=0):
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

            # Debug & Drawing edges and points! 
            if not self.display_pipe(frame, keypoint_locs, keypoint_edges, edge_colors):
                break
            
            # #
            # Get new crop region!
            crop_region = th_utils.determine_crop_region(keypoints_with_scores, image_height, image_width)
            
        
        # Let it goooooo....
        vcap.release()
        del(vcap)


    def display_pipe(self, cv_frame, keypoint_locs, keypoint_edges, edge_colors) -> bool:
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
        cv.imshow("Image" , cv_frame)
        
        # # #
        # ESC pressed
        if cv.waitKey(20) == 27:
            print("Escape hit, closing...")
            return False

        return True