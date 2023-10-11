import os
import argparse
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a path.")
    parser.add_argument("model_path", type=str, help="Path to ckpt and pipeline.config")

    args = parser.parse_args()
    path_to_model = args.model_path

    # Replace this with your pipeline config path and checkpoint path
    pipeline_config_path = os.path.join(path_to_model,'pipeline.config')
    model_dir = path_to_model

    # Load pipeline config and build the detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    manager = tf.train.CheckpointManager(ckpt, path_to_model, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()

    @tf.function
    def detect_fn(image):
        image = tf.cast(image, tf.float32)/255.
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    # Save the detection model as SavedModel format
    saved_model_path = 'saved_model/'
    tf.saved_model.save(detection_model, saved_model_path, signatures=detect_fn.get_concrete_function(tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)))
