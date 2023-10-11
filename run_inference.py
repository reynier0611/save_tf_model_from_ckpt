import cv2
import tensorflow as tf

if __name__ == '__main__':

    img = cv2.imread("img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(799,504))

    saved_model_path = 'saved_model'
    loaded_model = tf.saved_model.load(saved_model_path)
    infer = loaded_model.signatures["serving_default"]

    tensor_img = tf.constant(img)

    detections = infer(tf.expand_dims(tensor_img, axis=0))
    print(detections)