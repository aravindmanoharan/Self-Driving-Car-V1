# Import all packages
import cv2
import numpy as np
import tensorflow as tf

CHECKPOINT = "./train_model.ckpt"

# Ignoring the INFO from the tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

loaded_graph = tf.Graph()

# Start the loaded graph session
with tf.Session(graph=loaded_graph) as sess:

    # Load the saved model
    loader = tf.train.import_meta_graph(CHECKPOINT + '.meta')
    loader.restore(sess, CHECKPOINT)

    # Load the required parameters from the graph
    final_layer = loaded_graph.get_tensor_by_name('fc3/BiasAdd:0')
    input_layer = loaded_graph.get_tensor_by_name('input:0')

    # Function which returns the predicted steering angle
    def steering_angle_predict(img):
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, 40, 40, 1))
        
        test_pred = sess.run(final_layer, feed_dict={input_layer: img})
            
        return np.squeeze(test_pred)

    steer = cv2.imread('steering_wheel_image.jpg', 0)
    rows, cols = steer.shape
    smoothed_angle = 0

    # Initiate the dataset video
    cap = cv2.VideoCapture('run.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()

        # Resize every frame into 40 x 40
        gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
        steering_angle = steering_angle_predict(gray)
        
        cv2.imshow('frame', cv2.resize(frame, (500, 240), interpolation=cv2.INTER_AREA))

        # Smoothing the predicted steering angle 
        smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(steer, M, (cols, rows))
        cv2.imshow("steering wheel", dst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()