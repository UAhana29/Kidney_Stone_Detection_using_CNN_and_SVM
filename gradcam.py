import tensorflow as tf
import cv2
import numpy as np
import os

def generate_heatmap(model, image_path, img_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = np.expand_dims(img_resized/255.0, axis=0)

    last_conv = model.get_layer("block5_conv3")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output_path = "static/heatmaps/heatmap.jpg"
    cv2.imwrite(output_path, heatmap)

    return output_path