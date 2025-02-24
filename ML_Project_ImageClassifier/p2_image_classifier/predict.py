import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt


def load_model(model_path):
    """Load a pre-trained Keras model from the given path."""
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})


def process_image(image):
    """Process an image to match the input size of the model."""
    img_size = 224  # Image size for MobileNetV2
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (img_size, img_size))
    image /= 255
    return image.numpy()


def predict(image_path, model, class_names, top_k=5):
    """Predict the top_k classes of the given image using the model."""
    if top_k < 1:
        top_k = 1

    # Load and process the image
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)

    # Add batch dimension
    expanded_image = np.expand_dims(processed_image, axis=0)

    # Make predictions
    predictions = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)

    # Convert predictions to numpy arrays
    top_k_values = top_k_values.numpy()[0]
    top_k_indices = top_k_indices.numpy()[0]

    # Map indices to class names
    predicted_classes = [class_names[str(idx + 1)] for idx in top_k_indices]
    return top_k_values, predicted_classes, image


def plot_predictions(image, top_k_values, predicted_classes):
    """Plot the image and its top predictions."""
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)

    # Plot the image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(predicted_classes[0])

    # Plot the predictions
    ax2.barh(range(len(predicted_classes)), top_k_values, color='blue')
    ax2.set_yticks(range(len(predicted_classes)))
    ax2.set_yticklabels(predicted_classes)
    ax2.invert_yaxis()
    ax2.set_title('Class Probability')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Image Class Prediction Tool")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('model_path', type=str, help="Path to the saved Keras model")
    parser.add_argument('label_map_path', type=str, help="Path to the JSON label map")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return (default: 5)")

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Load class names
    with open(args.label_map_path, 'r') as f:
        class_names = json.load(f)

    # Make predictions
    top_k_values, predicted_classes, input_image = predict(args.image_path, model, class_names, top_k=args.top_k)

    # Print results
    print("Top Predictions:")
    for i, (value, cls) in enumerate(zip(top_k_values, predicted_classes)):
        print(f"{i + 1}: {cls} ({value:.2f})")

    # Plot the predictions
    plot_predictions(input_image, top_k_values, predicted_classes)
