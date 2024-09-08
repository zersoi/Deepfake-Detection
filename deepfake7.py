import tensorflow as tf
import cv2
import sys
import os

# Function to resize a face to a specified target size
def resize_face(face, target_size=(256, 256)):
    return cv2.resize(face, target_size)

# Function to create a CNN model for deepfake detection
def create_cnn_model():
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the deepfake detection model
def train_model(real_images, fake_images):
    real_images = tf.convert_to_tensor(real_images, dtype=tf.float32)
    fake_images = tf.convert_to_tensor(fake_images, dtype=tf.float32)

    x_train = tf.concat([real_images, fake_images], axis=0)
    y_train = tf.concat([tf.ones(len(real_images)), tf.zeros(len(fake_images))], axis=0)

    model = create_cnn_model()
    model.fit(x_train, y_train, epochs=10)

    return model

# Function to load and preprocess images from a directory
def load_images_from_directory(directory, target_size=(256, 256)):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize and preprocess the image
                image = cv2.resize(image, target_size)
                image_list.append(image)
    return image_list

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is not None:
        # Resize and preprocess the image
        image = cv2.resize(image, target_size)
    return image

if __name__ == "__main__":
    action = sys.argv[1]

    if action == "train":
        real_images = load_images_from_directory("dataset/real", target_size=(256, 256))
        fake_images = load_images_from_directory("dataset/fake", target_size=(256, 256))

        model = train_model(real_images, fake_images)
        model.save("image_deepfake_model.h5")

    elif action == "check":
        model = tf.keras.models.load_model("image_deepfake_model.h5")
        image_path = sys.argv[2]

        image = load_and_preprocess_image(image_path, target_size=(256, 256))

        result = model.predict(tf.convert_to_tensor([image], dtype=tf.float32))
        if result < 0.5:
            print("Fake")
        else:
            print("Real")

    else:
        print("Unknown action")
