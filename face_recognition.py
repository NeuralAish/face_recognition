import os
import cv2
import numpy as np
import streamlit as st
from scipy.linalg import svd
from PIL import Image

# =====================================
# 1. LOAD DATASET
# =====================================
def load_images(dataset_path, img_size=(100, 100)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        temp_images = []
        temp_labels = []

        for root, _, files in os.walk(person_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        continue

                    img = cv2.resize(img, img_size)
                    temp_images.append(img.flatten())
                    temp_labels.append(current_label)

        if len(temp_images) > 0:
            images.extend(temp_images)
            labels.extend(temp_labels)
            label_dict[current_label] = person
            current_label += 1

    return np.array(images), np.array(labels), label_dict


# =====================================
# 2. PCA
# =====================================
def apply_pca(X, k=30):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    _, _, Vt = svd(X_centered, full_matrices=False)
    eigenfaces = Vt[:k]
    X_pca = np.dot(X_centered, eigenfaces.T)
    return X_pca, eigenfaces, mean_face


# =====================================
# 3. SIMPLE ANN
# =====================================
class ANN:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def train(self, X, y, epochs=300, lr=0.01):
        for _ in range(epochs):
            for i in range(len(X)):
                scores = np.dot(X[i], self.W)
                probs = self.softmax(scores)

                target = np.zeros(self.W.shape[1])
                target[y[i]] = 1

                grad = np.outer(X[i], (probs - target))
                self.W -= lr * grad

    def predict(self, x):
        scores = np.dot(x, self.W)
        probs = self.softmax(scores)
        return np.argmax(probs)


# =====================================
# 4. FACE RECOGNITION
# =====================================
def recognize_face(img, mean_face, eigenfaces, model):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (100, 100))
    img = img.flatten()
    img_pca = np.dot(img - mean_face, eigenfaces.T)
    return model.predict(img_pca)


# =====================================
# 5. STREAMLIT UI
# =====================================
st.set_page_config(page_title="Face Recognition", layout="centered")

st.title("🧑‍🦱 Face Recognition System")
st.subheader("PCA + ANN based Recognition")

st.sidebar.header("⚙️ Training Settings")
dataset_path = st.sidebar.text_input("Dataset folder path", "dataset")
train_button = st.sidebar.button("Train Model")

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button:
    if not os.path.exists(dataset_path):
        st.error("Dataset path not found.")
    else:
        with st.spinner("Loading dataset and training model..."):
            X, y, label_dict = load_images(dataset_path)

            if len(X) == 0:
                st.error("No images found in dataset.")
            else:
                X_pca, eigenfaces, mean_face = apply_pca(X, k=30)
                ann = ANN(input_size=30, output_size=len(label_dict))
                ann.train(X_pca, y)

                st.session_state.mean_face = mean_face
                st.session_state.eigenfaces = eigenfaces
                st.session_state.model = ann
                st.session_state.label_dict = label_dict
                st.session_state.trained = True

        st.success("Model trained successfully!")

st.markdown("---")

st.header("📤 Upload Image for Recognition")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg", "pgm"])

if uploaded_file and st.session_state.trained:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Recognize Face"):
        img_array = np.array(image)
        result = recognize_face(
            img_array,
            st.session_state.mean_face,
            st.session_state.eigenfaces,
            st.session_state.model
        )
        name = st.session_state.label_dict[result]
        st.success(f"✅ Recognized Person: **{name}**")

elif uploaded_file and not st.session_state.trained:
    st.warning("Please train the model first.")

st.markdown("---")
st.caption("PCA + ANN Face Recognition | Streamlit UI")
