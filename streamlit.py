import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
vgg_model = VGG16(weights='imagenet', include_top=False)
resnet_model = ResNet50(weights='imagenet', include_top=False)

# Function to extract features from an image using VGG16
def extract_features_vgg(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # VGG16 input size
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input_vgg(img_data)
        features = vgg_model.predict(img_data)
        return features.flatten()
    except UnidentifiedImageError:
        print(f"Error loading image: '{img_path}'. Image is corrupted or in an unsupported format.")
        return None  # Return None to indicate failure
    except Exception as e:
        print(f"Error processing image '{img_path}': {e}")
        return None  # Return None to indicate failure

# Function to extract features from an image using ResNet50
def extract_features_resnet(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 input size
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input_resnet(img_data)
        features = resnet_model.predict(img_data)
        return features.flatten()
    except UnidentifiedImageError:
        print(f"Error loading image: '{img_path}'. Image is corrupted or in an unsupported format.")
        return None  # Return None to indicate failure
    except Exception as e:
        print(f"Error processing image '{img_path}': {e}")
        return None  # Return None to indicate failure

# Function to find similar images given a query image
def find_similar_images(query_img, dataset_dir=None, top_n=5, model="VGG16"):
    if model == "VGG16":
        extract_features = extract_features_vgg
    else:
        extract_features = extract_features_resnet
    
    query_features = extract_features(query_img)
    similarities = {}
    if dataset_dir is not None:
        dataset_dir = os.path.join("Closet/", dataset_dir)  # Ensure "Closet/" prefix is included
        for img_file in os.listdir(dataset_dir):
            img_path = os.path.join(dataset_dir, img_file)
            img_features = extract_features(img_path)
            if img_features is not None:
                similarity = cosine_similarity([query_features], [img_features])[0][0]
                similarities[img_file] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    similar_images = [img_file for img_file, _ in sorted_similarities[:top_n]]
    return similar_images

# Streamlit app
def main():
    st.title("StyleSuggest: Your personal fashion stylist")
    st.write("StyleSuggest is your personal fashion stylist at your fingertips. With our innovative app, you can discover endless style inspiration and receive tailored outfit recommendations based on your unique preferences and body type. Whether you're seeking the perfect ensemble for a special occasion or everyday wear, StyleSuggest helps you elevate your fashion game effortlessly. Say goodbye to fashion dilemmas and hello to confidence-boosting style advice with StyleSuggest.")
    
    query_img = st.file_uploader("Upload Inspo Image", type=["jpg", "jpeg", "png"])
    if query_img is not None:
        st.image(query_img, caption="Inspo Image", use_column_width=True)
        dataset_dirs = st.sidebar.selectbox("Choose your Fashion Outlet", [""] + os.listdir("Closet/"))
        st.write("Selected Fashion Outlet:", dataset_dirs)  # Add this line to print the selected directory
        top_n = st.sidebar.slider("Number of Outfit Recommendations", min_value=1, max_value=10, value=5)
        model = st.sidebar.selectbox("Select Model", ["VGG16", "ResNet50"])
        if st.sidebar.button("Find Outfits"):
            similar_images = find_similar_images(query_img, dataset_dirs, top_n, model)
            st.subheader("Your Outfit Recommendations:")
            if similar_images:
                for img_file in similar_images:
                    img_path = os.path.join("Closet/", dataset_dirs, img_file)  # Include "Closet/" prefix
                    img_data = Image.open(img_path)
                    st.image(img_data, caption=img_file, use_column_width=True)
            else:
                st.write("Sorry, no outfits found.")

if __name__ == "__main__":
    main()
