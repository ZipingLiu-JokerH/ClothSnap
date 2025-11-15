# ClothSnap

Ottawa DTI6302 Machine Learning Opearations End to End project

## :star: Project Overview

This project builds an end-to-end machine learning system that classifies clothing images into predefined categories. With modern households owning large and ever-changing wardrobes, organizing and tracking clothing items can be difficult. A reliable image-based classifier can serve as the first step toward an automated wardrobe-management system, where users simply take a photo and have the item identified and categorized.

To demonstrate this concept, the project uses TensorFlow ( :heavy_exclamation_mark: TODO [include other tech stack used] ) to train a clothing-classification model and integrates it into a simple web application that provides immediate predictions for user-uploaded images.

## :mag_right: Dataset Selection

For this project, we use the [Top-10 Clothing Dataset](https://github.com/alexeygrigorev/clothing-dataset-small), a subset of the [Clothing Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full) published on Kaggle by [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/). This version contains images from the ten most common clothing categories, addressing the class-imbalance issues present in the full 20-class dataset.

### Features

Each data point consists of a single RGB image of a clothing item. The images vary in background, lighting, orientation, and zoom level, providing realistic visual diversity.

### Target Variable

The target variable is the clothing category, represented as one of the ten class labels:
T-shirt, Long Sleeve, Pants, Shirt, Shoes, Dress, Shorts, Outwear, Hat, and Skirt.
Each image belongs to exactly one category.

### Relevance and Significance

This dataset aligns well with the project’s goal of building an image-based clothing classifier that could support a future wardrobe-management system. By focusing on the ten most common clothing categories, the dataset provides enough balanced examples for reliable model training while still reflecting the types of items people frequently own. Its natural variation in lighting and backgrounds makes it practical for a real user scenario where photos may be taken casually at home.
