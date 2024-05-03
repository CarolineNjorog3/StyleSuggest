# Project Description:
**Project Overview**
* Business Understanding
  - Explaining stakeholders, stakeholder audience
* Business Problem
* Data Analysis:
  - Data Description and Usage
  - Data Source
  - Data Understanding
    ***Deployment*** 
    ***Modeling***
* Recommendations 
* Conclusion and key findings
* Links to deliverables and relevant sources**

# Project Title: Personalized Virtual Fashion Platform
## Business understanding
The Personalized Virtual Fashion Platform aims to revolutionize the digital fashion experience by providing tailored styling guidance and outfit recommendations to users. Individuals struggle to navigate the vast landscape of online fashion due to the absence of personalized recommendations. This platform addresses this gap by leveraging deep learning models and user preferences to deliver hyper-personalized outfit suggestions. By understanding individual style preferences and incorporating body shape considerations, the platform empowers users to curate cohesive digital outfits with confidence. Through transparent AI practices and ethical data use, the project seeks to enhance user engagement and satisfaction, offering tangible benefits for businesses in the fashion tech industry.

#### Stakeholders
* Target stakeholders: 
- Users: Individuals interacting with the platform and benefit from its features.
- Fashion Brands: Entities that may partner with on the platform.
- Fashion Tech Companies: Organizations specializing in AI and technology solutions.
- Investors: Financial backers with a vested interest in the platform's success.

* Stakeholder Audience:
- Project Team: Involved in development, implementation, and maintenance.
- Management and Executives: Oversee project progress and provide resources.
- Marketing and Sales Teams: Promote the platform and generate revenue.

## Business Problem 
In the digital fashion landscape, the absence of personalized guidance and tailored styling experiences poses a significant challenge for users. Despite the abundance of online fashion platforms, individuals often struggle to find outfit recommendations that align with their unique style preferences and body shapes, leading to dissatisfaction and confusion, hindering confidence and self-expression. Generic product recommendations contribute to lower conversion rates. 

## Data Analysis
#### Data Description and Data Usage
The Fashion AI dataset is used in this project for training and evaluating machine learning models for virtual closet applications. It contains a collection of fashion images and associated metadata, including clothing categories, styles and attributes. Using the Fashion AI dataset, machine learning models, CNNs (convolutional neural networks) and a KNN (K-Nearest Neighbors) algorithm and a GridSearch, are used for virtual closet recommendations. The dataset is preprocessed and used to extract features representing clothing items, which are then used to generate personalized outfit suggestions for users.

#### Data Source
- Fashion AI dataset is sourced from the	AiDLab fashion dataset, which is publicly available for research purposes. It consists of images and metadata collected from various sources.
- Access:[https://github.com/AemikaChow/AiDLab-fAshIon-Data] 

#### Data Understanding 
***objectives***
1. Employ deep learning models like CNNs and transformers for fashion item visual analysis.
2.	Train models for compatibility and style cohesion to create balanced outfits.
3.	Enable hyper-personalized recommendations.
4.	Gather user preferences through explicit input (questionnaires, style quizzes) and implicit data (browsing history, saved items).
5.	Refine deep learning models to incorporate user data for hyper-tailored recommendations.
6.	Implement a user-centric visual search:
7. Enable users to upload inspirational images or reference items.
8. Develop a search system using deep learning for identifying similar or complementary styles.

#### Modeling
>>> RESNET50 M0DEL:
* Utilization of ResNet50 deep learning model to extract features from fashion images which will enable the creation of a personalized virtual fashion platform. Therefore, the following tasks are performed:
- Import necessary libraries including TensorFlow, Keras etc. 
- Load Pretrained Model, the ResNet50 model is pretrained on the ImageNet dataset and removes the top classification layers.
- Model configuration done by constructing a TensorFlow Keras sequential model by integrating a pre-trained ResNet50 model with a GlobalMaxPooling2D layer, forming a comprehensive architecture.
- Image Preprocessing by resizing and normalizing them for input into the ResNet50 model. 
- Feature Extraction, by defining a function (extract_features) to extract features from each fashion image where it: resizes the image to match the input size of the model (224x224), converts the image to a NumPy array, preprocesses the image, obtains predictions from the model and flattens the results, normalizes the results array for better predictions and then returns the normalized results array iterate through the dataset of fashion images.
- The extracted features are stored along with their corresponding filenames for further analysis.

***Reproduction Instructions***
To utilize the Virtual Closet notebook:
1. Ensure all required libraries are installed.
2. Set the path to the fashion image dataset.
3. Run each code block sequentially, following the provided instructions.

* KNN utilizes features extracted by ResNet50 to make predictions. The GridSearch process helps optimize KNN hyperparameters for the best performance when using ResNet50 features. Therefore a KNN GridSearch is performed:-

>KNN GRIDSEARCH:
- Import necessary libraries: pickle, NearestNeighbors and GridSearchCV from scikit-learn. 
- Load feature and filenames lists from pickle files containing pre-extracted features and corresponding filenames.
- Define a parameter grid for performing a grid search over hyperparameters for the K-Nearest Neighbors (KNN) algorithm.
- Initialize a KNN model without specifying hyperparameters.
- Define a dummy scoring function for use in grid search, which returns a constant value of 0.0 and is used as a placeholder.
- Perform a grid search over the defined parameter grid using GridSearchCV from scikit-learn.
- Finally, print the best parameters found by the grid search.
***Reproduction Instructions***
> To reproduce the grid search:
1. Ensure all necessary libraries are installed.
2. Load the feature vectors and filenames from the provided pickle files.
3. Define the parameter grid for KNN.
4. Initialize a KNN model.
5. Perform grid search using GridSearchCV with the defined parameter grid and dummy scorer.
6. Print the best parameters obtained from the grid search.

#### Modeling
>>> VGG16 MODEL:
* The FashionAI dataset focuses on the usage of VGG16 , a deep learning model used for image classification, to extract features from fashion images which will facilitate in similarity search and recommendation systems by: 
- Importing necessary libraries i.e Keras, sklearn , matplotlib, numpy, pandas and PIL
- Followed by Data Collection and Preprocessing where images from designated directories are gathered, their dimensions are then resized/standardized using a function called resize_images.
- The pretrained VGG16 model is loaded from the Keras Library. This model has already learned features from a large dataset of images called ImageNet. 
- The function 'extract_features' takes an image path as input, preprocesses the image to match the format expected by the VGG16 model and then extracts features from the image using the pre-trained VGG16 model and returns a flattened array of features.
- A function named find_similar_images is defined to find similar images from a given query image, calculating how much two sets of image features are alike/similar. This is done by comparing the visual traits of a chosen image with those of others in the dataset, it figures out which images are most alike.
- The visualization of both the query image and the other images that share similar visual features,  through the utilization of Matplotlib.
- Before using the model, we need to combine and preprocess label data by defining a function 'combine_and_modify_labels', providing paths to the label files (file1_path and file2_path) and the desired output path (output_path).
- Then load the CSV file containing combined labels and perform preprocessing steps such as removing specific substrings from filenames and concatenating label information.

>>> Custom CNN MODEL:
- A custom CNN model is defined using the Sequential API from Keras.
- The model architecture includes convolutional layers, max-pooling layers and fully connected layers.
- Prepare the data for model training and set up the model architecture. Load image filenames and labels, performs label encoding.
- Creates an ImageDataGenerator for data augmentation, defining the model architecture and compiling the model.
- Generate batches of training and validation data.
- Define a custom CNN model using Sequential API.
- Compilation and training the CNN model.
- Evaluation metrics such as validation loss and accuracy are computed for the model.

***Reproduction Instructions***
1. Ensure all required libraries are installed
2. Set up the directory paths to the images you want to analyze.
3. Run the provided functions to resize images to a standard size, e.g. 224x224 pixels.
4. Load the pre-trained VGG16 model to extract image features.
5. Utilize the extract_features function to compute features for each image.
6. Employ the find_similar_images function to identify similar images to a given query image.
7. Provide the path to the query image and the directory containing the dataset images.
8. Execute the function to find similar images and visualize the results.
9. Execute the combine_and_modify_labels function with paths to label files and output location.
10. Load combined labels CSV and preprocess labels.
11. Data Preparation and Model Setup where you load label DataFrame and prepare data for training.
12. Train the model using the prepared data.
13. Evaluate model performance on the validation set.

#### CNN Model evaluation
- Data: Dataset contains images; first line indicates the number of images and classes in training and validation.
- Training: Multiple epochs; progress info logged includes steps, duration, training accuracy, and loss.
- Validation: After each epoch, model assessed on validation data: validation accuracy and loss reported.
- Final Evaluation: Upon training completion, model evaluated once more on validation data, providing final accuracy and loss.

#### Deployment 
> Virtual Web App Python Script, [VirtualWebApp.py](virtualwebapp.py): 
* This Python script facilitates the creation of a user-friendly web application, which is implemented using Streamlit , allowing users to upload images and receive personalized fashion recommendations based on the virtual closet's feature extraction and recommendation functionalities.
* It includes the following components:
- Importing necessary libraries such as Streamlit, PIL (Python Imaging Library), NumPy, OpenCV, and TensorFlow.
- Setting up the layout by configuring the layout of the Streamlit web app using st.et_page_config() to set the page layout to wide and a background image using CSS styling.
- Load feature vectors and filenames from pickle files containing pre-extracted features and corresponding filenames.
- Model Configuration, loading the pre-trained ResNet50 model with ImageNet weights and removes the top classification layers.
- Define the appearance of the web application, including the title "VIRTUAL CLOSET FASHION RECOMMENDATION". 
- Functions are then defined to handle file uploads and extract features from uploaded images using the pre-trained model.
- Another function is defined to recommend fashion items based on extracted features and a nearest neighbor search algorithm.
- It then allows users to upload images through the web interface and displays the uploaded image. This is done by extracting features from the uploaded image, generates fashion recommendations and, finally, displays them on the web app.

#### Key Findings
1. The feature extraction process transforms raw fashion images into compact and informative feature vectors.
2. The KNN grid search identifies optimal parameters for the K-Nearest Neighbors algorithm, enhancing the accuracy and efficiency of fashion recommendations.
3. The Virtual Web App script provides a user-friendly interface for users to upload images and receive personalized fashion recommendations based on their preferences.
4. The system adapts to individual user preferences, ensuring personalized and relevant outfit suggestions.

## Conclusion
The integration of feature extraction tools, recommendation algorithms and the Virtual Web App script offers a comprehensive solution for personalized fashion recommendations. With deep learning models like ResNet50, the Virtual Closet notebook extracts essential features from fashion images, enabling the creation of a robust feature vector dataset. The KNN grid search further refines the recommendation process by optimizing parameters for the K-Nearest Neighbors algorithm. These extracted features and optimized parameters are then utilized by the Virtual Web App script to provide users with personalized fashion recommendations in real-time. Together, these components form a data pipeline that transforms raw fashion images into actionable insights, enhancing the digital fashion experience for users.

## Recommendations
1. Implement user feedback mechanisms to enhance recommendation accuracy.
2. Improve scalability of the virtual closet to accommodate a growing user base.
3. Explore additional deep learning models to improve fashion feature extraction.
4. Integrate user engagement analytics to track user interactions and preferences.

#### Links to deliverables and any other relevant sources
There are three deliverables for this project:
1. A **non-technical presentation** - https://www.canva.com/design/DAGCyrdVuK4/ceGwOwhQqUrrcVmLs0W05g/edit  
2. A **Jupyter Notebook**[VirtualCloset.ipynb](../Virtualcloset_notebook.ipynb) 
3. A **GitHub repository** - https://github.com/Lewis-Gitari/VIRTUAL-CLOSET-2
4. Articles can be accessed from: [https://github.com/AemikaChow/AiDLab-fAshIon-Data] under 'References' in each file where pdfs are provided for each file
