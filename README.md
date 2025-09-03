### **Multiclass Fish Image Classification Project🐟**

#### **Project Overview**

This project is a deep learning solution for classifying different species of fish from images. The goal was to build and deploy a robust image classification system. I explored two main strategies: training a simple Convolutional Neural Network (CNN) from scratch and using **transfer learning** with a variety of powerful, pre-trained models. My primary objective was to find the model that delivered the best performance and then package it into a user-friendly web application using **Streamlit**.

#### **Technologies Used**

  * **Python:** The core programming language for the project.
  * **TensorFlow/Keras:** The main deep learning framework used for building, training, and evaluating all the models.
  * **Streamlit:** For creating an interactive web application to showcase the model.
  * **NumPy & Pandas:** For data manipulation and analysis.
  * **Matplotlib & Seaborn:** For visualizing training history and model performance.

-----

#### **Project Workflow**

**1. Data Preprocessing & Augmentation**
To prepare the dataset for training, I used Keras's `ImageDataGenerator` to handle image loading and apply data augmentation on the fly. This was a crucial step to make the model more robust. Techniques like rotation, zooming, and flipping were applied to the images. The images were also automatically rescaled to a `[0, 1]` range.

**2. Model Training & Selection**
I experimented with multiple approaches to find the best-performing model:

  * A custom CNN was trained from scratch.
  * Transfer learning was implemented using five different pre-trained models: VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0.
  * The models were fine-tuned on the fish dataset.
  * After extensive testing, the **MobileNet** model was selected as the final model due to its high performance on the test set, achieving an overall accuracy of **0.99**.

**3. Model Evaluation**
The performance of each model was thoroughly evaluated. I generated a classification report, confusion matrix, and plots of training history (loss and accuracy) for all models to compare their effectiveness. The final chosen model, MobileNet, was saved in the `.h5` format for later use in the Streamlit application.

-----

#### **Execution and Deployment**

**1. Setup**
To get the project running, you first need to download the project files, including the `app.py` script and the trained model file (`mobilenet_finetuned.h5`).

**2. Running the App**
Once you have the files, you can start the Streamlit web application with a single command from your terminal. The application will automatically open in your web browser.

```bash
# Start the Streamlit application
streamlit run app.py
```

The application provides a simple interface where you can upload a fish image, and the model will predict its species and show the confidence score.

-----

**Demo Video📹**

Check out a live demo of the project on my LinkedIn:
https://www.linkedin.com/posts/monica-umamageswaran_python-deeplearning-machinelearning-activity-7368976547417894914--XmH?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE_7PqYBCyvYmCOnir7XtTdIJhnL6JtNqSA
