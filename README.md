# Image-Caption-Generator

Whenever an image appears in front of us, our brain can annotate or label it. But what about computers? How can a machine process and label an image with a highly relevant and accurate caption? It seemed quite impossible a few years back. 


Still, with the enhancement of Computer Vision and Deep learning algorithms, the availability of relevant datasets, and AI models, it becomes easier to build a relevant caption generator for an image. Even Caption generation is growing worldwide, and many data annotation firms are earning billions. In this guide, we will build one such annotation tool capable of generating relevant captions for the image with the help of datasets. Basic knowledge of two Deep learning techniques, including LSTM and CNN, is required.

Dataset Link - https://www.kaggle.com/datasets/adityajn105/flickr8k

# Image Caption Generator with CNN – About the Python based Project

The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.

In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks)* and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions

# Explaination of approach 

Steps Involved:

1 . Importing Modules:

This step involves importing necessary libraries and modules required for the subsequent tasks, including os, pickle, numpy, tqdm, and various modules from TensorFlow and Keras.

2 . Extracting Image Features:

- Image features are extracted using the pre-trained VGG16 model, which serves as a feature extractor for images.
- The last classification layer of the VGG16 model is removed, leaving it as a feature extractor.
- Extracted image features are stored in a dictionary with image IDs as keys.

3 . Loading the Captions Data:

- Captions data from a text file is loaded into memory.
- Each image ID is associated with a list of its corresponding captions, creating a mapping between images and their descriptions.

4 . Preprocessing Text Data:

- Text preprocessing is performed to prepare textual data for training machine learning models.
- Steps include converting text to lowercase, removing special characters, punctuation, and digits, tokenizing text into words or tokens, and padding sequences.
  
5 . Train-Test Split:

- The dataset is divided into training and testing sets to evaluate the model's performance.
- The training set is used to train the model, while the testing set is used for evaluation.

6 . Data Generator:

- A data generator function is defined to generate batches of data during training.
- It preprocesses captions, encodes them into sequences, and prepares input-output pairs for the model.

7 . Model Creation:

- The model architecture is defined using the Functional API of Keras.
- It typically consists of an encoder-decoder architecture where image features are combined with text features to generate captions.
- The model is compiled with appropriate loss function and optimizer.
  
8 . Training the Model:

- The model is trained using the training data and evaluated on the testing data.
- Training involves updating the model's parameters to minimize the loss function over multiple epochs.

9 . Generating Captions for Images:

- Functions are provided to generate captions for new images using the trained model.
- This involves passing an image through the model and using the predicted output to generate a textual description.

10 . Model Validation:

- The model's performance is evaluated using metrics such as BLEU scores, which measure the similarity between predicted and actual captions.
- BLEU Score : 52%(Approx.)

11 . Visualizing Results:

- Functions are available to visualize the actual captions and predicted captions for sample images, allowing qualitative assessment of the model's performance.
