# 📸 Image Caption Generator

When we see an image, our brain can quickly generate a relevant description or caption. But what about computers? How can a machine process and label an image with a highly relevant and accurate caption? A few years back, this seemed quite impossible. However, with advancements in Computer Vision and Deep Learning algorithms, the availability of relevant datasets, and powerful AI models, building a relevant caption generator for images has become much easier. Caption generation is now a growing field, and many data annotation firms are earning billions. 

In this guide, we'll build an annotation tool capable of generating relevant captions for images using the power of deep learning. A basic understanding of two deep learning techniques, CNN and LSTM, is required.

## 📚 Dataset
Download the dataset : [Flickr 8K Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## 🛠️ Image Caption Generator with CNN - About the Project

The objective of this project is to understand the concepts of CNN and LSTM models and build a working model of an image caption generator by implementing CNN with LSTM.

In this Python project, we will implement the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long Short Term Memory). Image features will be extracted from Xception (a CNN model trained on the ImageNet dataset) and then fed into the LSTM model, which will be responsible for generating image captions.

## 📝 Explanation of Approach

### Steps Involved:

1. **Importing Modules**:
   - Import necessary libraries and modules required for subsequent tasks, including `os`, `pickle`, `numpy`, `tqdm`, and various modules from `TensorFlow` and `Keras`.

2. **Extracting Image Features**:
   - Image features are extracted using the pre-trained VGG16 model, which serves as a feature extractor for images.
   - The last classification layer of the VGG16 model is removed, leaving it as a feature extractor.
   - Extracted image features are stored in a dictionary with image IDs as keys.

3. **Loading the Captions Data**:
   - Captions data from a text file is loaded into memory.
   - Each image ID is associated with a list of its corresponding captions, creating a mapping between images and their descriptions.

4. **Preprocessing Text Data**:
   - Perform text preprocessing to prepare textual data for training machine learning models.
   - Steps include converting text to lowercase, removing special characters, punctuation, and digits, tokenizing text into words or tokens, and padding sequences.
  
5. **Train-Test Split**:
   - Divide the dataset into training and testing sets to evaluate the model's performance.
   - The training set is used to train the model, while the testing set is used for evaluation.

6. **Data Generator**:
   - Define a data generator function to generate batches of data during training.
   - It preprocesses captions, encodes them into sequences, and prepares input-output pairs for the model.

7. **Model Creation**:
   - Define the model architecture using the Functional API of Keras.
   - Typically consists of an encoder-decoder architecture where image features are combined with text features to generate captions.
   - Compile the model with appropriate loss function and optimizer.
  
8. **Training the Model**:
   - Train the model using the training data and evaluate on the testing data.
   - Training involves updating the model's parameters to minimize the loss function over multiple epochs.

9. **Generating Captions for Images**:
   - Provide functions to generate captions for new images using the trained model.
   - Pass an image through the model and use the predicted output to generate a textual description.

10. **Model Validation**:
    - Evaluate the model's performance using metrics such as BLEU scores, which measure the similarity between predicted and actual captions.
    - BLEU Score: 52% (Approx.)

11. **Visualizing Results**:
    - Functions to visualize the actual captions and predicted captions for sample images, allowing qualitative assessment of the model's performance.

## 📊 Results & Performance

- **Model BLEU Score**: ~52%
- Functions to visualize actual vs. predicted captions for qualitative assessment.

### 🌟 Features

- **Feature Extraction**: Using pre-trained VGG16 model.
- **Text Preprocessing**: Lowercasing, removing special characters, tokenizing, and padding.
- **Model Architecture**: Encoder-Decoder architecture combining CNN and LSTM.
- **Evaluation Metrics**: BLEU score for performance evaluation.
- **Visualization**: Compare actual and predicted captions for sample images.

Feel free to explore the dataset, modify the code, and experiment with different configurations to improve the model's performance. Happy coding! 🚀
