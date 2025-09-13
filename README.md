# dog-breed-identification
This project builds an end-to-end multi-class image classifier using TensorFlow and TensorFlow Hub.
## 1. Problem
> Identifying the breed of a dog given an image of a dog
## 2. Data
> As Data, we are using Kaggle's Dog Breed Identification. You can find it here
https://www.kaggle.com/competitions/dog-breed-identification
## 3. Evaluation
>The evaluation is a file with prediction probabilities for each dog breed of each test image
## 4. Features
> Some information about the data:
* We are dealing with images (Unstructured Data), so it's probably best we use deep learning/transfer learning.
* There are 120 breeds of Dogs, which means it has  120 different classes.
* There are around 10k+ test images and 10k+ train images
### Get our workspace ready
`import tensorflow as tf
import tensorflow_hub as hub
print(tf.__version__)
Check for GPU availability
print(tf.config.list_physical_devices("GPU"))`

### Getting our data ready (Turning into tensors)
With all machine learning models, our data has to be in numerical format,So that's what we will be doing first. Turning our images into tensors/numerical representations:

Let's start by accessing our data and checking out the labels

`import pandas as pd
labels_csv = pd.read_csv("/content/drive/MyDrive/Dog Breed Identificattion/labels.csv")
print(labels_csv.describe())
labels_csv.head()`

`labels_csv["breed"].value_counts().plot.bar(figsize=(20,5))`
<img width="1606" height="644" alt="image" src="https://github.com/user-attachments/assets/b0f80c3c-7063-473f-b41d-d54947acde64" />

Let's view an image
from IPython.display import Image
`Image("/content/drive/MyDrive/Dog Breed Identificattion/train/000bec180eb18c7604dcecc8fe0dba07.jpg")`

> <img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/d45c6ded-e833-4e5f-98e5-2cda1e261ca5" />
### Getting images and their labels
Let's get a list of all our image file pathnames
Create pathnames from image ID<br>
`filename = ["/content/drive/MyDrive/Dog Breed Identificattion/train/" + fname + ".jpg" for fname in labels_csv["id"]]
filename[:10]`<br>
Let's check whether filenames match the actual number of files<br>
`import os
if len(os.listdir("/content/drive/MyDrive/Dog Breed Identificattion/train/")) == len(filename):
  print("Success")
else:
  print("Check again")`
  Image(filename[200])<br>
> <img width="500" height="374" alt="image" src="https://github.com/user-attachments/assets/37d27574-505b-4474-a11e-db4f892e1bc7" />
Since we have now got our training image file paths in a list, let's prepare our labels<br>
> `import numpy as np
labels = labels_csv["breed"].to_numpy()
len(labels)`<br>
`Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)`<br>
`Turn every label into a boolean array
boolean_labels = [labels == unique_breeds for labels in labels]
boolean_labels[2]`<br>
`print(boolean_labels[0].astype(int))`
### Setting our very own validation set
As the Kaggle dataset doesn't provide us with any validation set, we are going to create our own
Set up X & y variables
`X = filename
y = boolean_labels<br>`
Setup X & y variables<br>
`X = filename
y = boolean_labels`<br>
Let's split the data into train and validation<br>
`from sklearn.model_selection import train_test_split
X_train, y_train, X_val, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)`
## Preprocessing Images (Turning images into Tensors)
Define Image size
IMG_SIZE = 224
<br>
Creating a function to preprocess the images<br>
`def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])`

 ` return image`<br>
## Turning our data into batches
Create a function to return a tuple (image, label)<br>
`def def_image_label(image_path, label):
  image = process_image(image_path)
  return image, label`<br>
BATCH_SIZE = 32
Create a function to turn data into batches
`def create_data_batches(X, y=None, batch_size = BATCH_SIZE, valid_data=False, test_data=False):
  if test_data:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  elif valid_data:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  else:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    data = data.shuffle(buffer_size=len(X))
    data = data.map(get_image_label).batch(batch_size=BATCH_SIZE)
  return data_batch`
Create training and validation data batches
`train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)`<br>
## Visualizing Data Batches
Our data is now in batches, which can be hard to understand, so we will try to visualize it
`import matplotlib.pyplot as plt
def show_25_images(images, labels):
  plt.figure(figsize=(10,10))
  for i in range (25):
    ax = plt.subplots(5, 5, i++)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i]].argmax)`
Then we unbatchify the batch and turn it into a numpy iterator so that we can see the images
`train_images, train_lables = next(train_data.as_numpy_iterator())
train_images, train_lables`
Then, if we visualize the data
<img width="832" height="811" alt="image" src="https://github.com/user-attachments/assets/ed6e02a0-5f0f-461f-96dc-a1e015b87c2d" /><br>
Notice that every time we run this, we will see different images, because in the function we have called a method named shuffle.
Now, if we want to visualize for val images
<img width="860" height="790" alt="image" src="https://github.com/user-attachments/assets/cde8c4c1-14dc-4c9b-bb9f-968613e2475b" /><br>
For val images, you won't see the images changing everything we run this, because we are not shuffling it in our function
## Building a model
Set up the input shape for the model
INPUT_SHAPE = [None. IMG_SIZE, IMG_SIZE, 3]

OUTPUT_SHAPE = len(unique_breeds)
> Our model URL https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/5

import tf_keras as keras                # <-- use tf-keras API
from tensorflow_hub import KerasLayer

INPUT_SHAPE = (128, 128, 3)
OUTPUT_SHAPE = 10
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/5"  # use feature_vector

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building with:", model_url)
    model = keras.Sequential([
        KerasLayer(model_url, input_shape=input_shape, trainable=False),  # freeze base
        keras.layers.Dense(output_shape, activation="softmax")
    ])
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model

model = create_model()
model.summary()
## Creating Callbacks

Callbacks help us as a function for our model that can be used during training for things such as saving progress, checking its progress

### Tensoboard Callback 
Load Tensorboard notebook extension
`%load_ext tensorboard<br>
import datetime
def create_tensorboard_callback():
  logdir = os.path.join("/content/drive/MyDrive/Dog Breed Identificattion/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
return tf.keras.callbacks.TensorBoard(logdir)`
### Early stopping callback
Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving
`early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience=3)`
## Training a model (On subset of data)<br>
`NUM_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}`
model = train_model()
<img width="1083" height="491" alt="image" src="https://github.com/user-attachments/assets/1b6a2ee0-aeae-4cb6-9629-534b224d768b" />
## Making and evaluating predictions using a trained model
`predictions = model.predict(val_data, verbose=1)
predictions`<br>
`Turn prediction probabilities into label
def get_pred_label(prediction_probabilities):
  """Turns an array of prediction probabilities into a label."""
  return unique_breeds[np.argmax(prediction_probabilities)]
pred_label = get_pred_label(predictions[81])
pred_label`<br>
Create a function to unbatch a dataset<br>
`def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels"""
  images = []
  labels = []
  for image, label in data.unbatch():
    images.append(image.numpy())
    labels.append(unique_breeds[label.numpy().argmax()])
  return images, labels
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]`<br>
`def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction"""
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  pred_label = get_pred_label(pred_prob)
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"
  plt.title(f"Prediction: {pred_label}, {np.max(pred_prob)*100:.1f}% | GT: {true_label}", color=color)
  plt.axis(False)`<br>
<img width="644" height="470" alt="image" src="https://github.com/user-attachments/assets/1feec23d-a78b-454c-b4b8-a10a922294fd" /><br>
`def plot_pred_conf(prediction_probabilities, labels, n=1):
  pred_prob, true_label = prediction_probabilities[n], labels[n]
  pred_label = get_pred_label(pred_prob)
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]
  colors = ["blue"] * 10  # Start with all blue bars
  if true_label in top_10_pred_labels:
    true_label_index = np.where(top_10_pred_labels == true_label)[0][0]
    colors[true_label_index] = "green" # Set the color of the true label bar to green
  plt.bar(range(10), top_10_pred_values, tick_label=top_10_pred_labels, color=colors)
  plt.xticks(rotation="vertical")`<br>
<img width="594" height="659" alt="image" src="https://github.com/user-attachments/assets/ed4f3e9f-45a8-4d5d-a8de-b61a2ff5dfb0" />
## Training a dog model on full data
`full_data = create_data_batches(X, y)
full_data`<br>

`from tf_keras.callbacks import EarlyStopping, TensorBoard
full_model_tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/Dog Breed Identificattion/logs/full_model_run")
full_model_early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)`<br>
Loading test images file name
`test_path = "/content/drive/MyDrive/Dog Breed Identificattion/test"
test_filenames = [test_path + "/" + fname for fname in os.listdir(test_path)]
test_filenames[:10]`<br>
Making Predictions
test_predictions = model.predict(test_data, verbose=1)
Create pandas dataframe
`preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()`<br>
Filling the Id column
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()<br>
