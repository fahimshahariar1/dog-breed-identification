# dog-breed-identification
This project builds an end-to-end multi-class image classifier using TensorFlow and TensorFlow Hub.
## 1. Problem
> Identifying the breed of a dog given an image of a dog
## 2. Data
> As Data we are using Kaggle's Dog breed identification. You can find it here
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
With all machine learning models our data has to be in numerical format,So that's what we will be doing first. Turning our images into tensors/numerical representations:

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
Create pathnames from image ID
`filename = ["/content/drive/MyDrive/Dog Breed Identificattion/train/" + fname + ".jpg" for fname in labels_csv["id"]]
filename[:10]`
Lets' check whether filenames matches actual amount of files
`import os
if len(os.listdir("/content/drive/MyDrive/Dog Breed Identificattion/train/")) == len(filename):
  print("Success")
else:
  print("Check again")`
  Image(filename[200])
> <img width="500" height="374" alt="image" src="https://github.com/user-attachments/assets/37d27574-505b-4474-a11e-db4f892e1bc7" />
