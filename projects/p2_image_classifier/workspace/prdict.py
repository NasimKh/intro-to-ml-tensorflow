

'''
/Users/nasim/Documents/Udacity/intro-to-ml-tensorflow/projects/p2_image_classifier/test_images/cautleya_spicata.jpg

/Users/nasim/Documents/Udacity/intro-to-ml-tensorflow/projects/p2_image_classifier/workspace/label_map.json

/Users/nasim/Documents/Udacity/intro-to-ml-tensorflow/projects/p2_image_classifier/1588939321h.h5
'''




from PIL import Image
import warnings
warnings.filterwarnings('ignore')

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow_hub as hub

import json
import os


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



def predict(image_path, model, top_k):
    im = Image.open(image_path)
    numpy_image = np.asarray(im)
    image = process_image(numpy_image)
    image = np.expand_dims(image , 0 )
    prediction = model.predict(image)
    classes_unsorted = np.argpartition(prediction[0], -top_k)[-top_k:]
    probs_unsorted = prediction[0][classes_unsorted]
    classes_unsorted = classes_unsorted +1
    probs =[a for a,b in sorted(zip(probs_unsorted,classes_unsorted))]
    classes =[b for a,b in sorted(zip(probs_unsorted,classes_unsorted))]
    return probs, classes




def process_image (image) :
    IMG_SIZE = 224
    tensor_image = tf.convert_to_tensor(image)
    tensor_image = tf.cast(tensor_image, tf.float32)
    tensor_image = tf.image.resize(tensor_image, (IMG_SIZE,IMG_SIZE ))
    tensor_image /= 255
    image = tensor_image.numpy()
    return image




print ( 'please enter your image path to be classified :' )
with open(input() , 'r') as input_file:
    print ('File selected : **************************')
    print(input_file)





flower = input_file.name




print ( 'please enter your json file path for labels:' )

with open(input(), 'r') as f:
    class_names = json.load(f)
print ('File selected : **************************')




print ( 'please enter your model path :' )
with open(input() , 'r') as model:
    print ('File selected : **************************')
    print(model)





user_input = input("Enter your top n class to be shown in prediction : ")
try:
    top_k = int(user_input)
    print("you choose to show top classes of  ", top_k)
except ValueError:
    print("No.. input is not an integer.default value (5) would be taken")
    top_k = 5




saved_keras_model_filepath =  model.name




reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath , custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()





im = Image.open(flower)
test_image = np.asarray(im)
processed_test_image = process_image(test_image)

probs , classes = predict(flower, reloaded_keras_model, top_k)

keys = classes
fig, (ax1, ax2) = plt.subplots(figsize=(10,12), ncols=2)
ax1.imshow(processed_test_image, cmap = plt.cm.binary)
ax1.axis('off')

title = os.path.splitext(os.path.split(flower)[-1])[0]
ax1.set_title(title)
ax2.barh(np.arange(top_k), probs)
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(top_k))
ax2.set_yticklabels([class_names.get(str(key)) for key in classes], size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()
plt.show()

print ('top %s classesnr , class name , probability' %top_k)
np.vstack((classes ,label_class, probs)).T
