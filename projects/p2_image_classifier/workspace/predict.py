import argparse

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

import argparse

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
    probs =[a for a,b in sorted(zip(probs_unsorted,classes_unsorted),reverse=True)]
    classes =[b for a,b in sorted(zip(probs_unsorted,classes_unsorted),reverse=True)]
    return probs, classes




def process_image (image) :
    IMG_SIZE = 224
    tensor_image = tf.convert_to_tensor(image)
    tensor_image = tf.cast(tensor_image, tf.float32)
    tensor_image = tf.image.resize(tensor_image, (IMG_SIZE,IMG_SIZE ))
    tensor_image /= 255
    image = tensor_image.numpy()
    return image



def main():
    parser = argparse.ArgumentParser(
        description='adding image and model path.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       default='./test_images/cautleya_spicata.jpg',
                       help='Input flower image for the classification.')
    parser.add_argument('model',
                       metavar='path',
                       type=str,
                       default='my_model.h5',
                       help='Input model for the classification.')
    # parser.add_argument('Path',
    #     '--input',
    #     #type=argparse.FileType('r'),
    #     metavar='path',
    #     type=str,

    #     default='./test_images/cautleya_spicata.jpg',
    #     #required=True ,
    #     help='Input flower image for the classification.'
    #     )
    # parser.add_argument('model',
    #     '--checkout',
    #     metavar='path',
    #     type = str,
    #     default='my_model.h5',
    #     #required=True,
    #     help='Input model for the classification.'
    #     )
    parser.add_argument(
        '--category_names',
        type=argparse.FileType('r'),
        default='label_map.json',
        required=False,
        help='provide a json file for label names(default: label_map.json)'
    )
    parser.add_argument(
        '--top_k',
        default=5,
        type=int,
        help='provide an integer for the top_k classes (default: 5)'
    )
    args = parser.parse_args()
    #print(args)
    return args




if __name__ == '__main__':
    # taking the args from command line and pathing them
    args = main()
    #print(args)
    flower = args.Path
    saved_keras_model_filepath = args.model
    top_k = args.top_k
    class_names = json.load(args.category_names)
    # print(flower)
    # print(saved_keras_model_filepath)
    # print(class_names)

    reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath , custom_objects={'KerasLayer':hub.KerasLayer})

    #reloaded_keras_model.summary()





    im = Image.open(flower)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)

    probs , classes = predict(flower, reloaded_keras_model, top_k)



    keys = classes
    label_class = [class_names.get(str(key)) for key in classes]

    print ('top %s classesnr , class name , probability' %top_k)
    print(np.vstack((classes ,label_class, probs)).T)

    fig, (ax1, ax2) = plt.subplots(figsize=(10,8), ncols=2)
    ax1.imshow(processed_test_image, cmap = plt.cm.binary)
    ax1.axis('off')

    title = os.path.splitext(os.path.split(flower)[-1])[0]
    ax1.set_title(title)
    ax2.barh(np.arange(top_k), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(label_class, size='small');
    for i, v in enumerate(probs):
        ax2.text(v + 0.03 , i , str(v), color='black', size=5)

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


