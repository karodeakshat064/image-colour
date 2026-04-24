import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import joblib
import glob
import random
import json
import tensorflow as tf

#from google.colab import files

from skimage.io import imread, imsave, imshow
from skimage.transform import resize, rotate, rescale
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from sklearn.model_selection import train_test_split

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

seed = 42
random.seed = seed
np.random.seed = seed

#Mount google drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

loaded_features_path = 'drive/My Drive/Colab Notebooks/Image Colorization/features.jbl'
kaggle_file_path = 'drive/My Drive/Colab Notebooks/Image Colorization/kaggle.json'
inception_weights_path = 'drive/My Drive/Colab Notebooks/Image Colorization/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
trained_model_path = 'drive/My Drive/Colab Notebooks/Image Colorization/'
test_image_path = 'drive/My Drive/Colab Notebooks/Image Colorization/test_images/'

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp "drive/My Drive/Colab Notebooks/Image Colorization/kaggle.json" ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets list
!kaggle datasets download -d ikarus777/best-artworks-of-all-time
!unzip 'best-artworks-of-all-time.zip'
!unzip 'resized.zip'

image_dir = 'resized/'
image_width = 256
image_height = 256
image_channels = 3
image_sample_size = 2500

def create_labels(directory):
    labels = []
    for image in os.listdir(directory):
      filename = image.split('.')
      label = filename[0].split('_')
      result = ""
      for word in label:
        if not word.isdigit():
          result = result + word + '_'
      labels.append(result[:-1])
    return set(labels)

image_labels = create_labels(image_dir)
image_labels = list(image_labels)

print(image_labels)

def create_label_folders(directory, labels):
    for label in labels:
      label_dir = directory + '{}'.format(label)
      if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        
    for image in glob.glob('resized/*.jpg'):
      
      filename = image.split('.')
      label = filename[0].split('_')
      result = ""
      for word in label:
        if not word.isdigit():
          result = result + word + '_'
      result = result[:-1]
      #print(image)
      #print(result + '/' + image.split('/')[1])
      if result.split('/')[1] in labels:
        os.rename(image, result + '/' + image.split('/')[1])
    
    print('Created folders for labels.')

create_label_folders(image_dir, image_labels)

def get_labels_and_class_distributions(directory, labels, title=''):
    filenames = []
    class_lengths = []
    for label in labels:
        label_dir = directory + '{}'.format(label)
        images = [label + '/' + im for im in os.listdir(label_dir)]
        filenames.extend(images)
        print("{0} photos of {1}".format(len(images), label)) 
        class_lengths.append(len(images))
        
    print('Image sample total: ', len(filenames))
    plt.bar(range(len(class_lengths)), class_lengths)
    #plt.xticks(range(len(labels)), labels)
    plt.xticks([])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    return filenames

image_filenames = get_labels_and_class_distributions(image_dir, image_labels, title='Artworks class distribution')

print(len(image_filenames))

def read_image(src, width, height, channels, preserve_range=False):
    img = imread(src)
    if preserve_range:
      img_resized = resize(img, (width, height, channels), mode='reflect', preserve_range=True)
    else:
      img_resized = resize(img, (width, height, channels), mode='reflect')
    return img_resized

def plot_images(directory, labels, image_width=image_width, image_height=image_height, image_channels=image_channels,
                examples=25, disp_labels=True): 
  
    if not math.sqrt(examples).is_integer():
      print('Please select a valid number of examples.')
      return
    
    imgs = []
    classes = []
    for i in range(examples):
        rnd_idx_cl = np.random.randint(0, len(labels))
        rnd_class = labels[rnd_idx_cl]
        rnd_idx = np.random.randint(0, len(os.listdir(directory + '/' + rnd_class)))
        filename = os.listdir(directory + '/' + rnd_class)[rnd_idx]
        img = read_image(os.path.join(directory + '/' + rnd_class, filename), width=image_width, height=image_height, 
                        channels=image_channels)
        imgs.append(img)
        classes.append(rnd_class)
    
    
    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),
    subplot_kw = {'xticks':[], 'yticks':[]},
    gridspec_kw = dict(hspace=0.3, wspace=0.01))
    
    for i, ax in enumerate(axes.flat):
        if disp_labels == True:
          ax.title.set_text(classes[i])
        ax.imshow(imgs[i])

plot_images(image_dir, image_labels)

plot_images(image_dir, image_labels, examples=100, disp_labels=False)

def load_features(directory, labels, image_width=image_width, image_height=image_height, image_channels=image_channels, 
                  sample_size=image_sample_size, loaded=False, features_path=loaded_features_path):
  
    
    if loaded:
        print('Loading saved dataset..')
        if os.path.isfile(features_path):
            features = joblib.load(features_path)
        print('Done')
        return features.astype('float32') / 255.
    
    features = np.zeros((image_sample_size, image_width, image_height, image_channels), dtype=np.uint8)
    for i in range(sample_size):
        rnd_idx_cl = np.random.randint(0, len(labels))
        rnd_class = labels[rnd_idx_cl]
        rnd_idx = np.random.randint(0, len(os.listdir(directory + '/' + rnd_class)))
        filename = os.listdir(directory + '/' + rnd_class)[rnd_idx]
        img = read_image(os.path.join(directory + '/' + rnd_class, filename), width=image_width, height=image_height, channels=
                        image_channels, preserve_range=True)
        features[i] = img
        if i % 100 == 0:
              print(('Processed {:.2f}% of images').format((i/sample_size) * 100))

    print('Done')
    if loaded == False:
        print('Saving dataset to disk..')
        joblib.dump(features, features_path)
    return features.astype('float32') / 255.

#img_features = load_features(directory=image_dir, labels=image_labels)
img_features = load_features(directory=image_dir, labels=image_labels, loaded=True)

img_features.shape

img_features[0]

def plot_loaded_features(feature, num_cols=3, num_rows=1, random=1, first_idx=0):
    f, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15,8))
    idx = first_idx
    for i in range(num_cols):
        if random:
            idx = np.random.randint(0, len(feature))

        ax[i].grid(False)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(feature[idx])
        idx += 1
        
    plt.show()

plot_loaded_features(img_features)

imshow(rgb2gray(img_features[10]))
plt.xticks([])
plt.yticks([])
plt.show()
print('Grayscale image shape: ', rgb2gray(img_features[10]).shape)

features_train, features_test = train_test_split(img_features, test_size=25, random_state=seed)
features_train, features_val = train_test_split(features_train, test_size=0.1, random_state=seed)

print('Train features size: ', features_train.shape)
print('Validation features size: ', features_val.shape)
print('Test features size: ', features_test.shape)

# download from github
#inception = InceptionResNetV2(include_top=True)
#inception.graph = tf.get_default_graph()

# or load locally
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights(inception_weights_path)
inception.graph = tf.get_default_graph()

inception.summary()

batch_size = 20
num_epochs = 30
num_steps_per_epoch = len(features_train) / batch_size
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)

def augment_data(dataset, batch_size=batch_size):
    for batch in datagen.flow(dataset, batch_size=batch_size):
        X_batch = rgb2gray(batch)
        grayscaled_rgb = gray2rgb(X_batch)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield [X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch

def Colorizer():
    embed_input = Input(shape=(1000,))
    
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

model = Colorizer()
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model_stopper = EarlyStopping(monitor='val_loss', patience=6)
model_checkpointer = ModelCheckpoint(trained_model_path + 'best_colorizer.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')

def plot_colorization_results(features_test, samples=10):
  
  grayscaled_rgb = gray2rgb(rgb2gray(features_test))
  grayscaled_rgb_embed = create_inception_embedding(grayscaled_rgb)
  grayscaled_rgb = rgb2lab(grayscaled_rgb)[:,:,:,0]
  grayscaled_rgb = grayscaled_rgb.reshape(grayscaled_rgb.shape+(1,))

  output = model.predict([grayscaled_rgb, grayscaled_rgb_embed])
  output = output * 128

  decoded_imgs = np.zeros((len(output),256, 256, 3))
  
  for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = grayscaled_rgb[i][:,:,0]
    cur[:,:,1:] = output[i]
    decoded_imgs[i] = lab2rgb(cur)
    
  fig, axes = plt.subplots(nrows=3, ncols=samples, figsize=(25, 10), subplot_kw = {'xticks':[], 'yticks':[]}, 
                           gridspec_kw = dict(hspace=0.05, wspace=0.01))
  rows = ['{}'.format(row) for row in ['Grayscaled images', 'Colorized images', 'Original images']]
  for ax, row in zip(axes[:,int(samples/2)], rows):
    ax.set_title(row, rotation=0, size='large')
    

  for i, ax in enumerate(axes.flat):
      if i < samples:
        ax.imshow(rgb2gray(features_test)[i].reshape(256, 256), cmap='gray')
      elif i < samples*2 and i >= samples:
        ax.imshow(decoded_imgs[i - samples].reshape(256, 256, 3))
      else:
        ax.imshow(features_test[i - (samples*2)].reshape(256, 256, 3))
      

  plt.show()
  return decoded_imgs

def plot_learning_curves(training_history):
  
    plt.figure(figsize=(16,6))
    plt.subplot(1, 2, 1)
    plt.plot(training_history['loss'], label="Training Loss")
    plt.plot(training_history['val_loss'], label="Validation loss")
    plt.title('Image Colorization: Loss functions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history['lr'], label="Learning rate")
    plt.title('Image Colorization: Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.show()

training_history = model.fit_generator(augment_data(features_train, batch_size),
                        epochs=num_epochs,
                        validation_data=augment_data(features_val, batch_size),
                        validation_steps=(len(features_val) / batch_size),
                        steps_per_epoch=num_steps_per_epoch,
                        callbacks=[model_stopper, model_checkpointer, learning_rate_reduction])

#model.save(trained_model_path + 'colorization_model.h5')
#model.save_weights(trained_model_path + 'colorization_weights.h5')
model.load_weights(trained_model_path + 'colorization_weights.h5')

def save_hist(filename, history):
    with open(filename, "w") as write_file:
        json.dump(history, write_file)
def load_hist(filename):
    with open(filename, 'r', encoding='utf-8') as read_file:
        train_hist = json.loads(read_file.read())
        return train_hist

#save_hist(trained_model_path + 'colorization_model.json', str(training_history.history))
training_history = load_hist(trained_model_path + 'colorization_model.json')

#plot_learning_curves(training_history.history)
plot_learning_curves(eval(training_history))

decoded_test_imgs = plot_colorization_results(features_test)

for i in range(len(decoded_test_imgs)):
  plt.imshow(decoded_test_imgs[i])
  plt.xticks([])
  plt.yticks([])
  plt.show()

batch_size = 16
num_epochs = 30
num_steps_per_epoch = len(features_train) / batch_size
datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10,
        horizontal_flip=True)

training_history = model.fit_generator(augment_data(features_train, batch_size),
                        epochs=num_epochs,
                        validation_data=augment_data(features_val, batch_size),
                        validation_steps=(len(features_val) / batch_size),
                        steps_per_epoch=num_steps_per_epoch,
                        callbacks=[model_checkpointer, learning_rate_reduction])

#model.save(trained_model_path + 'colorization_model_opt.h5')
#model.save_weights(trained_model_path + 'colorization_weights_opt.h5')
model.load_weights(trained_model_path + 'colorization_weights_opt.h5')

#save_hist(trained_model_path + 'colorization_model_opt.json', str(training_history.history))
training_history = load_hist(trained_model_path + 'colorization_model_opt.json')

#plot_learning_curves(training_history.history)
plot_learning_curves(eval(training_history))

decoded_test_imgs = plot_colorization_results(features_test)

for i in range(len(decoded_test_imgs)):
  plt.imshow(decoded_test_imgs[i])
  plt.xticks([])
  plt.yticks([])
  plt.show()

plt.figure(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(decoded_test_imgs[0])
plt.subplot(1, 2, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(decoded_test_imgs[10])

plt.show()

def load_test_images(directory, image_sample_size=3, image_width=image_width, image_height=image_height, 
                     image_channels=image_channels):
    
    features = np.zeros((image_sample_size, image_width, image_height, image_channels), dtype=np.uint8)
    idx = 0
    for image_file in os.listdir(directory):
        img = read_image(os.path.join(directory, image_file), width=image_width, height=image_height, channels=
                        image_channels, preserve_range=True)
        features[idx] = img
        idx = idx + 1

    print('Done')
    return features.astype('float32') / 255.

test_images = load_test_images(directory=test_image_path)

def colorize_test_images(features_test, samples=3):
  
  grayscaled_rgb = gray2rgb(rgb2gray(features_test))
  grayscaled_rgb_embed = create_inception_embedding(grayscaled_rgb)
  grayscaled_rgb = rgb2lab(grayscaled_rgb)[:,:,:,0]
  grayscaled_rgb = grayscaled_rgb.reshape(grayscaled_rgb.shape+(1,))

  output = model.predict([grayscaled_rgb, grayscaled_rgb_embed])
  output = output * 128

  decoded_imgs = np.zeros((len(output),256, 256, 3))
  
  for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = grayscaled_rgb[i][:,:,0]
    cur[:,:,1:] = output[i]
    decoded_imgs[i] = lab2rgb(cur)
    
  fig, axes = plt.subplots(nrows=2, ncols=samples, figsize=(16, 14), subplot_kw = {'xticks':[], 'yticks':[]}, 
                           gridspec_kw = dict(hspace=0.25, wspace=0.01))
  rows = ['{}'.format(row) for row in ['Colorized images', 'Original images']]
  for ax, row in zip(axes[:,int(samples/2)], rows):
    ax.set_title(row, rotation=0, size='large')
    

  for i, ax in enumerate(axes.flat):
      if i < samples:
        ax.imshow(decoded_imgs[i].reshape(256, 256, 3))
      else:
        ax.imshow(features_test[i - samples].reshape(256, 256, 3))
      

  plt.show()
  return decoded_imgs

test_colorized_images = colorize_test_images(test_images)