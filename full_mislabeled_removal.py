import numpy as np
import os
import cv2
import tensorflow as tf

from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

import shutil


# Content directory and image dimension
img_folder = 'Reduced_Dataset'
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Loading Models
encoder = tf.keras.models.load_model('bug_encoder_128.h5')
decoder = tf.keras.models.load_model('bug_decoder_128.h5')

# If true -> further dim reduction through pca
PCA_reduction = True
variance_explained = 0.5 # if pca, defined what proportion of the variance you would like explained by the principal components kept

# Output directories
cleaned_data_dir = 'Cleaned_Data_512_pca_10'
removed_data_dir = 'Removed_512_pca_10'

def create_dataset(img_folder):
    img_data_array = []
    class_name = []
    file_list = []

    for dir1 in os.listdir(img_folder):
        if dir1 == '.DS_Store':
            print('Ignoring .DS_Store')
        else:
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path)[:,:,::-1]
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
                file_list.append(file)
    return img_data_array, class_name, file_list

def apply_PCA(X):
    scaler = StandardScaler()
    X_compressed_standard = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_compressed_standard)

    sum = 0.0
    for PC in range(len(pca.explained_variance_ratio_)):
        sum += pca.explained_variance_ratio_[PC]
        if sum > variance_explained:
            PC_kept = PC
            break

    pca = PCA(n_components=10)
    return pca.fit_transform(X_compressed_standard)

# extract the image array and class name
img_data, class_name, file_list = create_dataset(img_folder)

# Dictionary for class names
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}

# convert class_names to their values
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

# As array
X = np.asarray(img_data)
y = np.asarray(target_val)

# Encode inputs with encoder
X_compressed = encoder.predict(X)

# Change directory names here if you don't wish to overwrite previous results
if not os.path.exists(cleaned_data_dir):
    os.mkdir(cleaned_data_dir)

if not os.path.exists(removed_data_dir):
    os.mkdir(removed_data_dir)

for current_class in np.unique(class_name):
    print('Processing class '+ current_class)

    # keep track of the indeces of the current class
    bool_arr = (np.array(class_name) == current_class)

    if PCA_reduction is True:
        X_compressed = apply_PCA(X_compressed)

    # n_neighbors equal to the min points on dbscan, around 2*dimensionality of the data (encoding size)
    NN = NearestNeighbors(n_neighbors=2 * X_compressed.shape[-1])
    NN_fit = NN.fit(X_compressed[bool_arr, :])
    distances, indices = NN_fit.kneighbors(X_compressed[bool_arr, :])

    # Sort distances and explore
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    elbow = KneeLocator(np.arange(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    epsilon = elbow.elbow_y

    # Only the instances in the encodings corresponding to the chosen class are selected.
    X_db = X_compressed[bool_arr, :]

    db = DBSCAN(eps=epsilon, min_samples=2 * X_compressed.shape[-1]).fit(X_db)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # identify which samples in original data are outliers and which are not
    file_list_one_type = np.array(file_list)[bool_arr]
    outlier_list = file_list_one_type[np.where(labels == -1)]  # files to remove
    clean_list = file_list_one_type[np.where(labels == 0)]  # files to keep

    # indeces of outliers in original file_list
    bool_outliers = np.where(np.in1d(file_list, outlier_list))[0]

    if not os.path.exists(removed_data_dir + '/' + current_class):
        os.mkdir(removed_data_dir + '/' + current_class)

    if not os.path.exists(cleaned_data_dir + '/' + current_class):
        os.mkdir(cleaned_data_dir + '/' + current_class)

    src = img_folder + '/' + current_class + '/'
    dst = removed_data_dir + '/' + current_class + '/'
    for file in outlier_list:
        shutil.copyfile(src + file, dst + file)

    # Filling Clean Data folder
    src = img_folder + '/' + current_class + '/'
    dst = cleaned_data_dir + '/' + current_class + '/'
    for file in clean_list:
        shutil.copyfile(src + file, dst + file)

print('Done!')