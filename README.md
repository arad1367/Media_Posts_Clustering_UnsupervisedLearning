# Image Clustering using VGG16 and K-Means

This project demonstrates an unsupervised learning approach to cluster a set of images based on their visual features. The project leverages the VGG16 convolutional neural network for feature extraction and the K-Means clustering algorithm to group similar images together.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Feature Extraction](#feature-extraction)
- [Determining Optimal Clusters](#determining-optimal-clusters)
  - [Elbow Method](#elbow-method)
  - [Silhouette Analysis](#silhouette-analysis)
- [Clustering](#clustering)
- [Evaluating Results](#evaluating-results)
- [Conclusion](#conclusion)

## Overview
The project uses the VGG16 model, pre-trained on the ImageNet dataset, to extract deep features from images. These features are then clustered using the K-Means algorithm. The optimal number of clusters is determined using the Elbow method and validated using Silhouette analysis.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- scikit-learn
- matplotlib
- tabulate

## Setup
1. **Mount Google Drive**: Ensure that your dataset is stored on Google Drive.
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Unzip the Dataset
   `!unzip /content/drive/MyDrive/LLM_ART_Projects/ClusterData.zip -d /content/drive/MyDrive/LLM_ART_Projects/`

## Feature Extraction
We use the VGG16 model to extract features from each image. The fc2 layer of the VGG16 model, which is the second fully connected layer, is used as the feature vector.
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

```
def get_model(layer='fc2'):
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
    return model
```

## Model Creation
```
def get_model(layer='fc2'):
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
    return model

# Create the model
model = get_model()

# Model summary
model.summary()

# Plotting the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

## Accessing Image Files
To process the images, we need to load and resize them to the input size expected by the VGG16 model (224x224).

```
def get_files(path_to_files, size):
    fn_imgs = []
    files = [file for file in os.listdir(path_to_files)]
    for file in files:
        img = cv2.resize(cv2.imread(path_to_files + file), size)
        fn_imgs.append([file, img])
    return dict(fn_imgs)

img_path = '/content/drive/MyDrive/LLM_ART_Projects/ClusterData/'
imgs_dict = get_files(path_to_files=img_path, size=(224, 224))

```

## Extracting Features
We extract the features of each image using the VGG16 model.

```
def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)
    arr4d = np.expand_dims(img_arr, axis=0)
    arr4d_pp = preprocess_input(arr4d)
    return model.predict(arr4d_pp)[0,:]

def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = feature_vector(img, model)
    return f_vect

img_feature_vector = feature_vectors(imgs_dict, model)

```

## Determining Optimal Clusters
** Elbow Method **
- To determine the optimal number of clusters, we use the Elbow method. We run K-Means for a range of cluster values and calculate the sum of squared distances (inertia) for each. The point where the inertia starts to decrease slowly, forming an "elbow", indicates the optimal number of clusters.
```
images = list(img_feature_vector.values())
fns = list(img_feature_vector.keys())
sum_of_squared_distances = []
K = range(1, 30)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(images)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k based on variance')
plt.show()

```

## Silhouette Analysis
Silhouette analysis is used to validate the consistency within clusters of data. The silhouette score ranges from -1 to 1, where a high value indicates that the data points are well clustered.
```
n_clusters = 4  # Adjust based on the elbow method's result
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(images)
y_kmeans = kmeans.predict(images)

silhouette_avg = silhouette_score(images, y_kmeans)
print(f'Silhouette Score: {silhouette_avg}')

```

## Clustering
After determining the optimal number of clusters, we perform the final clustering using K-Means. This step groups the images based on their feature vectors.
```
kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(images)
y_kmeans = kmeans.predict(images)
file_names = list(imgs_dict.keys())

```

## Evaluating Results
To evaluate the clustering results, we calculate the percentage of duplicated images removed in each cluster and overall. The results are presented in a tabular format.
```
img_total = len(img_feature_vector)

# Hypothetical example; actual calculation requires the number of images per cluster
cluster0_imgs_total = sum([1 for i in y_kmeans if i == 0])
cluster1_imgs_total = sum([1 for i in y_kmeans if i == 1])
cluster2_imgs_total = sum([1 for i in y_kmeans if i == 2])
cluster3_imgs_total = sum([1 for i in y_kmeans if i == 3])

percent_dup_removed = (1 - (n_clusters / img_total)) * 100

p_d_c0 = (cluster0_imgs_total / img_total) * percent_dup_removed
p_d_c1 = (cluster1_imgs_total / img_total) * percent_dup_removed
p_d_c2 = (cluster2_imgs_total / img_total) * percent_dup_removed
p_d_c3 = (cluster3_imgs_total / img_total) * percent_dup_removed

table = [['Category', 'Images', 'Duplication (%)'],
         ['Cluster_0', str(cluster0_imgs_total), p_d_c0],
         ['Cluster_1', str(cluster1_imgs_total), p_d_c1],
         ['Cluster_2', str(cluster2_imgs_total), p_d_c2],
         ['Cluster_3', str(cluster3_imgs_total), p_d_c3],
         ['Total', str(img_total), percent_dup_removed]]

print('The details of overall duplication percentage and per are presented in the below table')
print(tabulate(table, headers='firstrow', tablefmt='grid'))

```

## Conclusion
This project demonstrates the process of clustering images using unsupervised learning techniques. By leveraging VGG16 for feature extraction and K-Means for clustering, we can effectively group similar images together. The Elbow method and Silhouette analysis are essential tools for determining and validating the optimal number of clusters.

## Contact
- Pejman Ebrahimi, email: `pejman.ebrahimi77@gmail.com`
