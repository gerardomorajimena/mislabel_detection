
# Unsupervised mislabeled image detection using Autoencoders

Annotated data of insects caught in sticky traps presents several labelling mistakes due to the complexity of the task. Many insect species are morphologically similar to each other and often of very small size, which makes them harder to recognize through images shot in standard cell phone cameras.  Therefore, even trained entomologists might fail to accurately label all images. 
Since a cleaned and correctly annotated dataset is vital for many computer vision tasks, we explored an unsupervised machine learning methodology for identifying and removing wrong-labelled images in the dataset. 
Convolutional autoencoders generate a lower dimensional feature representation of images in an unsupervised manner. Assumingly, theses features contain the most relevant information of the images to allow for a proper reconstruction though the decoding phase. Using the DBSCAN clustering algorithm on each individual class on the encoded features results in the detection of outliers which are often either poor or mislabeled samples.

## Note

For a detailed description of the methodology refer to the "mislabel_detection_intro.pdf". This work is inpired by "Yang, Y., & Whinston, A. (2021, November). Identifying mislabeled images in supervised learning utilizing autoencoder."

## Usage

All code is on python, and the packages used can be install through pip:

```bash
pip install package-name
```

## Main Files

There are three main notebooks to complete the full mislabel detection process:

1. Train_Autoencoder.ipynb -> Use this notebook to train and save your convolutional autoencoder. It automatically saves encoder, decoder and full model to the file path. It also applies some visualization techniques to the compressed encoded features.

2. Outlier_Detection.ipynb -> Notebook meant to detail the procedure of outlier detection and removal. It is done only for your chosen class. For the full automatized process use the full_mislabeled_removel.py file, which also gives the option to automatically apply PCA based on the amount of variance you wish to keep.

3. Simple_Classifier.ipynb -> This notebook trains a multiclass classifier on the selected dataset. It's not meant to train an accurate classifier, but rather use it as a tool to asses the quality of your dataset. Train first a model on your original dataset, and then on a new dataset resulting from the removal of the DBSCAN outliers. Keep architecture and training procedures equal for both datasets for better comparasions.

## Research Questions

1. Regarding the Autoencoder

- The architecture can still be greatly improved: how can we refine the model?
- The latent representation should be large enough for a good reconstruction while being compact enough for proper clustering. What should be the trade-off? Would a worse reconstruction, due to an aggresive dimensionalty reduction thorough the encoder, necessarily worsen the performance of the mislabels detection?
- What would be a better image resolution for this problem? Large or small?
- Are there better choices to obtain the compressed latent representation? e.g VAEs, transformers, etc.

2. Regarding the Clustering

- Are there better alternatives to DBSCAN?
- We've observed good results with a large encoded dimension followed by another dimensionality reduction technique (PCA). Why is this?
- Is PCA an optimal choice for this approach?
- Are there better heuristic methods for finding the optimal DBSCAN parameter values? Should we set them manually for each case?

3. Expanding the methodology

- So far, the autoencoder is trained on the entire dataset, and clustering is perform on each individual class (on their latent representations, to be exact). Is this optimal? would, for instance, training an autoencoder on each class independently work better? what about finding clusters on all classes simultaneously, rather than one by one?
- Could be improve the results by performing this process iteratively? i.e, performing the same autoencoder-clustering steps several times one after the other?
- What about training a binary classifier with the removed and clean datasets that learns what's a goood sample and what's not?
- Could we expand upon the current methodology by suggesting a class for each sample?
- For some applications we might need to use as much data as possible. Since the removed samples are not always wrong-labeled, a very interesting approach could be to still use the removed dataset, but downweighting it with respect to clean dataset. The weights could be, for instance, proportional to the probability that each removed sample was indeed an outlier in the feature space.





