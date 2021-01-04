import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as cluster


def predict_all(model: tf.keras.Model, dataset, embedding_dim):
    """
    model: tensorflow model
    dataset: callable object generating batched examples
    """
    
    batch_embeddings = []
    batch_labels = []

    # collect all results
    # calculate number of examples
    num_examples = 0
    for n, (x, y) in enumerate(dataset):
        em = model(x).numpy()
        batch_embeddings.append(em)
        batch_labels.append(y)
        num_examples += em.shape[0]

    embeddings = np.zeros((num_examples, embedding_dim))
    labels = np.zeros(num_examples)
    for n, (em, y) in enumerate(zip(batch_embeddings, batch_labels)):
        sample_size = em.shape[0]
        embeddings[n*sample_size:(n+1)*sample_size,:] = em
        labels[n*sample_size:(n+1)*sample_size] = y

    return embeddings, labels


def evaluate(model, dataset, embedding_dim, classes):
    # calculate embeddings with model and get targets
    embeddings, labels = predict_all(model, dataset, embedding_dim)

    # calculate NMI with kmeans clustering
    clustered_labels = KMeans(classes).fit(embeddings).labels_
    nmi_score = cluster.normalized_mutual_info_score(clustered_labels, labels)
    return nmi_score
