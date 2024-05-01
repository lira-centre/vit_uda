import numpy as np
import torch
import torchvision
import time
import sklearn
import scipy
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import confusion_matrix
from sklearn.cluster import k_means, AgglomerativeClustering
from munkres import Munkres
import torchvision.transforms as transforms
from sklearn.neighbors import KernelDensity
import pickle
import sys
import geomloss
import multiprocessing.pool

RANDOM_SEED = 0
import random
random.seed (RANDOM_SEED)


MODEL='dinov2_vitg14'#"resnet152"#'dinov2_vitg14'#vit_h14_in1k'
DATASET = sys.argv[1]#'UDA_PAINTING'
TARGET_DATASET = sys.argv[2]#'UDA_SKETCH'
CLUSTERING_ALGORITHM = 'KMEANS'

LOG_CLOSEST_PROTOTYPES= True
N_CLOSEST_PROTOTYPES = 10

RESOLUTION = 224
AVG_SHOTS_PER_CLASS = 5
DEBUG = False
DEBUG_TARGET = False
def feature_extraction(img,model):
    img = img.to(device)

    feature_extractor = create_feature_extractor(
        model, return_nodes=['getitem_5'])

    with torch.no_grad():
        out = feature_extractor(img)

    return out['getitem_5']
# from https://gist.github.com/siolag161/dc6e42b64e1bde1f263b
def make_cost_matrix(c1, c2):
    """
    """
    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m
def translate_clustering(clt, mapper):
    return np.array([ mapper[i] for i in clt ])

def accuracy(cm):
    """computes accuracy from confusion matrix"""
    return np.trace(cm, dtype=float) / np.sum(cm)

def find_closest_prototypes(centroids, features):
    distance_matrix = sklearn.metrics.pairwise_distances(centroids, features)
    return np.argmin(distance_matrix,axis=1)

def find_N_closest_prototypes(centroids, features, N):
    distance_matrix = sklearn.metrics.pairwise_distances(centroids, features)
    distance_ids = []
    distances = []
    for i in range (0, distance_matrix.shape[0], 1000):
        distance_ids.append(np.argsort (distance_matrix[i:np.min([i+1000, distance_matrix.shape[0]]), :], axis = 1)[:, :N])
        distances.append(np.sort(distance_matrix[i:np.min([i+1000, distance_matrix.shape[0]]), :], axis = 1)[:, :N])
    return np.concatenate (distance_ids, axis=0), np.concatenate (distances, axis=0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(f'train_features_{MODEL}_{DATASET}.npy', 'rb') as f:
        train_features = np.load(f)
    with open(f'train_labels_{MODEL}_{DATASET}.npy', 'rb') as f:
        train_labels = np.load(f)
    with open(f'test_features_{MODEL}_{DATASET}.npy', 'rb') as f:
        test_features = np.load(f)
    with open(f'test_labels_{MODEL}_{DATASET}.npy', 'rb') as f:
        test_labels = np.load(f)
    train_labels = np.array([int (x) for x in train_labels])
    test_labels = np.array([int(x) for x in test_labels])
    if DEBUG:
        train_features = train_features[:1500, :]
        train_labels = train_labels[:1500]
      
    print ('train_features.shape: ', train_features.shape)
    print (len(np.unique(train_labels)))
    print (len(np.unique(test_labels)))

    print ('Number of unique training labels: ', len(np.unique(train_labels)))
    if CLUSTERING_ALGORITHM == 'BATCH' or CLUSTERING_ALGORITHM == 'ONLINE':
        if CLUSTERING_ALGORITHM == 'BATCH':
            n_clusters, predicted_labels = incremental_clustering.batch_clustering (train_features)
        else:
            curr_distance_list = []
            X = []
            ind_from = []
            ind_to = []
            for i in range (train_features.shape[0]):
                X, curr_distance_list, ind_from, ind_to, n_clusters, predicted_labels = incremental_clustering.incremental_clustering (X, curr_distance_list, ind_from, ind_to, [train_features[i, :]])
                print (f'{i}/{train_features.shape[0]}: n_clusters={n_clusters}')
            print ('n_clusters: ', n_clusters)
        centroids = []
        for i in range (n_clusters):
            centroids.append(np.mean(train_features[predicted_labels==i, :], axis=0))
        centroids = np.array(centroids)
    elif CLUSTERING_ALGORITHM == 'KMEANS':
        centroids, predicted_labels,_ = k_means(train_features,n_clusters=AVG_SHOTS_PER_CLASS*len(np.unique(train_labels)), random_state=RANDOM_SEED, verbose=1, n_init=1)
        n_clusters = AVG_SHOTS_PER_CLASS*len(np.unique(train_labels))

    centroids_closest_prototypes_ids = find_closest_prototypes (centroids, train_features)
    print ('find_closest_prototypes')
    centroids_ids = train_labels[centroids_closest_prototypes_ids]
    print ('centroids_ids = train_labels[centroids_closest_prototypes_ids]') 
    mapper = { i: centroids_ids[i] for i in range(len(centroids_ids))}
    predicted_labels_original = predicted_labels
    predicted_labels = translate_clustering(predicted_labels, mapper)
    print ('predicted_labels = translate_clustering(predicted_labels, mapper)')
    centroids_real_data = train_features[centroids_closest_prototypes_ids, :]
    print ('centroids_real_data = train_features[centroids_closest_prototypes_ids, :]')
    test_predicted_labels = find_closest_prototypes (test_features, centroids_real_data)
    print ('test_predicted_labels = find_closest_prototypes (test_features, centroids_real_data)')
    test_predicted_labels = translate_clustering(test_predicted_labels, mapper)
    print ('test_predicted_labels = translate_clustering(test_predicted_labels, mapper)', flush=True)

    print ('predicted_labels: ', predicted_labels)
    print ('train_labels: ', train_labels)

    print ('test_predicted_labels: ', test_predicted_labels)
    print ('test_labels', test_labels)

    print ('training: ')
    new_cm = confusion_matrix(train_labels, predicted_labels, labels=range(len(np.unique(train_labels))))
    print ("---------------------\nnew confusion matrix:\n" \
      " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)))

    print ('testing: ')
    new_cm = confusion_matrix(test_labels, test_predicted_labels, labels=range(len(np.unique(train_labels))))
    print ("---------------------\nnew confusion matrix:\n" \
           " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)), flush=True)

    if not (TARGET_DATASET is  None):
        with open(f'train_features_{MODEL}_{TARGET_DATASET}.npy', 'rb') as f:
            train_features_target = np.load(f)
        with open(f'train_labels_{MODEL}_{TARGET_DATASET}.npy', 'rb') as f:
            train_labels_target = np.load(f)
        with open(f'test_features_{MODEL}_{TARGET_DATASET}.npy', 'rb') as f:
            test_features_target = np.load(f)
        with open(f'test_labels_{MODEL}_{TARGET_DATASET}.npy', 'rb') as f:
            test_labels_target = np.load(f) 
        train_labels_target = np.array([int (x) for x in train_labels_target])
        test_labels_target = np.array([int(x) for x in test_labels_target])
        if DEBUG_TARGET:
            train_features_target = train_features_target [:1500, :]
            train_labels_target = train_labels_target[:1500]
        print ('Number of unique training labels: ', len(np.unique(train_labels)), flush=True)
        if CLUSTERING_ALGORITHM == 'BATCH' or CLUSTERING_ALGORITHM == 'ONLINE':
            if CLUSTERING_ALGORITHM == 'BATCH':
                n_clusters_target, predicted_labels_target = incremental_clustering.batch_clustering (train_features_target)
            else:
                curr_distance_list_target = []
                X_target = []
                ind_from_target = []
                ind_to_target = []
                for i in range (train_features_target.shape[0]):
                    X_target, curr_distance_list_target, ind_from_target, ind_to_target, n_clusters_target, predicted_labels_target = incremental_clustering.incremental_clustering (X_target, curr_distance_list_target, ind_from_target, ind_to_target, [train_features_target[i, :]])
                    print (f'{i}/{train_features_target.shape[0]}: n_clusters_target={n_clusters_target}', flush=True)
                print ('n_clusters_target: ', n_clusters_target)
            centroids_target = []
            for i in range (n_clusters_target):
                centroids_target.append(np.mean(train_features_target[predicted_labels_target==i, :], axis=0))
            centroids_target = np.array(centroids_target)
        elif CLUSTERING_ALGORITHM == 'KMEANS':
            centroids_target, predicted_labels_target,_ = k_means(train_features_target,n_clusters=AVG_SHOTS_PER_CLASS*len(np.unique(train_labels_target)), random_state=RANDOM_SEED, verbose=1, n_init=1)
            n_clusters_target = AVG_SHOTS_PER_CLASS*len(np.unique(train_labels_target)) 
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_features_cuda = torch.tensor(train_features, device=device).float()
        test_features_cuda = torch.tensor(test_features, device=device).float()
        kde = {}
        for l in range(n_clusters):
            train_features_l = train_features[predicted_labels_original == l, :]
            kde[l] = train_features_l

        train_features_target_cuda = torch.tensor (train_features_target, device=device).float()
        test_features_target_cuda = torch.tensor (test_features_target, device=device).float()
        kde_target = {}
        for l in range(n_clusters_target):
            train_features_target_l = train_features_target[predicted_labels_target == l, :]
            kde_target[l] = train_features_target_l
        
        wasserstein_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1e-5)
        y_mean = []
        with torch.no_grad():
            for y in kde_target:
                y_mean.append(torch.mean (torch.tensor (kde_target[y], device='cpu').float(), axis = 0))
            y_mean = torch.stack (y_mean)
        with torch.no_grad():
            x_mean = []
            for x in kde:
                x_mean.append(torch.mean (torch.tensor(kde[x], device='cpu').float(), axis = 0))
            x_mean = torch.stack (x_mean)
            wasserstein_matrix = []
            wasserstein_matrix_2 = []
            for i in range (x_mean.shape[0]):
                wasserstein_matrix.append(torch.cdist(x_mean[i:i+1, :],y_mean).detach().numpy())
            wasserstein_matrix = np.concatenate (wasserstein_matrix,axis=0)
        
        wasserstein_matrix_2 = np.zeros ([n_clusters, n_clusters_target])
        with torch.no_grad():
            def nested_loop_func (i):
                print (f'{i}/{n_clusters}', flush=True)
                train_features_l = train_features_cuda[predicted_labels_original == i, :]
                matrix = wasserstein_matrix_2[i, :]
                for j in range (n_clusters_target):
                    train_features_target_l = train_features_target_cuda[predicted_labels_target == j, :]
                    try:
                        matrix[j] = wasserstein_loss (train_features_l, train_features_target_l).item()
                    except:
                        matrix[j] = 100000
                        print (f'train_features_l: {train_features_l}', flush=True)
                        print (f'train_features_target_l: {train_features_target_l}', flush=True)
                        print (f'torch.min (train_features_l): {torch.min (train_features_l)}', flush=True)
                        print (f'torch.max (train_features_l): {torch.max (train_features_l)}', flush=True)
                        print (f'torch.min (train_features_target_l): {torch.min (train_features_target_l)}', flush=True)
                        print (f'torch.max (train_features_target_l): {torch.max (train_features_target_l)}', flush=True)
                return matrix
            with multiprocessing.Pool(processes=32) as pool:
                wasserstein_matrix_2 = np.array(list(pool.map(nested_loop_func, range(n_clusters))))
        closest_cluster_target_wasserstein = np.argmin (wasserstein_matrix_2, axis=0)
        mapper_target_wasserstein = { i: centroids_ids[closest_cluster_target_wasserstein[i]] for i in range(len(closest_cluster_target_wasserstein))}
        
        closest_cluster_target = np.argmin (wasserstein_matrix, axis=0)
        mapper_target = { i: centroids_ids[closest_cluster_target[i]] for i in range(len(closest_cluster_target))}

        print ('~~~~~~~~~')
        print ('L2 matrix between clusters: ')
        print (str(wasserstein_matrix))
        np.save (f"domain_{MODEL}_{DATASET}_{TARGET_DATASET}_L2_matrix.npy", wasserstein_matrix)
        print ('~~~~~~~~~~~')
        print ('Wasserstein matrix between clusters: ')
        print (str(wasserstein_matrix_2), flush=True)
        np.save (f"domain_{MODEL}_{DATASET}_{TARGET_DATASET}_Wasserstein_matrix.npy", wasserstein_matrix_2)
        num_classes = len(np.unique(train_labels)) 
        classes = np.unique(train_labels)
        
        wasserstein_matrix_classes = np.zeros ((num_classes, num_classes))
        with torch.no_grad():
            for i in range (num_classes):
                test_features_l = torch.tensor (test_features[test_labels == classes[i], :], device = device).float()
                print (i, flush=True)
                if len(test_features_l) == 0:
                    continue
                for j in range (num_classes):
                    test_features_target_l = torch.tensor (test_features_target[test_labels_target == classes[j], :], device = device).float()
                    if len(test_features_target_l) == 0:
                        continue
                    wasserstein_matrix_classes[i, j] = wasserstein_loss (test_features_l, test_features_target_l).item()
        print ('~~~~~~~~~')
        print ('Wasserstein matrix between classes: ')
        print (wasserstein_matrix_classes)
        np.save (f"domain_{MODEL}_{DATASET}_{TARGET_DATASET}_Wasserstein_matrix_classes.npy", wasserstein_matrix_classes)

        predicted_labels_target_l2 = translate_clustering(predicted_labels_target, mapper_target)
        predicted_labels_target_wasserstein = translate_clustering(predicted_labels_target, mapper_target_wasserstein)
        
        centroids_closest_prototypes_ids_target = find_closest_prototypes (centroids_target, train_features_target)
        centroids_real_data_target = train_features_target[centroids_closest_prototypes_ids_target, :]
        
        test_predicted_labels_target = find_closest_prototypes (test_features_target, centroids_real_data_target)
        test_predicted_labels_target_l2 = translate_clustering(test_predicted_labels_target, mapper_target)
        test_predicted_labels_target_wasserstein = translate_clustering(test_predicted_labels_target, mapper_target_wasserstein)


        if LOG_CLOSEST_PROTOTYPES:
            centroids_closest_prototypes_ids_target_train, distances_closest_prototypes_ids_target_train = find_N_closest_prototypes (train_features_target, centroids_real_data_target, N_CLOSEST_PROTOTYPES)
            for i in range(centroids_closest_prototypes_ids_target_train.shape[0]):
                for j in range(centroids_closest_prototypes_ids_target_train.shape[1]):
                    centroids_closest_prototypes_ids_target_train[i][j] = centroids_closest_prototypes_ids_target[centroids_closest_prototypes_ids_target_train[i][j]]

            centroids_closest_prototypes_ids_train, distances_closest_prototypes_ids_train = find_N_closest_prototypes (train_features_target, centroids_real_data, N_CLOSEST_PROTOTYPES)
            for i in range(centroids_closest_prototypes_ids_train.shape[0]):   
                for j in range(centroids_closest_prototypes_ids_train.shape[1]):
                    centroids_closest_prototypes_ids_train[i][j] = centroids_closest_prototypes_ids[centroids_closest_prototypes_ids_train[i][j]]

            centroids_closest_prototypes_ids_target_test, distances_closest_prototypes_ids_target_test = find_N_closest_prototypes (test_features_target, centroids_real_data_target, N_CLOSEST_PROTOTYPES)
            for i in range(centroids_closest_prototypes_ids_target_test.shape[0]):   
                for j in range(centroids_closest_prototypes_ids_target_test.shape[1]):
                    centroids_closest_prototypes_ids_target_test[i][j] = centroids_closest_prototypes_ids_target[centroids_closest_prototypes_ids_target_test[i][j]]

            centroids_closest_prototypes_ids_test, distances_closest_prototypes_ids_test = find_N_closest_prototypes (test_features_target, centroids_real_data, N_CLOSEST_PROTOTYPES)
            for i in range(centroids_closest_prototypes_ids_test.shape[0]):          
                for j in range(centroids_closest_prototypes_ids_test.shape[1]):
                    centroids_closest_prototypes_ids_test[i][j] = centroids_closest_prototypes_ids[centroids_closest_prototypes_ids_test[i][j]]

            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_target_train', centroids_closest_prototypes_ids_target_train)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_train', centroids_closest_prototypes_ids_train)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_target_test', centroids_closest_prototypes_ids_target_test)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_test', centroids_closest_prototypes_ids_test)

            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_target_train_d', distances_closest_prototypes_ids_target_train)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_train_d', distances_closest_prototypes_ids_train)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_target_test_d', distances_closest_prototypes_ids_target_test)
            np.save (f'domain_{MODEL}_{DATASET}_{TARGET_DATASET}_closest_prototypes_test_d', distances_closest_prototypes_ids_test)

        print ('predicted_labels_target_l2: ', predicted_labels_target_l2, flush=True)
        print ('predicted_labels_target_l2: ', predicted_labels_target_wasserstein, flush=True)
        print ('train_labels_target: ', train_labels_target)

        print ('test_predicted_labels_target_l2: ', test_predicted_labels_target_l2)
        print ('test_predicted_labels_target_wasserstein: ', test_predicted_labels_target_wasserstein)
        print ('test_labels_target', test_labels)

        print ('training_target (l2): ')
        new_cm = confusion_matrix(train_labels_target, predicted_labels_target_l2, labels=range(len(np.unique(train_labels_target))))
        print ("---------------------\nnew confusion matrix:\n" \
               " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)))

        print ('training_target (Wasserstein): ')
        new_cm = confusion_matrix(train_labels_target, predicted_labels_target_wasserstein, labels=range(len(np.unique(train_labels_target))))
        print ("---------------------\nnew confusion matrix:\n" \
               " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)))

        print ('testing_target (l2): ')
        new_cm = confusion_matrix(test_labels_target, test_predicted_labels_target_l2, labels=range(len(np.unique(train_labels_target))))
        print ("---------------------\nnew confusion matrix:\n" \
             " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)), flush=True)
        print ('testing_target (Wasserstein): ')
        new_cm = confusion_matrix(test_labels_target, test_predicted_labels_target_wasserstein, labels=range(len(np.unique(train_labels_target))))
        print ("---------------------\nnew confusion matrix:\n" \
             " %s\naccuracy: %.4f" % (str(new_cm), accuracy(new_cm)), flush=True)
