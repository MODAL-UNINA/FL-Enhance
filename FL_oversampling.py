# %%

# import the necessary packages

import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
# import opencv as cv2
import os
# from imutils import paths
from sklearn.model_selection import train_test_split
import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd
from numpy.random import choice
from collections import Counter
# import packages for the oversampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, ADASYN
# import warnings
# warnings.filterwarnings("ignore")
# from time import time

debug=0
from FL_utils import *

mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['axes.grid'] = False

# %%
# # see physical devices
# tf.config.list_physical_devices('GPU')
# tf.config.list_physical_devices()

# # connect to GPU:0 if available
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

# GPU:0 is available, so we can use it
# tf.test.gpu_device_name()


CVD = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = CVD
# Some other determinism...
SEED = 3
os.environ['PYTHONHASHSEED'] = str(SEED)

# # check if I am using GPU
# tf.test.is_gpu_available()


random.seed(SEED)
np.random.seed(SEED)

# %%

# load the MNIST dataset from internet 

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data() # 60000 training images, 10000 test images

trainX.shape
type(trainX)


X = np.concatenate((trainX, testX), axis=0)
y = np.concatenate((trainY, testY), axis=0)

#apply our function
image_list, label_list = X, y


#binarize the labels
# lb = LabelBinarizer()
# label_list = lb.fit_transform(label_list)

image_data = []
label_data = []

for i in tqdm.tqdm(range(len(image_list))):
    #convert to grayscale
    # img_gray = rgb2gray(dataset[i][0])
    # #flatten and normalize
    # image = (np.array(img_gray).flatten()) / 255.0
    image_data.append(image_list[i].flatten()/255) # flatten the image (784,), normalize and append
    label_data.append(label_list[i])

# label_data = lb.fit_transform(label_data)

# plt.imshow(image_data[0].reshape(28,28), cmap='gray')
# label_data[0].argmax()


# %% 

X_train, X_test, y_train, y_test = train_test_split(image_data, 
                                                    label_data, 
                                                    test_size=0.1, 
                                                    random_state=42)



len(X_train), len(X_test), len(y_train), len(y_test)

# %%


clients = create_shards_function(X_train, y_train, no_clients=20, 
no_classes=10, niid=True, balance=False, 
partition='pathological', class_per_client=6, initial='clients')

for client, data in clients.items():
    print(pd.Series(data[1]).value_counts())


# %%

# clients.keys()
# clients['clients_0'][0].shape, clients['clients_0'][1].shape
# len(clients['clients_0'][1]) # 5363
# create dummies for the labels in clients  

# create dummies on "longest data"
# y_test
# y_test_lb = lb.fit_transform(y_test)
y_test_dummy = pd.get_dummies(y_test)
# %% 
# create dummies on train fitting the holes according to the testù

# def batch_data(data, bs=32):

#     # put in data, label zip the values from the clients dictionary
#     train = tf.data.Dataset.from_tensors(data[0])
#     label = tf.data.Dataset.from_tensors(data[1])
#     # label = lb.fit_transform(label)
#     # dataset =  tf.data.Dataset.from_tensor_slices((list(data), list(label)))
#     dataset =  tf.data.Dataset.zip((train, label))
    
#     return dataset.shuffle(len(data)).batch(bs)


# # %%
# clients_trn_data = clients_train.copy()
clients_train = dict()
clients_label = dict()
for client_name, data in clients.items(): 
    clients_train[client_name] = data[0]
    clients_label[client_name] = pd.get_dummies(data[1]).reindex(columns=y_test_dummy.columns, fill_value=0)

test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test_dummy)).batch(len(y_test_dummy))




# %% 
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model      


# %% 


# %%

from scipy import sparse
import sys



# %% 

lr = 0.01
comm_rounds = 50
loss='categorical_crossentropy'
metrics = ['categorical_accuracy']
optimizer = SGD(lr=lr, 
                decay=lr / comm_rounds, 
                momentum=0.9)  


#commence global training loop

#initialize global model

build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST

smlp_global = SimpleMLP()
global_model = smlp_global.build(build_shape, 10) 
global_acc_list = []
global_loss_list = []
global_time_list = []
total_training_time = []

# %%

random.seed(42)

k_list = []
clients_list = []

for comm_round in range(comm_rounds):
    all_client_names = list(clients_train.keys())
    k=random.randint(10, len(all_client_names)/2)

    # select a random sample of clients between 10 and len(all_client_names)
    client_names = random.sample(all_client_names, k=k)
    # print(k, client_names)

    # client_names = random.sample(all_client_names, k=10)
    # I randomly select 10 clients for each round - 
    # print(client_names, len(client_names))
    random.shuffle(client_names)

    k_list.append(k)
    clients_list.append(client_names)


# %%

for comm_round in range(comm_rounds): 


    start_time = time.time()
    # compute time for each round

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    
    #randomize client data - using keys
    # all_client_names = list(clients_train.keys())
    # k=random.randint(10, len(all_client_names)/2)
    # # set seed for reproducibility
    # random.seed(42)
    # # select a random sample of clients between 10 and len(all_client_names)
    # client_names = random.sample(all_client_names, k=k)
    # print(k, client_names)

    # # client_names = random.sample(all_client_names, k=10)
    # # I randomly select 10 clients for each round - 
    # # print(client_names, len(client_names))
    # random.shuffle(client_names)
    # select random number of clients for each round
    #print selected number of clients
    # print('The selected clients for round {} are: {}'.format(comm_round, client_names))
    # print('The number of clients for round {} is: {}'.format(comm_round, len(client_names)))

    # if debug: 
    #     # print('all_client_names', all_client_names)
    #     print('client_names', client_names, len(client_names))
                
    # clients_involved.append(len(client_names))
    k = k_list[comm_round]
    client_names = clients_list[comm_round]

    compressed_data = dict()
    compressed_data_sparse = dict()
    label_distribution = dict()
    data_to_send_to_server = dict()
    frequency = dict()
    frequency2 = dict()
    most_frequent_labels = dict()
    df = dict()
    u = dict()
    share_list = dict()
    total_for_df_round = []

    samples_to_create = []

    # each client double compresses its data and converts it into a sparse matrix
    for client in client_names: #break
        label_distribution[client] = clients_label[client].sum(axis=0)
    
    # totale di ciascun label per round
    total_labels_round = np.sum([label_distribution[client] for client in client_names], axis=0)
    
    relative_total = np.round(total_labels_round/total_labels_round.sum(), 3)

    # check distance between the most full client and the least full client
    # print('The distance between the most full client and the least full client is: {}'.format(np.max(total_labels_round)-np.min(total_labels_round)))

    # samples_to_create = np.max(total_labels_round)-np.min(total_labels_round)

    
    # max = np.max(total_labels_round)
    # samples_to_create = [max - total_labels_round[i] for i in range(len(total_labels_round))]

    

    # plot label distribution for each client
    # for client in client_names:
        # plt.figure(figsize=(10, 5))
        # plt.bar(label_distribution[client].index, label_distribution[client].values)
        # plt.title('Label Distribution for Client {}'.format(client))
        # plt.xlabel('Label')
        # plt.ylabel('Frequency')
        # plt.show()

    # print I was here
    # print('I was hereeeeeeeeeeeee!')

    # frequency2 = [(label_distribution[client]/total_labels_round)*100 for client in client_names]

    for client in client_names:
        # frequency[client] = (label_distribution[client]/label_distribution[client].sum())*100

        # sum total labels for each class for all clients in the round from label distribution
        # total_labels_round = np.sum([label_distribution[client] for client in client_names], axis=0)
        
        # label distribution per client per class divided the total amount of labels per class in that round
        frequency2[client] = (label_distribution[client]/total_labels_round)

    df = pd.DataFrame(frequency2).T

    share_list = dict()
    # create a vector named share list with frequency product
    for client in client_names:
        share_list[client] = (np.round(frequency2[client], 3)*label_distribution[client]).astype('int')


    # cluster together clients with different label distribution for each class so to have a more balanced dataset  

    # df.sum(axis=0)

    # # highlight the clients with the most frequent labels 
    # df.style.highlight_max(color = 'lightgreen', axis = 0)

    # # # select the clients with the most frequent labels - greater than 10%
    # df[df > 0.05]

    # pick the name of the clients with the most frequent labels and the most frequent labels

    # create a dictionary with the name of the clients and the most frequent labels
    # client_frequency_threshold = len(client_names)/100

    # for client in client_names:
    #     most_frequent_labels[client] = df[df > client_frequency_threshold].loc[client].dropna().index.tolist()
    #     # consider clients with at least 5% of the most frequent labels
    
    # take from df only clients which are in most freuent labels
    # df = df[df.index.isin(most_frequent_labels.keys())]

    # # print the most frequent labels for each client
    # most_frequent_labels
    # most_frequent_labels.keys()
    # most_frequent_labels.values()

    # put all values in a list and check the number of unique values
    u, _ = np.unique([item for sublist in most_frequent_labels.values() for item in sublist], return_counts=True)

    # check that u equals the number of classes
    assert len(u) == df.shape[1]

    # from most_frequent_labels, select random samples of clients with associated labels 
    # according to their frequency in frquency2

    # create a share_list where each values is the label_distribution for each client multiplied by frequency2 

    # for client in client_names:
    #     share_list[client] = (label_distribution[client]*frequency2[client]).astype(int)
    
    # print the proportion of data each client is sending to the server
    # divide share list by the total amount of data for each client

    # for client in client_names:
    #     print('The proportion of data for client {} is: {}'.format(client, share_list[client].sum()/len(clients_train[client])))

    # if some proportion is greater than 0.5, reduce the proportion to 0.5
    # for client in client_names:
    #     if share_list[client].sum()/len(clients_train[client]) > 0.5:
    #         share_list[client] = ((share_list[client]/share_list[client].sum())*0.5*len(clients_train[client])).astype(int)
        
    # share_list = share_list.to_list()

    # pick random samples from clients_train for each client according to the share_list
    # for each label, pick the number of samples indicated in share_list for that client 
    # and append them to data_to_send_to_server together with the labels
    
    clients_df = dict()
    for client in client_names:
        client_train = clients_train[client]
        client_X = [client_train[i] for i in range(len(client_train))]
        s_client_X = pd.Series(client_X, name='X') 

        client_label = clients_label[client]
        client_y = [client_label.iloc[i].values for i in range(len(client_label))]

        s_client_y = pd.Series(client_y, name='y')

        s_client_label = s_client_y.apply(lambda x: np.argmax(x)).rename('label')

        clients_df[client] = pd.concat([s_client_X, s_client_y, s_client_label], axis=1)

    df_to_send_to_server = dict()
    clients_update = list()

    for client in client_names:
        client_share_list = share_list[client]
        # if it is empty, skip the client
        if client_share_list.sum() == 0:
            continue
        # else if it is not empty, pick the samples
        df_to_send_to_server[client] = []
        for label in client_share_list.index.tolist():
            df_data_client_label = clients_df[client][clients_df[client]['label'] == label]
            df_to_send_to_server[client].append(df_data_client_label.sample(n=client_share_list[label], random_state=42))
        
        clients_update.append(client)
        df_to_send_to_server[client] = pd.concat(df_to_send_to_server[client], axis=0).sample(frac=1, random_state=42)

        # plot label distribution for clients_8 in df_to_send_to_server
        df_to_send_to_server['clients_8']['label'].value_counts()
        
    # total elements in df_to_send_to_server for each client
    somma = []
    for client in clients_update:
        print('The total elements in df_to_send_to_server for client {} is: {}'.format(client, len(df_to_send_to_server[client])))
        somma.append(len(df_to_send_to_server[client]))

    # sum elements in somma list
    np.sum(somma)

    np.sum(somma) / total_labels_round.sum()


    

    # double compress the data to send to the server
    
    # access X column of df_to_send_to_server
    # df_to_send_to_server['clients_0']['X'].iloc[0].shape
    # type(df_to_send_to_server['clients_0']['X'].iloc[0])

    # print memory occupied by df_to_send_to_server for each client

    # for client in clients_update:
    #     print('The memory occupied by df_to_send_to_server for client {} is: {} MB'.format(client, df_to_send_to_server[client].memory_usage(index=True).sum()/1024**2))

    # double compress onle the X column of df_to_send_to_server
    # instantiate compressed_data as copy of df_to_send_to_server except for X column 
    compressed_data = dict()
    for client in clients_update:
        compressed_data[client] = df_to_send_to_server[client].copy()

    for client in clients_update:
        assert df_to_send_to_server[client].equals(compressed_data[client])

    # image = df_to_send_to_server['clients_0']['X'].iloc[0]
    # plt.imshow(image.reshape(28,28))

    for client in clients_update:
        compressed_data[client]['X'] = df_to_send_to_server[client]['X'].apply(lambda x: double_compress_images(x, 0.2, 0.1))
    
    # image_comp = compressed_data['clients_0']['X'].iloc[0]
    # plt.imshow(image_comp.reshape(28,28))

    for client in clients_update:
        assert not  df_to_send_to_server[client].equals(compressed_data[client])

    # print memory occupied by compressed_data for each client
    # for client in clients_update:
    #     print('The memory occupied by compressed_data for client {} is: {} MB'.format(client, compressed_data[client].memory_usage(index=True).sum()/1024**2))

    # NOTHING CHANGED - BECAUSE I ONLY REMOVED PIXELS, I SHOULD TRY AFTER CSR MATRIX
    
    # plot 1 original and 1 compressed image for each client
    # for client in clients_update:
    #     fig, ax = plt.subplots(1,2, figsize=(5,5))
    #     ax[0].imshow(df_to_send_to_server[client]['X'].iloc[0].reshape(28,28), cmap='gray')
    #     ax[1].imshow(compressed_data[client]['X'].iloc[0].reshape(28,28), cmap='gray')
    #     plt.show()

    # assert the no of zero values in original data is less than the no of zero values in compressed data for each image in the client's dataset
    for client in clients_update:
        for i in range(len(df_to_send_to_server[client])):
            assert np.count_nonzero(df_to_send_to_server[client]['X'].iloc[i]) > np.count_nonzero(compressed_data[client]['X'].iloc[i])

    # check new label distribution for each client
    # for client in clients_update:
    #     print('The new label distribution for client {} is:'.format(client))
    #     print(compressed_data[client]['label'].value_counts())

    # # check how many samples per label I need in order to balance all the clients
    # for client in client_names:
    #     print('The no of samples per label for client {} is:'.format(client))
    #     print(compressed_data[client]['label'].value_counts().min())

    
    # print done so far the compression
    # print('Done so far the compression for client {}'.format(client))

    # convert the compressed data into a sparse matrix

    compressed_data_sparse = dict()
    for client in clients_update:
        compressed_data_sparse[client] = compressed_data[client].copy()

        compressed_data_sparse[client]['X'] = compressed_data_sparse[client]['X'].apply(lambda x: sparse.csr_matrix(x))
        

    # compare memory usage of the compressed sparse and original data
    # print('The memory usage of the original data is: {} Kbytes'.format(sys.getsizeof(df_to_send_to_server[client])/1000))
    # print('The memory usage of the compressed data is: {} Kbytes'.format(sys.getsizeof(compressed_data[client])/1000))
    # print('The memory usage of the sparse matrix is: {} Kbytes'.format(sys.getsizeof(compressed_data_sparse[client])/1000))


    # put together all the data from all the clients in a single dataframe
    df_to_oversample = pd.concat(compressed_data_sparse.values(), axis=0)

    assert len(df_to_oversample) == np.sum([share_list[client].sum() for client in client_names])


    # type(df_to_oversample) # df_to_GAN is a dataframe with 3 columns: X, y, label
    # # check the type of the columns
    # type(df_to_oversample['X']) # series
    # type(df_to_oversample['y']) # series
    # type(df_to_oversample['label']) # series

    # # check the type of the elements of the columns
    # type(df_to_oversample['X'].iloc[0]) # csr matrix
    # type(df_to_oversample['y'].iloc[0]) # numpy.ndarray
    # type(df_to_oversample['label'].iloc[0]) # numpy.int64

    # create a new dataframe with X to convert and labels
    # df_to_GAN_final = pd.DataFrame(index=df_to_GAN.index)#, columns=['X', 'label'])
    df_to_oversample_final = df_to_oversample.copy()

    # create a new column with X converted to numpy array
    df_to_oversample_final = df_to_oversample_final.drop(columns=['y'])

    df_to_oversample_final['X'] = df_to_oversample_final['X'].apply(lambda x: x.toarray()[0])
    # df_to_oversample_final.loc[106, 'X'].shape # (60000, 1)
    # print(type(df_to_oversample_final.loc[31, 'X']))
    # # remove the second dimension
    

    # len(df_to_oversample_final.iloc[0]) # 2
    # len(df_to_oversample_final['X'].iloc[0]) 
    # len(df_to_oversample_final['label'])
    # len(df_to_oversample_final.iloc[0][0]) # 784
    # df_to_oversample_final.iloc[0][1] # label 

    # df_to_oversample_final['X'].iloc[106].shape # (784,)

    # check label distribution in df_to_overample_final
    # df_to_oversample_final['label'].value_counts()  
    # 7    1473
    # 8    1238
    # 6    1225
    # 9    1220
    # 1      71
    # 2      64
    # 3      64
    # 0      55
    # 4      55
    # 5      51

    # Since the distribution of classes in the original data is uneven, sampling methods in imbalanced-learn python package are used.
    # The sampling methods are used to balance the dataset and to create new samples for the minority classes.
   

    # convert df_to_oversample_final['X'] to shape (n_samples, n_features)
    X_to_oversample = np.array(df_to_oversample_final['X'].tolist())

    #check how many sample per label are in X_to_oversample
    print(sorted(Counter(df_to_oversample_final['label']).items()))

    # if imbalance == 'RandomOverSampler':
        # RandomOverSampler

    # create a dictionary with key classes and values samples to create
    # sampling_strategy = {0: samples_to_create, 1: samples_to_create, 2: samples_to_create, 3: samples_to_create, 4: samples_to_create, 5: samples_to_create, 6: samples_to_create, 7: samples_to_create, 8: samples_to_create, 9: samples_to_create}
    
    # create sampling strategy for each label 
   
    
    
    # ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    # X_resampled, y_resampled = ros.fit_resample(X_to_oversample, df_to_oversample_final['label'])
    # print(sorted(Counter(y_resampled).items()))

    # # see some oversampled samples for minority classes with their labels
    # plt.figure(figsize=(10, 10))
    # for i in range(10):
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(X_resampled[i].reshape(28, 28), cmap='gray')
    #     plt.title('Label: {}'.format(y_resampled[i]))
    #     plt.axis('off')

    # # check if there are duplicates in the oversampled data 
    # # (if there are duplicates, the oversampling method is not working properly)
    # X_resampled_df = pd.DataFrame(X_resampled)
    # X_resampled_df.duplicated().sum() # 25384

    # check similar samples between X_resampled_df and df_to_oversample_final['X']
    # (if there are similar samples, the oversampling method is not working properly)

    _, s = np.unique(df_to_oversample_final['label'], return_counts=True)

    perc = s/total_labels_round
    # modify total_labels_round so to subtract q1 for left element 

    samples_to_create = ((1-perc)*total_labels_round).astype('int')
    samples_to_create = ((1-perc)*s + s).astype('int')

    samples_to_create_s = ((1-perc)*s).astype('int')


    sampling_strategy = {
    0: samples_to_create[0], 1: samples_to_create[1], 2: samples_to_create[2], 3: samples_to_create[3], 4: samples_to_create[4], 5: samples_to_create[5], 6: samples_to_create[6], 7: samples_to_create[7], 8: samples_to_create[8], 9: samples_to_create[9]
    }

    # elif imbalance == 'SMOTE':
    # SMOTE
    sm = SMOTE(random_state=0, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled= sm.fit_resample(X_to_oversample, df_to_oversample_final['label'])
    print(len(y_resampled))

    n_generated = len(y_resampled) - len(X_to_oversample)
    assert np.all(X_to_oversample[:len(X_to_oversample)] == X_resampled[:len(X_to_oversample)])

    X_resampled = X_resampled[len(X_to_oversample):]
    y_resampled = y_resampled[len(X_to_oversample):]

    # ADASYN
    ada = ADASYN(random_state=40, sampling_strategy=sampling_strategy)
    X_resampled_adasyn2, y_resampled_adasyn2 = ada.fit_resample(X_to_oversample, df_to_oversample_final['label'])

    X_resampled_adasyn2 = X_resampled_adasyn2[len(X_to_oversample):]
    y_resampled_adasyn2 = y_resampled_adasyn2[len(X_to_oversample):]


    # plot 10 original 

    # print(sorted(Counter(y_resampled).items()))

    # # see some oversampled samples with their labels
    # plt.figure(figsize=(10, 10))
    # for i in range(len(X_to_oversample), len(X_to_oversample)+10):
    #     plt.subplot(5, 5, i+1-len(X_to_oversample))
    #     plt.imshow(X_resampled[i].reshape(28, 28), cmap='gray')
    #     plt.title('Label: {}'.format(y_resampled[i]))
    #     plt.axis('off')

    
    # plt.figure(figsize=(10, 10))
    # for i in range(10):
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(X_to_oversample[i].reshape(28, 28), cmap='gray')
    #     plt.title('Label: {}'.format(df_to_oversample_final['label'].iloc[i]))
    #     plt.axis('off')
        

    AAtr_ge = NNAA(X_to_oversample, X_resampled)

    AAtE_ge = NNAA(Rte, X_SMOTE)

    privacy_loss = np.mean(AAtE_ge - AAtr_ge)


    X_resampled_df = pd.DataFrame(X_resampled)
    # X_resampled_SMOTE_df.duplicated().sum() # 0
    y_resampled_df = pd.DataFrame(y_resampled)
    # check if X_resampled and X_resampled_SMOTE are the same
    # (if they are the same, the oversampling method is not working properly)

    # X_resampled_df.equals(X_resampled_SMOTE_df) # False
    # this means that the two oversampled data are different

    # distribute the oversampled data to the clients so that they are more balaned
    # starting fron the client with the least data points

    # distribute the oversampled data to the clients in client_names
    # so that they are more balaned
    # starting fron the client with the least data points
    # create a dictionary with key clients in client_names and values nothing 

    clients_train_new = dict.fromkeys(client_names)

    clients_labels_new = dict.fromkeys(client_names)

    for client in client_names:
        # get the number of data points for the client
        num_data_points = len(clients_train[client]) # sum of the number of data points for each client
        # get the number of data points to be added to the client
        num_data_points_to_add = int(n_generated * (num_data_points / total_labels_round.sum()))
        # print(num_data_points_to_add, (num_data_points / total_labels_round.sum()))
        # get the data points to be added to the client in a random order
        data_points_to_add = X_resampled_df.sample(n=num_data_points_to_add, random_state=0)        
        # data_points_to_add = X_resampled_SMOTE[:num_data_points_to_add]
        # add the data points to the client
        clients_train_new[client] = np.concatenate((clients_train[client], data_points_to_add))
        # remove the data points that have been added to the client
        X_resampled_df = X_resampled_df.drop(data_points_to_add.index)
        # X_resampled_SMOTE = X_resampled_SMOTE[num_data_points_to_add:]
        # get the labels to be added to the client
        labels_to_add = y_resampled_df.sample(n=num_data_points_to_add, random_state=0)
        # add the labels to the client
        clients_labels_new[client] = np.concatenate((clients_label[client], pd.get_dummies(labels_to_add['label']).reindex(columns=y_test_dummy.columns, fill_value=0)))
        y_resampled_df = y_resampled_df.drop(labels_to_add.index)
        # check len of clients_train[client] and clients_label[client]
        # print('len of clients_train[client]', len(clients_train_new[client]))
        # print('len of clients_label[client]', len(clients_label[client]))

        # for the first 10 clients, compare the original data points and the new data points
        # with barplots to see if the oversampling method is working properly
        # for client in client_names[:10]:
        #     # original data points
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.bar(np.arange(10), clients_label[client].sum(axis=0))
        #     plt.title('Original data points for client {}'.format(client))
        #     plt.xticks(np.arange(10), np.arange(10))
        #     # new data points
        #     plt.subplot(1, 2, 2)
        #     plt.bar(np.arange(10), clients_labels_new[client].sum(axis=0))
        #     plt.title('New data points for client {}'.format(client))
        #     plt.xticks(np.arange(10), np.arange(10))
        #     plt.show()

        # check that there are no duplicated data points among the clients
        # for client in client_names:
        #     print('duplicated data points for client {}: {}'.format(client, pd.DataFrame(clients_train_new[client]).duplicated().sum()))

    

        # for client_name, data in clients.items(): 
        #     clients_train[client_name] = data[0]
        #     clients_label[client_name] = pd.get_dummies(data[1]).reindex(columns=y_test_dummy.columns, fill_value=0)

        # test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test_dummy)).batch(len(y_test_dummy))

    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(build_shape, 10)
        local_model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=metrics)
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_train_new[client], clients_labels_new[client], 
                epochs=1, verbose=0, batch_size=32)
        
        #scale the model weights and add to list
        scaling_factor =  weight_scalling_factor(clients_train_new, client)
        # print('scaling_factor', scaling_factor)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)


    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss, global_time_taken = test_model(X_test, Y_test, global_model, comm_round)
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)
        global_time_list.append(global_time_taken)


    end_time = time.time()
    # print('Round {} took {} seconds'.format(comm_round, end_time-start_time))
    # compute total time for all rounds
    total_training_time.append(end_time-start_time)

# compute total time in minutes
sum(total_training_time)/60



# %%

global_loss_list = [float(i) for i in global_loss_list]
noniid_df = pd.DataFrame(list(zip(global_acc_list, global_loss_list, global_time_list)), 
columns =['global_acc_list', 'global_loss_list', 'global_time_list'])
noniid_df.to_csv('PROVA_SMOTE.csv',index=False)

# %% 

#read data from csv

data_SMOTE = pd.read_csv('PROVA_SMOTE.csv')
data_benchmark = pd.read_csv('MNIST_benchmark.csv')
data_randover = pd.read_csv('MNIST_RandOver.csv')

#plot the results on subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(data_SMOTE['global_acc_list'], label='SMOTE')
axs[0].plot(data_benchmark['global_acc_list'], label='benchmark')
axs[0].plot(data_randover['global_acc_list'], label='random oversampling')
axs[0].set_title('Accuracy')
axs[0].legend()

axs[1].plot(data_SMOTE['global_loss_list'], label='SMOTE')
axs[1].plot(data_benchmark['global_loss_list'], label='benchmark')
axs[1].plot(data_randover['global_loss_list'], label='random oversampling')
axs[1].set_title('Loss')
axs[1].legend()



