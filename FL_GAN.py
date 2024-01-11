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


clients = create_shards_function(X_train, y_train, no_clients=100, 
no_classes=10, niid=True, balance=False, 
partition='pathological', class_per_client=6, initial='clients')



# clients lack class 5

# %%

# clients.keys()
# clients['clients_0'][0].shape, clients['clients_0'][1].shape
# len(clients['clients_0'][1]) # 5363
# create dummies for the labels in clients  

# create dummies on "longest data"
# y_test
# y_test_lb = lb.fit_transform(y_test)
y_test_dummy = pd.get_dummies(y_test)

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


def double_compress_images(image, p_pix_obs=0.5, p_gauss_noise=0.5):
    np.random.seed(42)
    image_compressed = image.copy()
    for i in range(len(image)):
        if np.random.rand() < p_pix_obs:
            image_compressed[i] = 0
        if np.random.rand() < p_gauss_noise:
            image_compressed[i] = np.random.normal(0, 0.01)
        # if np.random.rand() < p_pix_obs:
        #     image_compressed[i] = 0
        #if there is a negative value, set it to 0
        if image_compressed[i] < 0:
            image_compressed[i] = 0
        #if the original image has more 0 values than the compressed image, set other values to 0
        if image[i] == 0 and image_compressed[i] != 0:
            image_compressed[i] = 0
    return image_compressed



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

for comm_round in range(comm_rounds): 

    start_time = time.time()
    # compute time for each round

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    clients_involved = list()
    random.seed(42)
    #randomize client data - using keys
    all_client_names = list(clients_train.keys())
    k=random.randint(10, len(all_client_names)/2)
    # set seed for reproducibility
    random.seed(42)
    # select a random sample of clients between 10 and len(all_client_names)
    client_names = random.sample(all_client_names, k=k)

    # client_names = random.sample(all_client_names, k=10)
    # I randomly select 10 clients for each round - 
    # print(client_names, len(client_names))
    random.shuffle(client_names)
    # select random number of clients for each round
    #print selected number of clients
    # print('The selected clients for round {} are: {}'.format(comm_round, client_names))
    # print('The number of clients for round {} is: {}'.format(comm_round, len(client_names)))

    # if debug: 
    #     # print('all_client_names', all_client_names)
    #     print('client_names', client_names, len(client_names))
                
    # clients_involved.append(len(client_names))

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

    # each client double compresses its data and converts it into a sparse matrix
    for client in client_names: #break
        label_distribution[client] = clients_label[client].sum(axis=0)
    
    # totale di ciascun label per round
    total_labels_round = np.sum([label_distribution[client] for client in client_names], axis=0)

    # check distance between the most full client and the least full client
    # print('The distance between the most full client and the least full client is: {}'.format(np.max(total_labels_round)-np.min(total_labels_round)))

    samples_to_create = np.max(total_labels_round)-np.min(total_labels_round)

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

    # cluster together clients with different label distribution for each class so to have a more balanced dataset  

    # df.sum(axis=0)

    # # highlight the clients with the most frequent labels 
    # df.style.highlight_max(color = 'lightgreen', axis = 0)

    # # # select the clients with the most frequent labels - greater than 10%
    # df[df > 15]

    # pick the name of the clients with the most frequent labels and the most frequent labels

    # create a dictionary with the name of the clients and the most frequent labels
    
    for client in client_names:
        most_frequent_labels[client] = df[df > 0.05].loc[client].dropna().index.tolist()
        # consider clients with at least 5% of the most frequent labels

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

    for client in client_names:
        share_list[client] = (label_distribution[client]*frequency2[client]).astype(int)
    
    # print the proportion of data each client is sending to the server
    # divide share list by the total amount of data for each client

    # for client in client_names:
    #     print('The proportion of data for client {} is: {}'.format(client, share_list[client].sum()/len(clients_train[client])))

    # if some proportion is greater than 0.5, reduce the proportion to 0.5
    for client in client_names:
        if share_list[client].sum()/len(clients_train[client]) > 0.5:
            share_list[client] = ((share_list[client]/share_list[client].sum())*0.5*len(clients_train[client])).astype(int)
        
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
    # for client in client_names:
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
    df_to_GAN = pd.concat(compressed_data_sparse.values(), axis=0)

    assert len(df_to_GAN) == sum([share_list[client].sum() for client in client_names])


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
    df_to_GAN_final = df_to_GAN.copy()

    # create a new column with X converted to numpy array
    df_to_GAN_final = df_to_GAN_final.drop(columns=['y'])

    df_to_GAN_final['X'] = df_to_GAN_final['X'].apply(lambda x: x.toarray()[0])
    # df_to_GAN_final.loc[106, 'X'].shape # (60000, 1)
    # print(type(df_to_oversample_final.loc[31, 'X']))
    # # remove the second dimension
    
    

        # define how many samples per class I want to have in the final dataset
        # in order to balance the classes of the clients in the original dataset 

    samples_to_generate_per_class = samples_to_create

    latent_dim = samples_to_generate_per_class * len(df_to_GAN_final['label'].value_counts())

        
















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
        num_data_points_to_add = int(num_data_points * (1 - (num_data_points / total_labels_round.sum())))
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
        clients_labels_new[client] = np.concatenate((clients_label[client], pd.get_dummies(labels_to_add['label'])))
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
noniid_df = pd.DataFrame(list(zip(global_acc_list, global_loss_list, global_time_list)), columns =['global_acc_list', 'global_loss_list', 'global_time_list'])
noniid_df.to_csv('MNIST_RandOver.csv',index=False)

# %% 

#read data from csv

data_SMOTE = pd.read_csv('MNIST_SMOTE.csv')
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






   
   
    # reshape the X column to 28x28
    
    


# %%
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples(df_to_GAN_final)
# dataset is a list with 2 elements: X and y

# train model
history = train(g_model, d_model, gan_model, dataset, latent_dim)

    


import matplotlib.pyplot as plt 
# create and save a plot of generated images in columns per class
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()

# visualize empty plot 
# save_plot(np.zeros((100, 28, 28, 1)), 10)



# load model
from keras.models import load_model
import numpy as np

model = load_model('cgan_generator.h5', compile=False)
# generate images
latent_points, labels = generate_latent_points(100, 100)
# specify labels
labels = np.asarray([x for _ in range(10) for x in range(10)])
# generate images
X = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 10)


# save the generated images and associated labels into a list of tuples

# create a list of tuples
# each tuple contains a numpy array with the image and the label
# the list is called 'generated_images'
generated_images = []
for i in range(len(X)):
    generated_images.append((X[i], labels[i]))


# plot rabdom images from the generated images

plt.figure(figsize=(15,15))
for i in range(50):
    plt.subplot(10,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(generated_images[i][0], cmap=plt.cm.binary)
    plt.xlabel(generated_images[i][1])
plt.show()

    
len(generated_images) # 100 
generated_images[0][0].shape # (28, 28, 1)
generated_images[0][1].shape # ()

# how many images per label do we have?
# create a dictionary with the label as key and the number of images as value
# the dictionary is called 'generated_images_per_label'
generated_images_per_label = {}
for i in range(len(generated_images)):
    if generated_images[i][1] in generated_images_per_label:
        generated_images_per_label[generated_images[i][1]] += 1
    else:
        generated_images_per_label[generated_images[i][1]] = 1

# print the dictionary
generated_images_per_label

#%%
# define how many images we want per label according to data imbalance in the original dataset
# so to balance the clients data

# create a dictionary with the label as key and the number of images as value
# the dictionary is called 'images_per_label'
images_per_label = {}
for i in range(len(df_to_GAN_final)):
    if df_to_GAN_final['label'][i] in images_per_label:
        images_per_label[df_to_GAN_final['label'][i]] += 1
    else:
        images_per_label[df_to_GAN_final['label'][i]] = 1
        




   def load_real_samples(data):
    # load dataset
    #  (trainX, trainy), (_, _) = load_data()
        trainX, trainy = data['X'], data['label']

        # convert trainX from object to float keeping indexes
        trainX = np.array(trainX.tolist(), dtype=np.float32)


        # index each element of trainX with df_to_GAN.index


        # trainX = [np.array(x, dtype=np.float32) for x in trainX]
        # trainX = np.array(trainX, dtype=np.float32)

        # trainX.shape #(3264, 784)

        # reshape to 28x28
        trainX = trainX.reshape((trainX.shape[0], 28, 28))

       
        #  trainX.shape
        #  trainy.shape
        # expand to 3d, e.g. add channels
        X = expand_dims(trainX, axis=-1)

        # convert trainy into array
        trainy = np.array(trainy.tolist(), dtype=np.int32)

        # convert from ints to floats
        # X = X.astype('float32')
        # scale from [0,255] to [-1,1]
        # X = (X - 127.5) / 127.5
        return [X, trainy]


# %% 

# define how many samples per class I want to have in the final dataset
# in order to balance the classes of the clients in the original dataset 

len(df_to_GAN_final) # 3264
len(df_to_GAN_final['label'].value_counts()) # 10

samples_to_generate_per_class = len(df_to_GAN_final) // len(df_to_GAN_final['label'].value_counts())

latent_dim = samples_to_generate_per_class * len(df_to_GAN_final['label'].value_counts())



# %%
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples(df_to_GAN_final)
# dataset is a list with 2 elements: X and y

# train model
history = train(g_model, d_model, gan_model, dataset, latent_dim)

    


import matplotlib.pyplot as plt 
# create and save a plot of generated images in columns per class
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()

# visualize empty plot 
# save_plot(np.zeros((100, 28, 28, 1)), 10)



# load model
from keras.models import load_model
import numpy as np

model = load_model('cgan_generator.h5', compile=False)
# generate images
latent_points, labels = generate_latent_points(100, 100)
# specify labels
labels = np.asarray([x for _ in range(10) for x in range(10)])
# generate images
X = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 10)


# save the generated images and associated labels into a list of tuples

# create a list of tuples
# each tuple contains a numpy array with the image and the label
# the list is called 'generated_images'
generated_images = []
for i in range(len(X)):
    generated_images.append((X[i], labels[i]))


# plot rabdom images from the generated images

plt.figure(figsize=(15,15))
for i in range(50):
    plt.subplot(10,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(generated_images[i][0], cmap=plt.cm.binary)
    plt.xlabel(generated_images[i][1])
plt.show()

    
len(generated_images) # 100 
generated_images[0][0].shape # (28, 28, 1)
generated_images[0][1].shape # ()

# how many images per label do we have?
# create a dictionary with the label as key and the number of images as value
# the dictionary is called 'generated_images_per_label'
generated_images_per_label = {}
for i in range(len(generated_images)):
    if generated_images[i][1] in generated_images_per_label:
        generated_images_per_label[generated_images[i][1]] += 1
    else:
        generated_images_per_label[generated_images[i][1]] = 1

# print the dictionary
generated_images_per_label

#%%
# define how many images we want per label according to data imbalance in the original dataset
# so to balance the clients data

# create a dictionary with the label as key and the number of images as value
# the dictionary is called 'images_per_label'
images_per_label = {}
for i in range(len(df_to_GAN_final)):
    if df_to_GAN_final['label'][i] in images_per_label:
        images_per_label[df_to_GAN_final['label'][i]] += 1
    else:
        images_per_label[df_to_GAN_final['label'][i]] = 1
        




# %% 
    

    #loop through each client and create new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(build_shape, 10)
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_train[client], clients_label[client], 
                epochs=1, verbose=0, batch_size=32)
        
        #scale the model weights and add to list
        scaling_factor =  weight_scalling_factor(clients_train, client)
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




# create a function set to zero some pixels in the image with ratio p 
# image = clients_train['clients_0'][373]
# image_2 = clients_train['clients_0'][0]

# image.shape
# plt.imshow(image.reshape(28,28), cmap='gray')

# #set seed for reproducibility
# def set_to_zero(image, p=0.5):
#     np.random.seed(42)
#     image_compressed = image.copy()
#     for i in range(len(image)):
#         if np.random.rand() < p:
#             image_compressed[i] = 0
#     return image_compressed

# # define function to add gaussian noise to the image
# def add_gaussian_noise(image, p=0.5):
#     np.random.seed(42)
#     image_noised = image.copy()
#     for i in range(len(image)):
#         if np.random.rand() < p:
#             image_noised[i] = np.random.normal(0, 0.01)
#     return image_noised

# global function to create compressed images for different p

# def compress_images(image, p_pix_obs=0.5, p_gauss_noise=0.5):
#     np.random.seed(42)
#     image_compressed = image.copy()
#     for i in range(len(image)):
#         if np.random.rand() < p_pix_obs:
#             image_compressed[i] = 0
#         if np.random.rand() < p_gauss_noise:
#             image_compressed[i] = np.random.normal(0, 0.01)
#         #if there is a negative value, set it to 0
#         if image_compressed[i] < 0:
#             image_compressed[i] = 0
#     return image_compressed

# # create compressed images for different p

# p_list = [0.0, 0.01, 0.04, 0.1, 0.15,  0.2, 0.25,  0.3, 0.35, 0.4, 0.45,  0.5, 0.55, 
# 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# for i in range(len(p_list[:12])):
#     plt.subplot(4,5,i+1)
#     plt.imshow(set_to_zero(image, p=p_list[i]).reshape(28,28), cmap='gray')
#     plt.title(f'p={p_list[i]}')
#     plt.axis('off')

# for i in range(len(p_list[:12])):
#     plt.subplot(4,5,i+1)
#     plt.imshow(add_gaussian_noise(image, p=p_list[i]).reshape(28,28), cmap='gray')
#     plt.title(f'p={p_list[i]}')
#     plt.axis('off')
    
# # compare image with different set to zero and noise

# plt.subplot(1,4,1)
# plt.imshow(image.reshape(28,28), cmap='gray')
# plt.title('original')
# plt.axis('off')

# plt.subplot(1,4,2)
# plt.imshow(set_to_zero(image, p=0.5).reshape(28,28), cmap='gray')
# plt.title('pixel obs')
# plt.axis('off')

# plt.subplot(1,4,3)
# plt.imshow(add_gaussian_noise(image, p=0.5).reshape(28,28), cmap='gray')
# plt.title('gaussian noise')
# plt.axis('off')

# plt.subplot(1,4,4)
# plt.imshow(add_gaussian_noise(set_to_zero(image, p=0.5), p=0.5).reshape(28,28), cmap='gray')
# plt.title('pixel obs + gaussian noise')
# plt.axis('off')

# %%

# apply the function to clients_0 for all p

# clients_0 = clients_train['clients_0']
# clients_0_comp = dict()

# for p in p_list[:14]:
#     clients_0_comp[p] = np.array([compress_images(image, p_pix_obs=p, p_gauss_noise=p) for image in clients_0])

# clients_0_comp.keys()

# # plot 2 random images for each p 

# for p in p_list[:14]:
#     plt.subplot(4,5,p_list.index(p)+1)
#     plt.imshow(clients_0_comp[p][373].reshape(28,28), cmap='gray')
#     plt.title(f'p={p}')
#     plt.axis('off')

# # %%

# # compress the images into a sparse matrix
# # when in a matrix a lot of values are equal zero, it is not efficient to store such
# # data in a matrix as it consumes a lot of memory. it is better to store only the non-zero
# # values and their position in the matrix. this is called sparse matrix

# # np arrays work better with functions needed for creating compressed sparse row CSR format

# # I convert the data into np.array and then into CSR format

# clients_0_comp.keys()

# # access data with p = 0.35

# clients_0_comp[0.35].shape

# clients_0_comp35 = clients_0_comp[0.35]

# len(clients_0)
# len(clients_0_comp)

# # isolate client 0 one sample

# clients_0_35 = clients_0[0]
# clients_0_35.shape # image 
# plt.imshow(clients_0_35.reshape(28,28), cmap='gray')

# # isolate client 0 one sample compressed

# clients_0_35_comp = clients_0_comp35[0]
# clients_0_35_comp.shape # image
# plt.imshow(clients_0_35_comp.reshape(28,28), cmap='gray')

# # compare the number of zero values in the original image and the compressed one

# print(f'number of zero values in the original image: {np.count_nonzero(clients_0_35 == 0)}') # 567
# print(f'number of zero values in the compressed image: {np.count_nonzero(clients_0_35_comp == 0)}') # 575


# # in percentage

# np.count_nonzero(clients_0_35)/len(clients_0_35) # 0.2767 values are non-zero
# np.count_nonzero(clients_0_35_comp)/len(clients_0_35_comp) # 0.2665 values are non-zero

# # check if there are negative pixel values in the compressed image

# np.count_nonzero(clients_0_35_comp < 0) # 0 

# ########################

# # I could also try to do pixel obscuration, gaussian and pixel obscuration again 



# def double_compress_images(image, p_pix_obs=0.5, p_gauss_noise=0.5):
#     np.random.seed(42)
#     image_compressed = image.copy()
#     for i in range(len(image)):
#         if np.random.rand() < p_pix_obs:
#             image_compressed[i] = 0
#         if np.random.rand() < p_gauss_noise:
#             image_compressed[i] = np.random.normal(0, 0.01)
#         if np.random.rand() < p_pix_obs:
#             image_compressed[i] = 0
#         #if there is a negative value, set it to 0
#         if image_compressed[i] < 0:
#             image_compressed[i] = 0
#     return image_compressed



# # clients_0 = clients_train['clients_0']
# clients_0_double_comp = dict()

# for p in p_list[:14]:
#     clients_0_double_comp[p] = np.array([double_compress_images(image, p_pix_obs=p, p_gauss_noise=p) for image in clients_0])

# # plot 2 random images for each p 

# for p in p_list[:14]:
#     plt.subplot(4,5,p_list.index(p)+1)
#     plt.imshow(clients_0_double_comp[p][373].reshape(28,28), cmap='gray')
#     plt.title(f'p={p}')
#     plt.axis('off')


# # compare the number of zero values in the compressed image and the double compressed one and original image

# print(f'number of zero values in the original image: {np.count_nonzero(clients_0_35 == 0)}') # 567

# print(f'number of zero values in the compressed image: {np.count_nonzero(clients_0_35_comp == 0)}') # 575

# print(f'number of zero values in the double compressed image: {np.count_nonzero(clients_0_double_comp[0.35][0] == 0)}') # 618

# # %%

# # convert every image into 28x28 matrix

# clients_0_comp35 = clients_0_double_comp[0.35].reshape(607, 28, 28)
# clients_0_comp35.shape

# # plt.imshow(clients_0_comp35[400], cmap='gray')


# # compare the original image with the compressed and double compressed one

# plt.subplot(1,3,1)
# plt.imshow(clients_0[400].reshape(28,28), cmap='gray')
# plt.title('original')
# plt.axis('off')

# plt.subplot(1,3,2)
# plt.imshow(clients_0_comp[0.35][400].reshape(28,28), cmap='gray')
# plt.title('compressed')
# plt.axis('off')

# plt.subplot(1,3,3)
# plt.imshow(clients_0_double_comp[0.35][400].reshape(28,28), cmap='gray')
# plt.title('double compressed')
# plt.axis('off')


# # %%

# # count zero values in original image and compressed image

# immagine_prova = clients_0_double_comp[0.35][400].reshape(28,28)
# immagine_prova.shape

# type(immagine_prova)

# # convert from array to sparse matrix

# from scipy import sparse

# immagine_prova_sparse = sparse.csr_matrix(immagine_prova)

# # check non zero values of immagine_prova

# np.count_nonzero(immagine_prova) # 129


# immagine_prova_sparse.data

# immagine_prova_sparse.data.shape # (129,)

# print(immagine_prova_sparse)

# type(immagine_prova_sparse) # scipy.sparse.csr.csr_matrix

# # I can convert the sparse matrix into a dense matrix/array with the toarray() method

# img_arr = immagine_prova_sparse.toarray()
# img_arr.shape # (28, 28)
# # flatten the array
# img_arr.flatten().shape # (784,)



# new function

# function to double compress
