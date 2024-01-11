# %%

# import the necessary packages

import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
# import opencv as cv2
import os

CVD = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = CVD
# Some other determinism...
SEED = 3
os.environ['PYTHONHASHSEED'] = str(SEED)


# from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras import backend as K
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# import warnings filter
import warnings

# import packages for the oversampling
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE

# from time import time

debug=0

# get working directory
os.getcwd()
# set working directory to FL_clust
# os.chdir('/home/predicts-workbench/Projects/Diletta/FL/FL_clust')

from FL_utils import *

from GAN_source_new import *


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



# # check if I am using GPU
# tf.test.is_gpu_available()


random.seed(SEED)
np.random.seed(SEED)

# %%

# load the MNIST dataset from internet 

dataset = 'mnist'

if dataset == 'mnist':
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    assert trainX.shape == (60000, 28, 28)
    assert testX.shape == (10000, 28, 28)
    assert trainY.shape == (60000,)
    assert testY.shape == (10000,)

elif dataset == 'cifar10':
    (trainX, trainY), (testX, testY) =  tf.keras.datasets.cifar10.load_data()
    assert trainX.shape == (50000, 32, 32, 3)
    assert testX.shape == (10000, 32, 32, 3)
    assert trainY.shape == (50000, 1)
    assert testY.shape == (10000, 1)

elif dataset == 'fashion_mnist':
    (trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
    assert trainX.shape == (60000, 28, 28)
    assert testX.shape == (10000, 28, 28)
    assert trainY.shape == (60000,)
    assert testY.shape == (10000,)


# fig = plt.figure(figsize=(8, 8))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = trainX[i]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)

# %% 
X = np.concatenate((trainX, testX), axis=0)

if dataset == 'cifar10':
    y = np.concatenate((trainY, testY), axis=0).reshape(-1)
else:
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


# # check label distribution among clients
# for client_name, data in clients.items():
#     print(client_name, len(data[1]), len(set(data[1])))
#     print(set(data[1]))

client0 = clients['clients_0']
client1 = clients['clients_1']
client98 = clients['clients_98']
client99 = clients['clients_99']



import seaborn as sns

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Distribution of the classes per clients for the MNIST dataset', fontsize=16)
# countplot 
sns.countplot(client0[1], ax=ax[0])
# set title to the figure
ax[0].set_title('Client 1', fontsize=14)
ax[0].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax[0].get_xticklabels()])
# set x and y axis labels
ax[0].set_xlabel('Class', fontsize=14)
ax[0].set_ylabel('No. of samples', fontsize=14)

ax[1].set_title('Client 2', fontsize=14)
sns.countplot(client1[1], ax=ax[1])
ax[1].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax[1].get_xticklabels()])
ax[1].set_xlabel('Class', fontsize=14)
ax[1].set_ylabel('No. of samples', fontsize=14)

sns.countplot(client98[1], ax=ax[2])
ax[2].set_title('Client 99', fontsize=14)
ax[2].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax[2].get_xticklabels()])
ax[2].set_xlabel('Class', fontsize=14)
ax[2].set_ylabel('No. of samples', fontsize=14)

sns.countplot(client99[1], ax=ax[3])
ax[3].set_title('Client 100', fontsize=14)
ax[3].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax[3].get_xticklabels()])
ax[3].set_xlabel('Class', fontsize=14)
ax[3].set_ylabel('No. of samples', fontsize=14)
plt.tight_layout()
plt.savefig('class_distribution_persamples_MNIST.png')
plt.show()

# original dataset label distribution - countplot price class

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle('Original label distribution in the MNIST dataset', fontsize=16)
# countplot
sns.countplot(y, ax=ax)
# set title to the figure
# ax.set_title('Original dataset', fontsize=14)
ax.set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
# set x and y axis labels
ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('No. of samples', fontsize=14)
plt.tight_layout()
plt.savefig('class_distribution_original_MNIST.png')
plt.show()

# CHECK HOW MANY CLIENTS HAVE class 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ... 

# create a function to store the number of clients that have each class

clients_per_class = {}

for client_name, data in clients.items():
    for label in np.unique(data[1]):
        if label in data[1]:
            if label in clients_per_class:
                clients_per_class[label] += 1
            else:
                clients_per_class[label] = 1

# make a plot where x axis is labels and y axis is the number of clients that have that label

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle('Number of clients per class', fontsize=16)
ax.bar(range(len(clients_per_class)), list(clients_per_class.values()), align='center')
ax.set_xticks(range(len(clients_per_class)))
# ax.set_xticklabels(list(clients_per_class.keys()))
# ax.set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('No. of clients', fontsize=14)
plt.tight_layout()
plt.savefig('class_distribution_clients_MNIST.png')
plt.show()



# %%

# clients.keys()
# clients['clients_0'][0].shape, clients['clients_0'][1].shape
# len(clients['clients_0'][1]) # 5363
# create dummies for the labels in clients  

# create dummies on "longest data"
y_test
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

clients_train = dict()
clients_label = dict()
for client_name, data in clients.items(): 
    clients_train[client_name] = data[0]
    clients_label[client_name] = pd.get_dummies(data[1]).reindex(columns=y_test_dummy.columns, fill_value=0)

test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test_dummy)).batch(len(y_test_dummy))

# # %%
# clients_trn_data = clients_train.copy()

# %% 
build_shape_cnn = (trainX[0].shape[0], trainX[0].shape[1], 1)
build_shape_cnn_cifar = (trainX[0].shape[0], trainX[0].shape[1], trainX[0].shape[2])

build_shape_smlp = 784

model = 'cifar_cnn'
# %% 
no_classes = 10
if model == 'cnn':
    from models_to_train import SimpleCNN
    cnn_global = SimpleCNN()
    global_model = cnn_global.build(build_shape_cnn, no_classes)
    # global_model.summary()
    # global_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
elif model == 'mlp':
    from models_to_train import SimpleMLP
    mlp_global = SimpleMLP()
    global_model = mlp_global.build(build_shape_smlp, no_classes)
    # global_model.summary()
    # global_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
elif model == 'cifar_cnn':
    from models_to_train import CNN_cifar10
    cifar_cnn_global = CNN_cifar10()
    global_model = cifar_cnn_global.build(build_shape_cnn_cifar, no_classes)
    # global_model.summary()
    # global_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

# %%



lr = 0.01
comm_rounds = 50
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import tqdm


metrics = ['categorical_accuracy']
# optimizer = SGD(lr=lr, 
#                 decay=lr / comm_rounds, 
#                 momentum=0.9
#                )          


 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST


# %% 
#initialize global model


global_acc_list = []
global_loss_list = []
# global_class_report_list = []
# global_conf_mat_list = []
macro_precision = []
macro_recall = []
macro_f1 = []
global_time_list = []
total_training_time = []
comm_overhead_list = []
# %%
import sys
import pickle

# the overall communication cost of the server is given by
# (2 * N * |Wt|) * T + (N * |Wt|) 
# where N is the size of the trained model in bytes
# |Wt| is the number ofselected clients at round t
# T is the number of training rounds

# We assume the size of the model updates and the number of training iterations to be fixed 

# define function to compute communication overhead
# def compute_comm_overhead(local_model, clients_involved):
#     # compute the size of the model
#     model_size = sys.getsizeof(pickle.dumps(local_model))
#     # compute the number of clients involved in each round
#     # clients_involved = len(client_names)
#     # compute the communication overhead
#     comm_overhead = (2 * len(clients_involved) * model_size) * 1 + (len(clients_involved) * model_size)
#     return comm_overhead



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

epochs= 20
batch_size = 32
algorithm = 'fednova'
server_lr = 1.0
mu_fedprox = 0.1
patience = None

algo_s = None
if algorithm == 'fedavg':
    algo_s = 'FEDAVG'
elif algorithm == 'fedopt':
    algo_s = 'FEDOPT'
elif algorithm == 'fednova':
    algo_s = 'FEDNOVA'
elif algorithm == 'fedprox':
    algo_s = 'FEDPROX'

assert algo_s is not None, 'Algorithm not supported'

if algorithm != 'fedprox':
    loss='categorical_crossentropy'
else:
    def loss(y_true, y_pred):
        loss_val = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=False))
        return {
            'loss': loss_val,
            'loss_noreg': loss_val,
        }

client_optimizer = 'SGD'

assert client_optimizer is not None

if client_optimizer == 'SGD':
    client_kwargs = dict(decay=lr / comm_rounds, momentum=0.9)
else:
    client_kwargs = {}

server_optimizer = None

if server_optimizer is not None:
    if server_optimizer == 'SGD':
        server_kwargs = dict(decay=server_lr / comm_rounds, momentum=0.9)
    else:
        server_kwargs = {}
    server_opt = get_optimizer(server_optimizer, server_lr, **server_kwargs)

class Config:
    epochs = epochs
    batch_size = batch_size
    algorithm = algorithm
    patience = patience
    seed = SEED
    plot_epochs = (100,250)
    epoch_time = 10
    algorithm = algorithm
    mu = mu_fedprox
    verbose = 0


cf = Config()

# %%

#commence global training loop
# comm_round = 0
for comm_round in range(comm_rounds): 
    # compute time for each round
    start_time = time.time()
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    local_weight_list = list()
    # clients_involved = list()
    # clients_involved_names = list()
    comm_overhead = list()
    #randomize client data - using keys
    k = k_list[comm_round]
    client_names = clients_list[comm_round]

    # clients_involved.append(len(client_names))
    # clients_involved_names.append(client_names)

    # select random number of clients for each round
    #print selected number of clients
    # print('The selected clients for round {} are: {}'.format(comm_round, client_names))
    # print('The number of clients for round {} is: {}'.format(comm_round, len(client_names)))
    
    # if debug: 
    #     # print('all_client_names', all_client_names)
    #     print('client_names', client_names, len(client_names))
                
    
    tau = []

    lens = []

    #loop through each client and create new local model
    # client = client_names[0]
    for client in client_names: 
        client_opt = get_optimizer(client_optimizer, lr, **client_kwargs)

        if model == 'mlp':
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(build_shape_smlp, no_classes)
            local_model.compile(loss=loss, 
                      optimizer=client_opt, 
                      metrics=metrics)

        elif model == 'cnn':
            cnn_local = SimpleCNN()
            local_model = cnn_local.build(build_shape_cnn, no_classes)
            # reshape data for cnn
            if dataset == 'mnist' or dataset == 'fashion_mnist':
                            clients_train[client] = clients_train[client].reshape(clients_train[client].shape[0], 28, 28, 1)
            elif dataset == 'cifar10':
                            clients_train[client] = clients_train[client].reshape(clients_train[client].shape[0], 32, 32, 3)
            local_model.compile(loss=loss, 
                      optimizer=client_opt, 
                      metrics=metrics)

        elif model == 'cifar_cnn':
            cifar_cnn_local = CNN_cifar10()
            local_model = cifar_cnn_local.build(build_shape_cnn_cifar, no_classes)
            # reshape data for cnn
            if dataset == 'mnist' or dataset == 'fashion_mnist':
                            clients_train[client] = clients_train[client].reshape(clients_train[client].shape[0], 28, 28, 1)
            elif dataset == 'cifar10':
                            clients_train[client] = clients_train[client].reshape(clients_train[client].shape[0], 32, 32, 3)
            local_model.compile(loss=loss, 
                      optimizer=client_opt, 
                      metrics=metrics)
        
        tau_client = 0
        
        len_client = len(clients_train[client])
        lens.append(len_client)

        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        # loss = loss + lr/ 2 * np.sum(np.square(global_weights))

        #fit local model with client's data
        if algorithm != 'fedprox':
            local_model.compile(loss=loss, 
                        optimizer=client_opt, 
                        metrics=metrics)
            history = local_model.fit(clients_train[client], clients_label[client], 
                    epochs=epochs, verbose=0, batch_size=batch_size)
        else:
            history, model = train(
                clients_train[client], clients_label[client], local_model, loss, client_opt, cf, global_weights)
            # TODO: implement manual training loop
            # raise NotImplementedError('fedopt not implemented yet')

        local_weight_list.append(local_model.trainable_weights)

        n_batches = len(clients_train[client]) // batch_size
        if len(clients_train[client]) % batch_size != 0:
            n_batches += 1
        
        tau_client = n_batches * epochs
        tau.append(tau_client)

        # #scale the model weights and add to list
        # scaling_factor =  weight_scalling_factor(clients_train, client)
        # # print('scaling_factor', scaling_factor)
        # scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        # scaled_local_weight_list.append(scaled_weights)

        # communication_overload = compute_comm_overhead(local_model, client_names)
        # print('communication_overload for round {} is: {}'.format(comm_round, communication_overload))
        #clear session to free memory after each communication round
        K.clear_session()
    
    lens = np.array(lens)
    tau = np.array(tau)
    n_per_round = sum(lens)

    if algorithm == 'fedavg':
        scaled_local_weight_list = [tf.nest.map_structure(
            lambda w: lens[i]/ n_per_round * w, local_weight_list[i]) for i in range(len(local_weight_list))]
        # scaled_local_weight_list = [[lens[i]/ n_per_round * w for w in local_weight_list[i]] for i in range(len(local_weight_list))]

        updated_weights = sum_weights(scaled_local_weight_list)
        global_model.set_weights(updated_weights)
    
    elif algorithm == 'fedopt' or algorithm == 'fedprox':
        delta_weights = [tf.nest.map_structure(
            lambda globalw, w: globalw - w, global_weights, local_weight_list[i]) for i in range(len(local_weight_list))]
        # delta_weights = [[global_weights[j] -local_weight_list[i][j]  for j in range(len(local_weight_list[i]))] for i in range(len(local_weight_list))]

        scaled_delta_weights = [
            tf.nest.map_structure(lambda dw: lens[i]/ n_per_round * dw, delta_weights[i]) for i in range(len(delta_weights))]
        # scaled_delta_weights = [[lens[i]/ n_per_round * w for w in delta_weights[i]] for i in range(len(delta_weights))]
        average_delta_weights = sum_weights(scaled_delta_weights)

        if server_optimizer is None:
            updated_weights = tf.nest.map_structure(
                lambda globalw, dw: globalw - server_lr*dw, global_weights, average_delta_weights)
            # updated_weights = [global_weights[j] - lr * average_delta_weights[j] for j in range(len(average_delta_weights))]
            global_model.set_weights(updated_weights)
        else:
            server_opt.apply_gradients(zip(average_delta_weights, global_model.trainable_variables))

    
    elif algorithm == 'fednova':
        delta_weights = [tf.nest.map_structure(
            lambda globalw, w: globalw - w, global_weights, local_weight_list[i]) for i in range(len(local_weight_list))]
        
        scaled_delta_weights = [
            tf.nest.map_structure(lambda dw: lens[i]/ (n_per_round * tau[i])* dw, delta_weights[i]) for i in range(len(delta_weights))]

        average_delta_weights = sum_weights(scaled_delta_weights)
        weighting_factor = np.sum(lens * tau) / n_per_round 

        if server_optimizer is None:
            updated_weights = tf.nest.map_structure(
                lambda globalw, dw: globalw - server_lr*weighting_factor *dw, global_weights, average_delta_weights)
            # updated_weights = [global_weights[j] - lr * average_delta_weights[j] for j in range(len(average_delta_weights))]
            global_model.set_weights(updated_weights)
        
        else:
            updated_average_delta_weights = tf.nest.map_structure(
                lambda dw: weighting_factor * dw, average_delta_weights)

            server_opt.apply_gradients(zip(updated_average_delta_weights, global_model.trainable_variables))


    #update global model 
    
    # N is the size of the trained model (in bytes)
    N = sys.getsizeof(global_model.get_weights())
    # Ws is the number of selected clients for each round
    Ws = len(client_names)
    # T is the number of epochs for each client
    T = epochs

    communication_overhead = 2 * N * Ws * T + (N * Ws)

    comm_overhead_list.append(communication_overhead)

    end_time = time.time()
    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:

        if model == 'cnn':

            # reshape the data to fit the model
            if dataset == 'mnist' or dataset == 'fashion_mnist':
                X_test = np.array(X_test).reshape(X_test.shape[0], 28, 28, 1)
        
        if dataset == 'cifar10':
                X_test = np.array(X_test).reshape(X_test.shape[0], 32, 32, 3)
            
        
        if algorithm == 'fedprox':
            if dataset == 'mnist' or dataset == 'fashion_mnist':
                X_test = X_test.numpy().reshape(X_test.shape[0], 28, 28, 1)
            elif dataset == 'cifar10':
                X_test = X_test.numpy().reshape(X_test.shape[0], 32, 32, 3)

    
    

        global_acc, class_report, conf_mat, global_loss, global_time_taken = test_model(X_test, Y_test, global_model, comm_round, dataset)
        
        global_acc_list.append(global_acc)
        macro_precision.append(class_report['macro avg']['precision'])
        macro_recall.append(class_report['macro avg']['recall'])
        macro_f1.append(class_report['macro avg']['f1-score'])
        global_loss_list.append(global_loss)
        
        global_time_list.append(global_time_taken)
        

    
    # print('The number of clients for round {} is: {}'.format(comm_round, len(client_names)))
    # print('The selected clients for round {} are: {}'.format(comm_round, client_names))

    # print('Round {} took {} seconds'.format(comm_round, end_time-start_time))
    # compute total time for all rounds
    total_training_time.append(end_time-start_time)
    

    #print taken time for each round
    print('Round {} took {} seconds'.format(comm_round, end_time-start_time))

    
    

    # assert len(comm_overhead) == comm_round

    # compute communication overhead

# %% 



# np.array(total_training_time).min()/60
# np.array(total_training_time).max()/60
# np.array(total_training_time).mean()/60
# np.array(total_training_time).std()/60

# boxplot training time
# plt.boxplot(np.array(total_training_time)/60)
# plt.xlabel('Training time (minutes)')
# plt.ylabel('CNN model - FedAvg - MNIST') 

# boxplot for each communication round

# plot histogram of clients_list for each round with loss on y-axis
global_loss_list = [float(i) for i in global_loss_list]

# %%

b = dict()


# see label distribution for each client in clients_list
for i in range(len(clients_list)):
    batch_clients = clients_list[i]

    # for each batch in clients_list extract the labels
    for name in batch_clients:
        data = clients_label[name]
        dist = data.sum(axis=0)
        k = no_classes
        n = data.shape[0]

        entropy = 0
        for i in range(k): 
            ci = dist[i]
            if ci == 0:
                continue
            entropy += (ci/n)*np.log(ci/n)
        entropy = -entropy
        
        # save mean entropy for each batch of clients
        b[name] = entropy


# aggregate the entropy for each batch of clients
total_entropy = []
for i in range(len(clients_list)):
    batch_clients = clients_list[i]
    entropy = 0
    for name in batch_clients:
        entropy += b[name]
    entropy = entropy/len(batch_clients)
    total_entropy.append(entropy)


# %% 
df= pd.DataFrame(list(zip(global_acc_list, global_loss_list, total_training_time, 
[len(i) for i in clients_list], total_entropy, comm_overhead_list, macro_precision, macro_recall, macro_f1)), columns =['global_acc_list', 'global_loss_list', 'time_per_round', 'clients_no', 'total_entropy', 'comm_overhead', 'macro_precision', 'macro_recall', 'macro_f1'])



df.to_csv(f'{algo_s}_{dataset}.csv',index=False)

# plot loss on y-axis and communication round on x-axis

# %% 

# from matplotlib.cm import ScalarMappable

# # np.random.seed(12345)
# # df = pd.DataFrame([np.random.normal(32000, 200000, 3650),
# #                    np.random.normal(43000, 100000, 3650),
# #                    np.random.normal(43500, 140000, 3650),
# #                    np.random.normal(48000, 70000, 3650)],
# #                   index=[1992, 1993, 1994, 1995])
# # plt.style.use('ggplot'
# norm = plt.Normalize()
# colors = plt.cm.jet(norm(total_entropy))
# lower = np.array(total_entropy).min()
# upper = np.array(total_entropy).max()
# colors = plt.cm.jet((total_entropy-lower)/(upper-lower))

# def get_colors(inp, colormap, vmin=None, vmax=None):
#     norm = plt.Normalize(vmin, vmax)
#     return colormap(norm(inp))

# colors = get_colors(total_entropy, plt.cm.jet)


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# # plt.rcParams["figure.autolayout"] = True
# # df = pd.DataFrame(dict(data=[2, 4, 1, 5, 9, 6, 0, 7]))
# my_colors = [total-1 for total in total_entropy]
# # my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(df))] 
# fig, ax = plt.subplots()
# df['clients_no'].plot(kind='bar', color=colors,  secondary_y=True)
# df['global_acc_list'].plot(kind='line', marker='', color='red', ms=20)

# # put x axis name
# ax.set_xlabel("Communication round")
# # put y axis name
# ax.set_ylabel("Accuracy")
# ax.right_ax.set_ylabel("Number of clients")

# plt.show()

# %%



# os.chdir('/home/predicts-workbench/Projects/Diletta/FL/FL_clust/Risultati')



# #%%
# # save accuracy and loss to csv for model 


# FedAVG = pd.read_csv('FEDAVG_MNIST.csv')
# FedOPT = pd.read_csv('FEDOPT_MNIST.csv')
# FedNOVA = pd.read_csv('FEDNOVA_MNIST.csv')

# FedAVG_SMOTE = pd.read_csv('FEDAVG_MNIST_SMOTE.csv')
# FedOPT_SMOTE = pd.read_csv('FEDOPT_MNIST_SMOTE.csv')
# FedNOVA_SMOTE = pd.read_csv('FEDNOVA_MNIST_SMOTE.csv')

# FedAVG_GAN = pd.read_csv('FEDAVG_MNIST_GAN.csv')
# FedOPT_GAN = pd.read_csv('FEDOPT_MNIST_GAN.csv')
# FedNOVA_GAN = pd.read_csv('FEDNOVA_MNIST_GAN.csv')

# FedAVG_GAN_alto = pd.read_csv('FEDAVG_MNIST_GAN_lr_alto.csv')

# FedOPT_MNIST_GAN_1000epoche = pd.read_csv('FEDOPT_MNIST_GAN_1000epoche.csv')

# FedAVG_MNIST_GAN_1000epoche = pd.read_csv('FEDAVG_MNIST_GAN_1000epoche.csv')


# plt.figure(figsize=(20,10))
# plt.subplot(121)

# plt.plot(list(range(0,len(FedAVG['macro_f1']))), FedAVG['macro_f1'], label='FedAVG')
# plt.plot(list(range(0,len(FedAVG_GAN['macro_f1']))), FedAVG_GAN['macro_f1'], label='FedAVG_GAN')
# plt.plot(list(range(0,len(FedAVG_MNIST_GAN_1000epoche['macro_f1']))), FedAVG_MNIST_GAN_1000epoche['macro_f1'], label='FedAVG_MNIST_GAN_1000epoche')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# plt.plot(list(range(0,len(FedAVG['global_loss_list']))), FedAVG['global_loss_list'], label='FedAVG')
# plt.plot(list(range(0,len(FedAVG_GAN['global_loss_list']))), FedAVG_GAN['global_loss_list'], label='FedAVG_GAN')
# plt.plot(list(range(0,len(FedAVG_MNIST_GAN_1000epoche['global_loss_list']))), FedAVG_MNIST_GAN_1000epoche['global_loss_list'], label='FedAVG_MNIST_GAN_1000epoche')

# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# # FOCUS ON THE LAST 10 COMMUNICATION ROUNDS

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.plot(list(range(0,20)), FedAVG[30:]['macro_f1'], label='FedAVG')   
# plt.plot(list(range(0,20)), FedAVG_GAN[30:]['macro_f1'], label='FedAVG_GAN')
# plt.plot(list(range(0,20)), FedAVG_MNIST_GAN_1000epoche[30:]['macro_f1'], label='FedAVG_MNIST_GAN_1000epoche')

# # x scale from 30 to 50
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()













# #compare the accuracy and loss of the three

# plt.figure(figsize=(20,10))
# plt.subplot(121)

# # plt.plot(list(range(0,len(FedOPT['macro_f1']))), FedOPT['macro_f1'], label='FEDOPT')
# plt.plot(list(range(0,len(FedOPT_GAN['macro_f1']))), FedOPT_GAN['macro_f1'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedOPT_MNIST_GAN_1000epoche['macro_f1']))), FedOPT_MNIST_GAN_1000epoche['macro_f1'], label='FEDOPTGAN_1000')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# # plt.plot(list(range(0,len(FedOPT['global_loss_list']))), FedOPT['global_loss_list'], label='FEDOPT')
# plt.plot(list(range(0,len(FedOPT_GAN['global_loss_list']))), FedOPT_GAN['global_loss_list'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedOPT_MNIST_GAN_1000epoche['global_loss_list']))), FedOPT_MNIST_GAN_1000epoche['global_loss_list'], label='FEDOPTGAN_1000')

# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()



# # COMPARE ACCURACY AND LOSS OF THE THREE
# plt.plot(list(range(0,len(FedOPT_GAN['comm_overhead']))), FedOPT_GAN['comm_overhead'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedOPT_MNIST_GAN_1000epoche['comm_overhead']))), FedOPT_MNIST_GAN_1000epoche['comm_overhead'], label='FEDOPTGAN_1000')
# plt.xlabel('Communication round')
# plt.ylabel('comm_overhead')
# plt.legend()


# plt.plot(list(range(0,len(FedOPT_SMOTE['macro_precision']))), FedOPT_SMOTE['macro_precision'], label='FEDOPT_smote')
# plt.plot(list(range(0,len(FedOPT_MNIST_GAN_1000epoche['macro_precision']))), FedOPT_MNIST_GAN_1000epoche['macro_precision'], label='FEDOPTGAN_1000')
# plt.xlabel('Communication round')
# plt.ylabel('macro_precision')
# plt.legend()


# # check the number of clients for round is the same for fedopt, fedoptgan and fedoptgan_1000

# for i in range(0, len(FedOPT['clients_no'])):
#   if FedOPT['clients_no'][i] != FedOPT_GAN['clients_no'][i] or FedOPT['clients_no'][i] != FedOPT_MNIST_GAN_1000epoche['clients_no'][i]:
#     print(i)
#     break

# # plot a barplot of the number of clients for each round and above the line plot of macro_f1

# plt.figure(figsize=(25,10))
# plt.subplot(121)
# plt.bar(list(range(0,len(FedOPT['clients_no']))), FedOPT['clients_no'])
# plt.plot(list(range(0,len(FedOPT['macro_f1']))), FedOPT['macro_f1'], label='FEDOPT')
# plt.plot(list(range(0,len(FedOPT_GAN['macro_f1']))), FedOPT_GAN['macro_f1'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedOPT_MNIST_GAN_1000epoche['macro_f1']))), FedOPT_MNIST_GAN_1000epoche['macro_f1'], label='FEDOPTGAN_1000')
# # logscale
# plt.yscale('log')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# fig, ax = plt.subplots(figsize=(25,10))
# FedOPT['clients_no'].plot(kind='bar', ax=ax, alpha = 0.3)
# ax2 = plt.twinx(ax = ax)
# FedOPT['macro_f1'].plot(kind='line', ax=ax2, color='red')
# # FedOPT_SMOTE['macro_f1'].plot(kind='line', ax=ax2, color='blue')
# FedOPT_MNIST_GAN_1000epoche['macro_f1'].plot(kind='line', ax=ax2, color='green')
# ax2.set_ylabel('F1-score')
# ax.set_ylabel('Number of clients')
# plt.legend()
# plt.show()

# # compute the nearest neighbour adverarial accuracy between true dataset and generated dataset


# X_vero = clients_train['clients_0']
# y_vero = clients_label['clients_0']




# X_vero.shape
# y_vero.shape

# latent_dim = 100

#     # create the discriminator
# d_model = define_discriminator()
# # create the generator
# g_model = define_generator(latent_dim)
# # create the gan
# gan_model = define_gan(g_model, d_model)
# # load image data
# dataset = load_real_samples(X_to_oversample, y_to_oversample)
# # DataFrame' object has no attribute 'tolist' resolve error


# # train model
# history = train(g_model, d_model, gan_model, dataset, latent_dim)
# generatore = history[1]


# x_input = np.random.randn(latent_dim * tot)
# z_input = x_input.reshape(tot, latent_dim)
# labels = np.array([i for i in range(10) for j in range(sampling_strategy[i])])

# # count how many labels are generated for each class
# # np.unique(labels, return_counts=True)

# X_resampled = generatore.predict([z_input, labels])
# # len(X_resampled)
# X_resampled = (X_resampled + 1) / 2.0
# # flatten the images
# X_resampled = X_resampled.reshape(X_resampled.shape[0], 28*28)

# #associate the labels to the generated data
# y_resampled = labels

# X_resampled.shape
# y_resampled.shape

# # take the same number as from the original dataset
# X_resampled = X_resampled[:len(X_vero)]
# y_resampled = y_resampled[:len(y_vero)]

# true_dataset = X_vero
# generated_dataset = X_resampled
# # compute the nearest neighbour adverarial accuracy between true dataset and generated dataset



# Rtr = X_to_oversample

# Rtr = Rtr[:len(X_vero)]
# Rtr.shape

# Rte = X_vero
# Rte.shape

# X_resampled.shape


# AAtr_ge = NNAA(Rtr, X_resampled)

# AAtE_ge = NNAA(Rte, X_resampled)

# privacy_loss = np.mean(AAtE_ge - AAtr_ge)

# #à prova con SMOTE

# sm = SMOTE(random_state=0, sampling_strategy=sampling_strategy)
# X_SMOTE ,_= sm.fit_resample(X_to_oversample, df_to_oversample_final['label'])

# X_SMOTE.shape

# X_SMOTE = X_SMOTE[:len(X_vero)]
# X_SMOTE.shape


# AAtr_ge = NNAA(Rtr, X_SMOTE)

# AAtE_ge = NNAA(Rte, X_SMOTE)

# privacy_loss = np.mean(AAtE_ge - AAtr_ge)


# # plot the first image from x_resampled. x_vero and x_smote, x_to_oversample

# plt.figure(figsize=(20,10))
# plt.subplot(141)
# plt.imshow(X_resampled[0].reshape(28,28), cmap='gray')
# plt.title('GAN')

# plt.subplot(142)
# plt.imshow(X_SMOTE[0].reshape(28,28), cmap='gray')
# plt.title('SMOTE')

# plt.subplot(143)
# plt.imshow(X_vero[0].reshape(28,28), cmap='gray')
# plt.title('X_vero')

# plt.subplot(144)
# plt.imshow(X_to_oversample[0].reshape(28,28), cmap='gray')
# plt.title('X_to_oversample')

# plt.show()






# #compare the accuracy and loss of the three

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.plot(list(range(0,len(FedAVG['macro_f1']))), FedAVG['macro_f1'], label='FEDAVG')
# plt.plot(list(range(0,len(FedOPT['macro_f1']))), FedOPT['macro_f1'], label='FEDOPT')
# plt.plot(list(range(0,len(FedNOVA['macro_f1']))), FedNOVA['macro_f1'], label='FEDNOVA')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# plt.plot(list(range(0,len(FedAVG['global_loss_list']))), FedAVG['global_loss_list'], label='FEDAVG')
# plt.plot(list(range(0,len(FedOPT['global_loss_list']))), FedOPT['global_loss_list'], label='FEDOPT')
# plt.plot(list(range(0,len(FedNOVA['global_loss_list']))), FedNOVA['global_loss_list'], label='FEDNOVA')

# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()


# # FEDAVG

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.plot(list(range(0,len(FedAVG['macro_f1']))), FedAVG['macro_f1'], label='FEDAVG')
# plt.plot(list(range(0,len(FedAVG_SMOTE['macro_f1']))), FedAVG_SMOTE['macro_f1'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['macro_f1']))), FedAVG_GAN['macro_f1'], label='FEDAVG_GAN')
# plt.plot(list(range(0,len(FedAVG_GAN_alto['macro_f1']))), FedAVG_GAN_alto['macro_f1'], label='FEDAVG_GAN_alto')
# #LOGSCALE
# plt.yscale('log')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# plt.plot(list(range(0,len(FedAVG['global_loss_list']))), FedAVG['global_loss_list'], label='FEDAVG')
# plt.plot(list(range(0,len(FedAVG_SMOTE['global_loss_list']))), FedAVG_SMOTE['global_loss_list'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['global_loss_list']))), FedAVG_GAN['global_loss_list'], label='FEDAVG_GAN')
# plt.plot(list(range(0,len(FedAVG_GAN_alto['global_loss_list']))), FedAVG_GAN_alto['global_loss_list'], label='FEDAVG_GAN_alto')
# plt.yscale('log')
# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()


# # %%
# # SMOTE VS GAN

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# # plt.plot(list(range(0,len(FedAVG['macro_f1']))), FedAVG['macro_f1'], label='FEDAVG')
# plt.plot(list(range(0,len(FedAVG_SMOTE['macro_f1']))), FedAVG_SMOTE['macro_f1'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['macro_f1']))), FedAVG_GAN['macro_f1'], label='FEDAVG_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# # plt.plot(list(range(0,len(FedAVG['global_loss_list']))), FedAVG['global_loss_list'], label='FEDAVG')
# plt.plot(list(range(0,len(FedAVG_SMOTE['global_loss_list']))), FedAVG_SMOTE['global_loss_list'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['global_loss_list']))), FedAVG_GAN['global_loss_list'], label='FEDAVG_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# # ORIGIN VS GAN

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.plot(list(range(0,len(FedAVG['macro_f1']))), FedAVG['macro_f1'], label='FEDAVG')
# # plt.plot(list(range(0,len(FedAVG_SMOTE['macro_f1']))), FedAVG_SMOTE['macro_f1'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['macro_f1']))), FedAVG_GAN['macro_f1'], label='FEDAVG_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# plt.plot(list(range(0,len(FedAVG['global_loss_list']))), FedAVG['global_loss_list'], label='FEDAVG')
# # plt.plot(list(range(0,len(FedAVG_SMOTE['global_loss_list']))), FedAVG_SMOTE['global_loss_list'], label='FEDAVG_SMOTE')
# plt.plot(list(range(0,len(FedAVG_GAN['global_loss_list']))), FedAVG_GAN['global_loss_list'], label='FEDAVG_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()


# # ALL GAN

# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.plot(list(range(0,len(FedAVG_GAN['macro_f1']))), FedAVG_GAN['macro_f1'], label='FEDAVG_GAN')
# plt.plot(list(range(0,len(FedOPT_GAN['macro_f1']))), FedOPT_GAN['macro_f1'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedNOVA_GAN['macro_f1']))), FedNOVA_GAN['macro_f1'], label='FEDAVG_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('F1-score')
# plt.legend()

# plt.subplot(122)
# plt.plot(list(range(0,len(FedAVG_GAN['global_loss_list']))), FedAVG_GAN['global_loss_list'], label='FEDAVG_GAN')
# plt.plot(list(range(0,len(FedOPT_GAN['global_loss_list']))), FedOPT_GAN['global_loss_list'], label='FEDOPT_GAN')
# plt.plot(list(range(0,len(FedNOVA_GAN['global_loss_list']))), FedNOVA_GAN['global_loss_list'], label='FEDNOVA_GAN')
# plt.xlabel('Communication round')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# # compute training time in hours

# # FedAVG
# np.round(FedAVG['time_per_round'].sum()/3600, 2) # it took 1 hour and 15 minutes

# # FedAVG_GAN
# np.round(FedAVG_GAN['time_per_round'].sum()/3600, 2) # it took 6 hours and 44 minutes

# # FedAVG_SMOTE
# np.round(FedAVG_SMOTE['time_per_round'].sum()/3600, 2) # it took 1 hour and 64 minutes

# # FedOPT
# np.round(FedOPT['time_per_round'].sum()/3600, 2) # it took 0.91 hours i.e 54 minutes

# # FedOPT_GAN
# np.round(FedOPT_GAN['time_per_round'].sum()/3600, 2) # it took 5 hours and 51 minutes

# # FedOPT_SMOTE
# np.round(FedOPT_SMOTE['time_per_round'].sum()/3600, 2) # it took 2 hour and 05 minutes

# # FedNOVA
# np.round(FedNOVA['time_per_round'].sum()/3600, 2) # it took 1 hour and 15 minutes

# # FedNOVA_GAN
# np.round(FedNOVA_GAN['time_per_round'].sum()/3600, 2) # it took 6 hours and 44 minutes

# # FedNOVA_SMOTE
# np.round(FedNOVA_SMOTE['time_per_round'].sum()/3600, 2) # it took 1 hour and 64 minutes

# # FedAVG_GAN_alto 
# np.round(FedAVG_GAN_alto['time_per_round'].sum()/3600, 2) # it took 5 hours and 91 minutes






# # %% 
# # read data from csv    
# data8 = pd.read_csv('MNIST_Non-IID_path_imbalanced_8classperclient.csv')
# data6 = pd.read_csv('MNIST_Non-IID_path_imbalanced_6classperclient.csv')
# data4 = pd.read_csv('MNIST_Non-IID_path_imbalanced_4classperclient.csv')
# data2 = pd.read_csv('MNIST_Non-IID_path_imbalanced_2classperclient.csv')
# # # plot the data

# # compare the accuracy and loss of the two models
# plt.figure(figsize=(16,4))
# plt.subplot(121)
# plt.plot(list(range(0,len(data6['global_acc_list']))), data6['global_acc_list'], label='6 classes per client')
# plt.plot(list(range(0,len(data4['global_acc_list']))), data4['global_acc_list'], label='4 classes per client')
# plt.plot(list(range(0,len(data2['global_acc_list']))), data2['global_acc_list'], label='2 classes per client')
# plt.plot(list(range(0,len(data8['global_acc_list']))), data8['global_acc_list'], label='8 classes per client')
# plt.xlabel('Communication Rounds')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.subplot(122)
# plt.plot(list(range(0,len(data6['global_loss_list']))), data6['global_loss_list'], label='6 classes per client')
# plt.plot(list(range(0,len(data4['global_loss_list']))), data4['global_loss_list'], label='4 classes per client')
# plt.plot(list(range(0,len(data2['global_loss_list']))), data2['global_loss_list'], label='2 classes per client')
# plt.plot(list(range(0,len(data8['global_loss_list']))), data8['global_loss_list'], label='8 classes per client')
# #set x and y labels
# plt.xlabel('Communication Rounds')
# plt.ylabel('Loss')
# plt.legend()

# #
# # %%


# plt.figure(figsize=(16,4))
# plt.subplot(121)
# plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
# plt.subplot(122)
# plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
# print('Non-IID | total comm rounds', len(global_acc_list))           

# # %%

