# %% 
import numpy as np
import tqdm as tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, balanced_accuracy_score, multilabel_confusion_matrix, classification_report
import tensorflow as tf
import time
from scipy import stats
from tensorflow.keras import backend as K 

debug = 0
batch_size = 32
train_size = 0.75
least_no_samples = batch_size / (1-train_size) # least number of samples in a shard/client
alpha = 0.1 # for Dirichlet distribution


# no_clients = 20
# no_classes = 10
# niid=True
# balance=False
# partition='pathological'
# class_per_client=6
# initial='clients'

# %% 
# define a function to create shards


def create_shards_function(image_list, label_list, no_clients, no_classes, niid=False, balance=False, 
partition=None, class_per_client=4, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                    data shards - tuple of images and label lists.
            args: 
                image_list: a list of numpy arrays of training images
                label_list:a list of binarized labels for each image
                no_clients: number of federated members (clients)
                no_classes : number of classes in the dataset
                niid: if True, data is non-i.i.d. - i.e. data is not uniformly distributed across clients - niid is True equals if not niid
                balance: if True, data is balanced across clients
                partition: if not None, data is partitioned according to the partition arg
                class_per_client: number of classes per client if niid is True
                initials: the clients'name prefix, e.g, clients_1 
                
        '''
    # create a list of clients' names
    client_names = ['{}_{}'.format(initial, i) for i in range(no_clients)]

    X_fed = [[] for _ in range(no_clients)] # create an empty list of lists for the shards
    y_fed = [[] for _ in range(no_clients)] # create an empty list of lists for the shards
    statistics = [[] for _ in range(no_clients)] # create an empty list of lists for the statistic of the shards

    # dataset_content, dataset_label = zip(*dataset) # so data should have two elements, the first is the content and the second is the label
    # data should be a list

    dataidx_map = {}

    if not niid: # not niid 
        # check non niid is True
       partition = 'pathological' 
       # if niid is False, not niid is True, then we have non-iid setting, and we have two situations: 
       # pathological and practical
       class_per_client = no_classes # I have to input something before I can use it
    
    if partition == 'pathological':
        # not hoe label
        idxs = np.array(range(len(label_list)))
        # create empty list for index for each label - len(idxs) = no_classes
        # plt.xlabel('Label')
        idx_for_class = []
 
        for cls in range(no_classes):
            idx_for_class.append([idx for idx in idxs if label_list[idx] == cls])
        
         # for i in range(no_classes):
         #     print('Label: {} Count: {}'.format(i, len(idx_for_class[i])))
         # Label: 0 Count: 6903
         # Label: 1 Count: 7877
         # Label: 2 Count: 6990
         # Label: 3 Count: 7141
         # Label: 4 Count: 6824
         # Label: 5 Count: 6313
         # Label: 6 Count: 6876
         # Label: 7 Count: 7293
         # Label: 8 Count: 6825
         # Label: 9 Count: 6958

            # # plot idx_for_class to see the distribution of the samples for each label
            # plt.figure(figsize=(10, 5))
            # plt.bar(range(no_classes), [len(idx_for_class[i]) for i in range(no_classes)])
            # plt.title('Distribution of samples for each label')
            # plt.xlabel('Label')
            # plt.ylabel('Number of samples')
            # plt.show()

            # check labels and indices
            # for i in range(no_classes):
            #     print(f"Label {i} has {len(idx_for_class[i])} samples")

        class_no_per_client = [class_per_client for _ in range(no_clients)]

        np.random.seed(0)
        rng = np.random.default_rng(0)
        # create a list of class_per_client for each client (so len no_clients). the list is long no_clients and each element is class_per_client
        # this list will be updated in the for loop below
        for i in range(no_classes): #break
            selected_clients = []
            for client in range(no_clients):
                if class_no_per_client[client] > 0:
                    selected_clients.append(client)
                # I have appended the client to the selected_clients list if the number of classes for that client is greater than 0
                # selected clients is a list long no_clients, and it contains the clients that have classes left to be assigned
        
                selected_clients = selected_clients[:int(no_clients/no_classes*class_per_client)]
                # I have selected only the first int(no_clients/no_classes*class_per_client) clients from the selected_clients list

    
            no_all_samples = len(idx_for_class[i]) # number of all samples with label i
            no_selected_clients = len(selected_clients) # number of clients selected - decremental number because of the for loop
            # no_samples_per_client = int(np.ceil(no_all_samples/no_selected_clients)) # number of samples per client for label i

            # np.random.seed(0)
            if balance: # i.e. if balance is True
                
                no_samples_per_client = no_all_samples // no_selected_clients # number of samples per client for label i
                no_clients_with_extra_sample = no_all_samples % no_selected_clients # number of clients with extra sample for label i
            
                no_samples = np.array([
                    no_samples_per_client + 1 if client < no_clients_with_extra_sample else no_samples_per_client
                    for client in range(no_selected_clients)])
                np.random.shuffle(no_samples)

                assert sum(no_samples) == no_all_samples

                
                # all clients have the same number of samples 
                # sum(no_samples)
            else:
    
                # no_samples = np.random.randint(max(no_samples_per_client/5, least_no_samples/no_classes), 
                # no_samples_per_client, no_selected_clients-1).tolist()
                # the number of samples for each client is randomly selected from a range between max(no_samples_per_client/10, least_no_samples/no_classes) and no_samples_per_client
                # no_samples.append(no_all_samples - sum(no_samples))
            # the last client has the remaining number of samples
            # sum(no_samples) == no_all_samples # True

                lower, upper = 0.15, 0.85
                probs = stats.truncnorm.rvs(lower, upper, size=no_selected_clients, random_state=rng)
                # probs = np.random.normal(0.5, 0.2, no_selected_clients)

                # probs[probs<0.2] = no_selected_clients/no_clients * 0.5

                # probs = probs/sum(probs)
                # no_samples = (probs * no_all_samples ).astype('int')

                # barplot no_samples
                # plt.figure(figsize=(10, 5))
                # plt.bar(range(no_selected_clients), no_samples)
                probs /= sum(probs)
                no_samples = np.round((probs * no_all_samples )).astype('int')


                if sum(no_samples) < no_all_samples:
                    # Compensate the missing values by increasing by 1 the top clients by their current number
                    N = no_all_samples - sum(no_samples)
                    no_samples[np.argpartition(no_samples, -N)[-N:]] += 1
                elif sum(no_samples) > no_all_samples:
                    # Remove the excessive values by decreasing by 1 the bottom clients by their current number
                    N = sum(no_samples) - no_all_samples
                    no_samples[np.argpartition(no_samples, N)[:N]] -= 1

                # print(i)
                # print(selected_clients)
                # print(no_samples)
                # print(stats.entropy(no_samples))
                assert sum(no_samples) == no_all_samples
                # barplot no_samples
                # plt.figure(figsize=(10, 5))
                # plt.bar(range(no_selected_clients), no_samples)

            idx=0

            for client, no_sample in zip(selected_clients, no_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_class[i][idx:idx+no_sample]
                    # dataidx_map is a dictionary, where the key is the client among selected clients and the value is the indexes of the samples for that client
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_class[i][idx:idx+no_sample], axis=0)
                    # if the client is already in the dataidx_map, then I append the indexes of the samples for that client to the value of the client
                
                idx += no_sample
                # idx is the index of the first sample for the next client - 
                class_no_per_client[client] -= 1
                # I have decreased the number of classes for that client by 1

    elif partition=='practical':
        # print something to check the code is running correctly
        print('Practical partitioning')
        min_size = 20
        K = no_classes
        N = len(label_list)

        while min_size < least_no_samples:
            a = f'{min_size}, {least_no_samples}'
            idx_batch = [[] for _ in range(no_clients)]
            # len(idx_batch) == no_clients # True
            # use tqdm to show the progress bar
            for k in range(K):
                # time.sleep(0.1)
                # print(f'\r{a}k = ', k, end='')
                # dataset_label is a list of labels, in order to apply the np.where function, I have to convert it to a numpy array
                idx_k = np.where(np.array(label_list) == k)[0]
                # idx_k is a list of indexes of the samples with label k
                np.random.seed(0)
                np.random.shuffle(idx_k)
                # I have shuffled the indexes of the samples with label k

                # should I set the seed? 
                
                # set the seed
                np.random.seed(0)
                shares = np.random.dirichlet(np.repeat(alpha, no_clients))
                # I have created a list of proportions for each client with the dirichlet distribution with parameter alpha and size no_clients 
                # np.repeat(alpha, no_clients) is a list of alpha repeated no_clients times
                shares = np.array([p*(len(idx_j)<N/no_clients) for p, idx_j in zip(shares, idx_batch)])
                # I have created a list of proportions for each client, where the proportion is 0 if the number of samples for that client is greater than N/no_clients
                shares = shares/shares.sum()
                # I have normalized the proportions
                # shares.sum() == 1 # True
                shares = (np.cumsum(shares)*len(idx_k)).astype(int)[:-1]
                # I have created a list of indexes, where the index is the number of samples for each client
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, shares))]
                # I have created a list of lists, where each list contains the indexes of the samples for each client
                # len(idx_batch) == no_clients # True    

                # # count elements in each list
                # for i in range(no_clients):
                #     print(len(idx_batch[i]))

                min_size = min([len(idx_j) for idx_j in idx_batch])
                # min_size is the minimum number of samples for all clients

        
        for client in range(no_clients):
            dataidx_map[client] = idx_batch[client]
            # dataidx_map is a dictionary, where the key is the client and the value is the indexes of the samples for that client

    else: 
        raise ValueError('Unknown partition method: {}'.format(partition))
    

    # assign the data
    dataset_content_array = np.stack(image_list, axis=0)
    dataset_label_array = np.stack(label_list, axis=0)

    # dataset_content_array.shape
    # dataset_label_array.shape

    for client in range(no_clients):
        idxs = dataidx_map[client]
        # idxs is a list of indexes of the samples for that client 
        
        X_fed[client] = dataset_content_array[np.array(idxs, dtype=int)]
        # I want y_fed for client to be long no_classes even if the client has less than no_classes classes
        y_fed[client] = dataset_label_array[np.array(idxs, dtype=int)]
    

        # # visualize first 5 images of client with label
        # for i in range(5):
        #     # plot reshaped image of client
        #     plt.subplot(1, 5, i+1)
        #     plt.imshow(X_fed[client][i].reshape(28, 28), cmap='gray')
        #     plt.ylabel(y_fed[client][i])
        #     plt.show()

            #    for i in range(5):
            # # plot reshaped image of client
            # plt.subplot(1, 5, i+1)
            # sns.violinplot(X_fed[client][i])
            # plt.ylabel(y_fed[client])
            # plt.show()


        for i in np.unique(y_fed[client]): 
            # if is a single class, then np.unique(y_fed[client]) is a list with one element
            # and I cannot sum a list of booleans
            # if is a list of classes, then np.unique(y_fed[client]) is a list with more than one element
            # and I can sum a list of booleans
            if len(np.unique(y_fed[client])) == 1:
                statistics[client].append((int(i), np.sum(y_fed[client]==i) ))
            statistics[client].append((int(i), int(sum(y_fed[client]==i) )))
            # statistics is a list of lists, where each list contains the number of samples for each class for each client


    # del dataset


    for client in range(no_clients):
        print(f"Client {client}\t size of data: {len(X_fed[client])}\t labels:", np.unique(y_fed[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistics[client]])
        print("-"*100)
    
    shards = [(X_fed[client], y_fed[client]) for client in range(no_clients)]

    return {client_names[i] : shards[i] for i in range(no_clients)}
    # return {client_names[i]: (X_fed[i], y_fed[i]) for i in range(len(no_clients))}



############################################################################################################
# %%

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    # bs = clients_trn_data[client_name].shape[0]
    bs = clients_trn_data[client_name][0].shape[0]
    #first calculate the total training data points across clients
    global_count = sum([len(clients_trn_data[client_name]) for client_name in client_names])*bs
    # global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name].tolist()) for client_name in client_names])*bs
    # get the total number of data points held by a client
    # local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    local_count = len(clients_trn_data[client_name])*bs
    
    
    if debug:
        print('global_count', global_count, 'local_count', local_count, 'bs', bs)
    
    return local_count/global_count

# for i in range(10):
#     print(weight_scalling_factor(clients_train, f'clients_{i}'))
#     # sum all the weights scaled by the weight_scalling_factor

# [sum([weight_scalling_factor(clients_train, f'clients_{i}') for i in range(10)])]

# weights sum to 1!
    
# print weights of global model


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)   # len(global_model.get_weights()) == 6
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final # shape of weight_final is (6,)



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list): 
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad



def sum_weights(weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*weight_list): 
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

# scaled_local_weight_list = []
# for client in clients.keys():
#     scaling_factor = weight_scalling_factor(clients_trn_data, client)
#     # print scaling factor for each client
#     print('client', client, 'scaling_factor', scaling_factor)
#     # print sum of all scaling factors
#     assert sum([weight_scalling_factor(clients_trn_data, client) for client in clients.keys()])

#     scaled_weights = scale_model_weights(global_model.get_weights(), scaling_factor)

#     scaled_local_weight_list.append(scaled_weights)


# average_weights = sum_scaled_weights(scaled_local_weight_list)

dataset = 'california_housing'
# check all scald weigths sum to 1

def test_model(X_test, Y_test,  model, comm_round, dataset):
    # see time taken to test model for each round
    start_time = time.time()

    if dataset == 'california_housing':
        mse = tf.keras.losses.MeanSquaredError()
        predictions = model.predict(X_test)
        loss = mse(Y_test, predictions)
        mae = K.mean(tf.keras.metrics.mean_absolute_error(Y_test, predictions), axis=-1)
        mse = K.mean(tf.keras.metrics.mean_squared_error(Y_test, predictions), axis=-1)
        mape = K.mean(tf.keras.metrics.mean_absolute_percentage_error(Y_test, predictions), axis=-1)

        end_time = time.time()
        time_taken = end_time-start_time

        print(f'Round: {comm_round} \t Test Loss: {loss} \t Test MAE: {mae} \t Test MSE: {mse} \t Test MAPE: {mape} \t Time Taken: {time_taken}')
        return loss, mae, mse, mape, time_taken
    
    

    else:
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(X_test)
        loss = cce(Y_test, logits)
        acc = accuracy_score(tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
        # bal_acc = balanced_accuracy_score(tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))

        class_report = classification_report(tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1), output_dict=True)
        # print(class_report)
        conf_mat = multilabel_confusion_matrix(tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
        end_time = time.time()
        time_taken = end_time-start_time
        print('comm_round: {} | global_acc: {:.3%}|  class_report: {} | conf_mat: {} | time_taken: {:.3f} sec'.format(comm_round, acc, class_report, conf_mat, time_taken))
        
    
        return acc, class_report, conf_mat, loss, time_taken

    # return acc, loss, time_taken


# Optimizator
def get_optimizer(name, learning_rate, **kwargs):
    if name == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
    elif name == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
    elif name == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, **kwargs)
    elif name == 'Adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, **kwargs)
    elif name == 'Adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, **kwargs)
    elif name == 'Adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate, **kwargs) 
    elif name == 'Nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError('Optimizer not supported')
    return opt


def NNAA(X_true, y_true, X_gen, y_gen):
    no_classes = len(np.unique(y_true))

    w = np.ones(len(X_true[0]))

    w = np.concatenate([w, np.sqrt(len(w))*np.ones(no_classes)])
    
    # w.shape

    # dts = euclidean_distances(true_dataset, generated_dataset) 
    dts_min = []    

    # assert len(true_dataset) == len(generated_dataset)
    # i= 0
    for i in range(len(X_true)): 

        # w is a vector of weights whose last 10 elements are equal to the sqrt of the number of elements in the vector

        imm = np.concatenate([X_true[i], np.arange(no_classes)==y_true[i]]) 
        
        imm_gen = [np.concatenate([X_gen[j], np.arange(no_classes)==y_gen[j]]) for j in range(len(X_true))]
      
        # rmse = np.sqrt(sum((X_true[i]- X_gen)**2*w) / sum(w))
        rmse_dts = [np.sqrt(sum((imm- imm_gen[j])**2*w) / sum(w)) for j in range(len(X_true))]
        # len rmse = len(X_true)
        
        # np.where(rmse == np.min(rmse_dts)) # index of the minimum rmse
        # plot the image with the minimum rmse and associated X_true image
        # plt.imshow(X_gen[np.where(rmse == np.min(rmse))].reshape(28,28))
        # plt.imshow(X_true[np.where(rmse == np.min(rmse))].reshape(28,28))

        # FUNNY THING: the image with the minimum rmse is with different label from the X_true image

        # index of max rmse
        # np.where(rmse == np.max(rmse))
        # plt.imshow(X_gen[np.where(rmse == np.max(rmse))].reshape(28,28))

        dts_min.append(np.min(rmse_dts))   


    # dts = np.linalg.norm(true_dataset - generated_dataset, axis=1)
    # take the minimum distance between the true dataset and the generated dataset
    dts = np.array(dts_min)


    dst_min = []  

    for i in range(len(X_true)):
        # dst = []
        # for j in range(len(generated_dataset)):
        #     dst.append(np.linalg.norm(generated_dataset[i] - true_dataset[j], axis=0))
        # dst = np.linalg.norm(generated_dataset[i] - true_dataset, axis=1)
        # dst_max.append(np.min(dst))      

        imm_gen = np.concatenate([X_gen[i], np.arange(no_classes)==y_gen[i]]) 
        
        imm = [np.concatenate([X_true[j], np.arange(no_classes)==y_true[j]]) for j in range(len(X_true))]
        

        # rmse = np.sqrt(sum((X_true[i]- X_gen)**2*w) / sum(w))
        rmse_dst = [np.sqrt(sum((imm_gen- imm[j])**2*w) / sum(w)) for j in range(len(X_true))]
        # len rmse = len(X_true)

        # np.where(rmse == np.min(rmse_dst)) # index of the minimum rmse
        # plot the image with the minimum rmse and associated X_true image
        # plt.imshow(X_gen[np.where(rmse_dst == np.min(rmse_dst))].reshape(28,28))
        # plt.imshow(X_true[np.where(rmse_dst == np.min(rmse_dst))].reshape(28,28))
        dst_min.append(np.min(rmse_dst)) 
        
    # dts = np.linalg.norm(true_dataset - generated_dataset, axis=1)
    # take the minimum distance between the true dataset and the generated dataset
    dst = np.array(dst_min)
    

    # dtt = euclidean_distances(true_dataset, true_dataset) except the diagonal
    dtt_min = []

    for i in range(len(X_true)):
        # dtt = []
    
    # compute rmse except when last no_classes elements are equal 
    
        imm = np.concatenate([X_true[i], np.arange(no_classes)==y_true[i]])
        
        rmse_dtt = [np.sqrt(sum((imm- np.concatenate([X_true[j], np.arange(no_classes)==y_true[j]]))**2*w) / sum(w)) for j in range(len(X_true)) if not np.array_equal(imm[-no_classes:], np.concatenate([X_true[j], np.arange(no_classes)==y_true[j]])[-no_classes:])]

        # check wheter the minimum rmse is equal to 0
        # np.where(rmse == np.min(rmse_dtt)) # index of the minimum rmse

        # plot the image with the minimum rmse and associated X_true image
        # plt.imshow(X_true[np.where(rmse_dtt == np.min(rmse_dtt))].reshape(28,28))
        dtt_min.append(np.min(rmse_dtt))

    dtt = np.array(dtt_min)


    dss_min = []

    for i in range(len(X_true)):

        imm_gen = np.concatenate([X_gen[i], np.arange(no_classes)==y_gen[i]])
        
        rmse_dss = [np.sqrt(sum((imm_gen- np.concatenate([X_gen[j], np.arange(no_classes)==y_gen[j]]))**2*w) / sum(w)) for j in range(len(X_true)) if not np.array_equal(imm[-no_classes:], np.concatenate([X_gen[j], np.arange(no_classes)==y_gen[j]])[-no_classes:])]

        # check wheter the minimum rmse is equal to 0
        # np.where(rmse == np.min(rmse_dss)) # index of the minimum rmse

        # plot the image with the minimum rmse and associated X_true image
        # plt.imshow(X_true[np.where(rmse_dtt == np.min(rmse_dtt))].reshape(28,28))
        dss_min.append(np.min(rmse_dtt))

    dss = np.array(dss_min)


    # AT counts how many time dts is greater than dtt for every element of the true dataset and divide by the number of elements of the true dataset
    AT = np.sum(dts > dtt) / len(X_true)

    # AS counts how many times dst is greater than dss for every element of the generated dataset and divide by the number of elements of the generated dataset
    AS = np.sum(dst > dss) / len(X_true)

    AATS = (AT + AS) / 2

    return AATS


# shuffle X_true and divide in half in X_tr

# reshape to 784

# samples_test = np.random.normal(0,1,(tot, 100))
# labels = np.array([i for i in range(10) for j in range(sampling_strategy[i])])
# labels_test = np.array([i for i in range(10) for j in range(sampling_strategy[i])]).reshape((-1, 1))
# results2 = g_model.predict([samples_test, labels_test])

# # normalize between 0 and 1
# results2 = (results2 - results2.min()) / (results2.max() - results2.min())


# # assert at least one element is different between results and results2
# assert not np.array_equal(results, results2)

# # plot the first 10 images of results and results2
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(results[i].reshape(28,28))
#     plt.subplot(2, 10, i+11)
#     plt.imshow(results2[i].reshape(28,28))


# X_to_oversample = X_to_oversample.reshape(X_to_oversample.shape[0], 784)
# results = results.reshape(results.shape[0], 784)
# results2 = results2.reshape(results2.shape[0], 784)

# X_tr, X_te, y_tr, y_te = train_test_split(X_to_oversample, y_to_oversample, test_size=0.5, random_state=42, stratify = y_to_oversample)
# # np.unique(y_tr, return_counts=True)
# # np.unique(y_te, return_counts=True)

# # GAN
# AA_tr_1 = NNAA(X_tr, y_tr, results, labels_test)
# AA_te_2 = NNAA(X_te, y_te, results2, labels_test)

# AA_tr_1 = np.round(AA_tr_1, 3)
# AA_te_2 = np.round(AA_te_2, 3)


# privacy_score = AA_te_2 -AA_tr_1
# np.mean(AA_tr_1 - AA_te_2)

# # SMOTE - X_resampled, y_resampled


# # assert at least one element is different between X_resampled and X_resampled2
# assert not np.array_equal(X_resampled, X_resampled2)

# # plot the first 10 images of X_resampled and X_resampled2
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(X_resampled[i].reshape(28,28))
#     plt.subplot(2, 10, i+11)
#     plt.imshow(X_resampled2[i].reshape(28,28))

# # X_tr, X_te, y_tr, y_te = train_test_split(X_to_oversample, y_to_oversample, test_size=0.5, random_state=42, stratify = y_to_oversample)
# # # X_gen1, X_gen2, y_gen1, y_gen2 = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=42, stratify = y_resampled)

# y_resampled = y_resampled.values.reshape((-1, 1))
# y_resampled2 = y_resampled2.values.reshape((-1, 1))

# AA_tr_1_SMOTE = NNAA(X_tr, y_tr, X_resampled, y_resampled.reshape((-1, 1)))
# AA_te_2_SMOTE = NNAA(X_te, y_te, X_resampled2, y_resampled2.reshape((-1, 1)))



# AA_tr_1_SMOTE = np.round(AA_tr_1_SMOTE, 3)
# AA_te_2_SMOTE = np.round(AA_te_2_SMOTE, 3)


# privacy_score = AA_te_2_SMOTE -AA_tr_1_SMOTE 

# # assert at least one element is different between X_resampled_adasyn and X_resampled_adasyn2

# assert not np.array_equal(X_resampled_adasyn, X_resampled_adasyn2)

# # plot the first 10 images of X_resampled_adasyn and X_resampled_adasyn2
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(X_resampled_adasyn[i].reshape(28,28))
#     plt.subplot(2, 10, i+11)
#     plt.imshow(X_resampled_adasyn2[i].reshape(28,28))


# y_resampled_adasyn = y_resampled_adasyn.values.reshape((-1, 1))
# y_resampled_adasyn2 = y_resampled_adasyn2.values.reshape((-1, 1))

# AA_tr_1_ADASYN = NNAA(X_tr, y_tr, X_resampled_adasyn, y_resampled_adasyn.reshape((-1, 1)))
# AA_te_2_ADASYN = NNAA(X_te, y_te, X_resampled_adasyn2, y_resampled_adasyn2.reshape((-1, 1)))

# AA_tr_1_ADASYN = np.round(AA_tr_1_ADASYN, 3)
# AA_te_2_ADASYN = np.round(AA_te_2_ADASYN, 3)

# privacy_score = AA_te_2_ADASYN -AA_tr_1_ADASYN

# # plot data divider per class of X_to_oversample, results, X_resampled, X_resampled_adasyn
# # for each feature, plot distplot of X_to_oversample, results, X_resampled, X_resampled_adasyn
# import seaborn as sns
# for i in range(784):
#     plt.figure(figsize=(20, 10))
#     # plt.subplot(2, 2, 1)
#     sns.distplot(X_to_oversample[:, i], label='X_to_oversample')
#     sns.distplot(results[:, i], label='results')
#     sns.displot(X_resampled[:, i], label='X_resampled')
#     sns.displot(X_resampled_adasyn[:, i], label='X_resampled_AdaSyn')

