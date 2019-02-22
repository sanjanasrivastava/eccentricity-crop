import itertools
import numpy as np
LIMIT_TESTS = False	# Toggle based on whether we want to run exhaustive experiments or a subset 


class Dataset(object):

    def __init__(self):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        # # #
        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        # # #
        # Set random labels
        self.random_labels = False
        self.scramble_data = False

        # # #
        # Transfer learning
        self.transfer_learning = False
        self.transfer_pretrain = True
        self.transfer_label_offset = 0
        self.transfer_restart_name = "_pretrain"
        self.transfer_append_name = ""

        # Find dataset path:
        for line in open("datasets/paths", 'r'):
            if 'Dataset:' in line:
                self.dataset_path = line.split(" ")[1].replace('\r', '').replace('\n', '')

    # function made by sanjana to alter amount of training data
    def set_num_train_ex(self, num_train_ex):
        self.proportion_training_set = float(num_train_ex) / 60000	# TODO currently hard-coded because Experiments currently invokes Dataset() but not MNIST() and I'm not sure how to get to MNIST specifically, but it appears to happen at some point  
    

    # # #
    # Dataset general
    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name

    # # #
    # Transfer learning
    def do_pretrain_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_append_name = self.transfer_restart_name

    def do_transfer_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_pretrain = True
        self.transfer_label_offset = 5
        self.transfer_append_name = "_transfer"


class DNN(object):	# TODO change for MNIST

    def __init__(self):
        self.name = "Alexnet"
        self.pretrained = False
        self.version = 1
        self.layers = 4
        self.stride = 2
        self.neuron_multiplier = np.ones([self.layers])

    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):	# TODO change for MNIST

    def __init__(self):
        self.batch_size = 2 
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.image_size = 28	# Changed for MNIST
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False
       
        # Params specific to this study, set to defaults
        self.background_size = 0
        self.num_train_ex = 2**3

    def set_background_size(self, background_size):
#         self.image_size += background_size
        self.background_size = background_size
        

class Experiments(object):

    def __init__(self, id, name):
        self.name = "base"
        self.log_dir_base = '/om2/user/sanjanas/eccentricity-data/models/'
            # '/om/user/sanjanas/eccentricity-data/models/' 
            #"/om/user/xboix/share/minimal-pooling/models/"
            #"/Users/xboix/src/minimal-cifar/log/"
            #"/om/user/xboix/src/robustness/robustness/log/"
            #"/om/user/xboix/src/robustness/robustness/log/"


        # Recordings
        self.max_to_keep_checkpoints = 5
        self.recordings = False
        self.num_batches_recordings = 0

        # Plotting
        self.plot_details = 0
        self.plotting = False

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()

    def do_recordings(self, max_epochs):
        self.max_to_keep_checkpoints = 0
        self.recordings = True
        self.hyper.max_num_epochs = max_epochs
        self.num_batches_recordings = 10

    def do_plotting(self, plot_details=0):
        self.plot_details = plot_details
        self.plotting = True

# # #
# Create set of experiments
opt = []
plot_freezing = []

# General hyperparameters
name = ['mnist_cnn']	# ["Alexnet"]
num_layers = [3]	# [5]
max_epochs = [20]	# [100]
learning_rates = [10**i for i in range(-6, 0)]	# 6 values
# batch_sizes = [2**i for i in range(8)] if not LIMIT_TESTS else [128]	# 8 values
batch_sizes = [1, 4, 10, 20, 40, 80, 160, 320] if not LIMIT_TESTS else [40]
initialization = None	# TODO 

# Experiment-specific hyperparameters
background_sizes = [0] + [int(28 * 2**i) for i in range(-3, 3)]	if not LIMIT_TESTS else [0, 7, 56]	# 7 values
num_train_exs = [2**i for i in range(3, 12)] + [5000]	# 10 values, last is full training set

idx = 0
for num_train_ex in num_train_exs:
    data = Experiments(idx, 'numtrainex' + str(num_train_ex)) 
#     data.dataset.set_num_train_ex(num_train_ex)
    opt.append(data)
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].hyper.num_train_ex = num_train_ex

    idx += 1


for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):

    for background_size in background_sizes:
        for data in opt[:len(num_train_exs)]:
            for batch_size in batch_sizes:	
                for learning_rate in learning_rates:
                    opt += [Experiments(idx, name_NN + '_' + str('backgroundsize') + str(background_size) + '_' + 'numtrainex' + str(data.hyper.num_train_ex))]
            
                    opt[-1].hyper.max_num_epochs = max_epochs_NN
                    opt[-1].hyper.background_size = background_size
                    opt[-1].hyper.num_train_ex = data.hyper.num_train_ex 
                    opt[-1].hyper.learning_rate = learning_rate
                    opt[-1].hyper.batch_size = batch_size

                    opt[-1].dnn.name = name_NN
                    opt[-1].dnn.set_num_layers(num_layers_NN)
                    opt[-1].dnn.neuron_multiplier.fill(3)
            
                    opt[-1].dataset.reuse_tfrecords(data) # opt[0])
                    opt[-1].hyper.max_num_epochs = int(max_epochs_NN)
                    opt[-1].hyper.num_epochs_per_decay = \
                        int(opt[-1].hyper.num_epochs_per_decay)   
             
                    idx += 1

'''
EXPERIMENT ID GUIDE
tfRecords writing for each <num_train_ex>: 0-9
batch_size=128, all <num_train_ex>, background_sizes=0,7,56: 

'''

def calculate_IDs(rbackground_size, rnum_train_ex, rbatch_size, rlearning_rate):

    data_offset = 10 

    BG = len(background_sizes)
    NTE = len(num_train_exs)
    BS = len(batch_sizes)
    LR = len(learning_rates)
    
    IDs, ID_subs, BG_adds = ([] for __ in range(3))
    for bg, nte, bs, lr in itertools.product(rbackground_size, rnum_train_ex, rbatch_size, rlearning_rate): 
        i_bg = background_sizes.index(bg)
        i_nte = num_train_exs.index(nte)
        i_bs = batch_sizes.index(bs)
        i_lr = learning_rates.index(lr)

        ID = i_lr + (i_bs * LR) + (i_nte * LR * BS) + (i_bg * LR * BS * NTE) + data_offset
        ID_sub = i_lr + (i_bs * LR) + (i_nte * LR * BS) + data_offset
        BG_add = i_bg * LR * BS * NTE

        IDs.append(ID) 
        ID_subs.append(ID_sub)
        BG_adds.append(BG_add)

    return IDs, ID_subs, BG_adds

if __name__ == '__main__':
    print(calculate_IDs([0, 7, 56], [8, 64, 128, 256], [40], learning_rates[:]))
    # print(calculate_IDs(background_sizes[:], [8], [128], [0.1]))
