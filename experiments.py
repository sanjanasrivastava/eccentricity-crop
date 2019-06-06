import itertools
import numpy as np
import pickle

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
        self.num_input_channels = 1

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
        self.full_size = False
        self.inverted_pyramid = False
        self.train_background_size = None	# Only used for different condition experiments, will be set below

    def set_background_size(self, background_size):
#         self.image_size += background_size
        self.background_size = background_size
        

class Experiments(object):

    def __init__(self, id, name):
        self.name = "base"
        self.log_dir_base = '/om/user/sanjanas/eccentricity-data/models/'
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
        if id < 8650 or id >= 8790:
            self.name = 'ID' + str(self.ID) + "_" + name
        else:
            self.name = name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()

	# Add file for saving results
        self.results_file = 'results.json'

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
# num_layers = [3]	# [5]
num_layers = {'mnist_cnn': 3, 'mnist_fcn': 4}
# max_epochs = [20]	# [100]
max_epochs = {'mnist_cnn': 20, 'mnist_fcn': 40}
learning_rates = [10**i for i in range(-6, 0)]	# 6 values
# batch_sizes = [2**i for i in range(8)] if not LIMIT_TESTS else [128]	# 8 values
batch_sizes = [1, 4, 10, 20, 40, 80, 160, 320] if not LIMIT_TESTS else [40]
initialization = None	# TODO 

# Experiment-specific hyperparameters
background_sizes = [0] + [int(28 * 2**i) for i in range(-3, 3)]	if not LIMIT_TESTS else [0, 7, 56]	# 7 values
num_train_exs = [2**i for i in range(3, 12)] + [5000]	# 10 values, last is full training set
full_sizes = [False, True]	# 2 values


def rangeify(nums):
    tmplst = nums[:]
    start = tmplst[0]

    currentrange=[start, start + 1]

    for item in tmplst[1:]:
        if currentrange[1] == item:
            currentrange[1] += 1
        else:
            yield tuple(currentrange)
            currentrange = [item, item + 1]
    yield tuple(currentrange)


def calculate_IDs(rfull_size, rbackground_size, rnum_train_ex, rbatch_size, rlearning_rate):

    FS = len(full_sizes)
    BG = len(background_sizes)
    NTE = data_offset = len(num_train_exs)
    BS = len(batch_sizes)
    LR = len(learning_rates)
    random_bg_offset = FS * BG * NTE * BS * LR 
    inverted_pyramid_offset = random_small_offset = NTE * BS * LR
    
    IDs, ID_subs = ([] for __ in range(2))

    for fs, bg, nte, bs, lr in itertools.product(rfull_size, rbackground_size, rnum_train_ex, rbatch_size, rlearning_rate): 
        i_nte = num_train_exs.index(nte)
        i_bs = batch_sizes.index(bs)
        i_lr = learning_rates.index(lr)
        ID = i_lr + (i_bs * LR) + (i_nte * LR * BS) + data_offset
        ID_subs.append(ID)

        if type(bg) == int:	# TODO change if another int-labeled background_size gets tacked on for another experiment
            i_fs = int(fs)
            i_bg = background_sizes.index(bg)
            BG_add = i_bg * LR * BS * NTE
            FS_add = i_fs * BG * NTE * BS * LR
            ID += BG_add + FS_add

        else:
            ID += random_bg_offset
            if bg != 'random':
                ID += inverted_pyramid_offset
                if bg != 'inverted_pyramid':
                    ID += random_small_offset     

        IDs.append(ID)

    # print(list(rangeify(IDs)), list(rangeify(ID_subs)))
    return IDs, ID_subs



# MAKE DATA
idx = 0
for num_train_ex in num_train_exs:
    data = Experiments(idx, 'numtrainex' + str(num_train_ex)) 
#     data.dataset.set_num_train_ex(num_train_ex)
    opt.append(data)
    opt[-1].hyper.max_num_epochs = 0
    opt[-1].hyper.num_train_ex = num_train_ex

    idx += 1


i = 0
very_small_datas = []
for num_train_ex in [1, 2, 4]:
    data = Experiments(9390 + i, 'numtrainex' + str(num_train_ex))
    data.hyper.max_num_epochs = 0
    data.hyper.num_train_ex = num_train_ex
    data.dnn.name = name[0]
    very_small_datas.append(data)
    i += 1 


# MAKE EXPERIMENTS
# for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):


for full_size in full_sizes:
    for background_size in background_sizes:
        for data in opt[:len(num_train_exs)]:
            for batch_size in batch_sizes:	
                for learning_rate in learning_rates:
                    name_NN = 'mnist_cnn'

                    opt += [Experiments(idx, name_NN + '_' + str('backgroundsize') + str(background_size) + '_' + 'numtrainex' + str(data.hyper.num_train_ex))]
            
                    opt[-1].hyper.max_num_epochs = max_epochs[name_NN]
                    opt[-1].hyper.background_size = background_size
                    opt[-1].hyper.num_train_ex = data.hyper.num_train_ex 
                    opt[-1].hyper.learning_rate = learning_rate
                    opt[-1].hyper.batch_size = batch_size
                    opt[-1].hyper.full_size = full_size

                    opt[-1].dnn.name = name_NN
                    opt[-1].dnn.set_num_layers(num_layers[name_NN])
                    opt[-1].dnn.neuron_multiplier.fill(3)
            
                    opt[-1].dataset.reuse_tfrecords(data) # opt[0])
                    opt[-1].hyper.max_num_epochs = int(max_epochs[name_NN])
                    opt[-1].hyper.num_epochs_per_decay = \
                        int(opt[-1].hyper.num_epochs_per_decay)   
             
                    idx += 1

# Random background, resized to max_input_size
for data in opt[:len(num_train_exs)]:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, name_NN + '_' + 'random_background_' + 'numtrainex' + str(data.hyper.num_train_ex))
            exp.hyper.background_size = 'random' 
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = batch_size
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            idx += 1

# Inverted pyramid
for data in opt[:len(num_train_exs)]:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, name_NN + '_' + 'inverted_pyramid_' + 'numtrainex' + str(data.hyper.num_train_ex))
            exp.hyper.background_size = 'inverted_pyramid' 
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = batch_size
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)
            exp.dnn.num_input_channels = 5

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            idx += 1

# Random background, resized to image_size 
for data in opt[:len(num_train_exs)]:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, name_NN + '_' + 'random_background_small_' + 'numtrainex' + str(data.hyper.num_train_ex))

            exp.hyper.background_size = 'random_small' 
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = batch_size
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            idx += 1


# Black background at training, variable background at testing
for data in opt[:len(num_train_exs)]:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, name_NN + '_' + 'different_conditions_' + 'numtrainex' + str(data.hyper.num_train_ex))
            exp.hyper.background_size = 'different_conditions'
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = batch_size
            exp.hyper.full_size = False

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)
            if exp.hyper.batch_size == 10 and exp.hyper.num_train_ex in [8, 256]:
                pass
            idx += 1


# Changed conditions ONLY FOR INVERTED PYRAMID AND RANDOM 
# ID=8650-8761
for train_background_size in ['random', 'inverted_pyramid']:
    for train_data in opt[:len(num_train_exs)]:
        for test_background_size in background_sizes:
	       
            # Identify best model given these training params
            with open(exp.log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
                optimal_models = pickle.load(ofile)
            try:
                obatch_size, olearning_rate = optimal_models[(True, train_background_size, train_data.hyper.num_train_ex)]
            except KeyError:
                print('num train ex:', train_data.hyper.num_train_ex, ', train bg:', train_background_size)
                opt.append(None)
                idx += 1
                continue
            optimal_id, __ = calculate_IDs([True], [train_background_size], [train_data.hyper.num_train_ex], [obatch_size], [olearning_rate])
	        
            # Load desired model, save results under different name from same-condition results
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, opt[optimal_id[0]].name)
            exp.results_file = 'results_testbg' + str(test_background_size) + '.json'

            exp.dataset.reuse_tfrecords(train_data)
            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            if train_background_size == 'inverted_pyramid':
                exp.dnn.num_input_channels = 5
            
            # For constructing testing images in main.py
            exp.hyper.full_size = True
            exp.hyper.background_size = test_background_size	
            exp.hyper.train_background_size = train_background_size
            exp.hyper.batch_size = obatch_size
            exp.hyper.learning_rate = olearning_rate
            exp.hyper.num_train_ex = train_data.hyper.num_train_ex

            # To skip training
            exp.test = True
            
            opt.append(exp)
            # if train_background_size == 'inverted_pyramid' and exp.hyper.background_size in [0, 3, 7, 14, 28, 56] and exp.hyper.num_train_ex in [8, 16, 32, 64, 128, 256]:
            if False:
                print(idx - 8650)
            idx += 1




# Inverted pyramid architecture trained on fixed-scale data
# ID=8790-9209
# Use ID=0-35,60-95,120-155,180-215,240-275,300-335 with offset 8790 
for background_size in background_sizes:
    for data in opt[:len(num_train_exs)]:
        for learning_rate in learning_rates:

            # exp = Experiments(idx, name_NN + '_' + 'inverted' + 'numtrainex' + str(data.hyper.num_train_ex))
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, '_'.join([name_NN, 'inverted_pyramid', 'numtrainex' + str(data.hyper.num_train_ex), 'backgroundsize' + str(background_size)]))

            exp.hyper.background_size = background_size     # All data should have the same amount of background...
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = 40
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)
            exp.dnn.num_input_channels = 5

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            # if exp.hyper.background_size == 112:
            if False:
                print(idx)
            idx += 1


# Inverted pyramid architecture trained on fixed-scale data, with resizing BEFORE background addition, with a perfect crop in every inverted pyramid.
# NOTE there are some repeated background_sizes, but because the resizing/background addition order has changed, these are functionally different. 14 background pixels on a 28-by-28 object are very different from 14 background pixels on a 112-by-112 object. 0 and 56 are functionally equivalent to their predecessors, so I am skipping them.
# ID=9210-9389
for background_size in [14, 28, 42]:
    for data in opt[:len(num_train_exs)]:
        for learning_rate in learning_rates:

            # exp = Experiments(idx, name_NN + '_' + 'inverted' + 'numtrainex' + str(data.hyper.num_train_ex))
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, '_'.join([name_NN, 'backgroundafterresize', 'invertedpyramid', 'numtrainex' + str(data.hyper.num_train_ex), 'backgroundsize' + str(background_size)]))

            exp.hyper.background_size = background_size     # All data should have the same amount of background...
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = 40
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)
            exp.dnn.num_input_channels = 5

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            # if exp.hyper.background_size == 112:
            # if exp.hyper.num_train_ex <= 256:
            if False:
                print(idx - 9210)

            idx += 1


# Skip ID=9390-9392 for making data with 1, 2, and 4 training examples
opt.extend(very_small_datas)
idx += 3

# Inverted pyramid or random architecture trained on fixed-scale data (resizing before background addition) on tiny amounts of data plus normal amounts of data, all batch_size=8.
# ID=9393-9716

for background_size in background_sizes[:-1]:       # skip background_size=112
    for data in (opt[9390:9393] + opt[:len(num_train_exs) - 4]):    # skip num_train_ex=512,1024,5000
        for learning_rate in learning_rates:
            
            name_NN = 'mnist_cnn'
            exp = Experiments(idx, '_'.join([name_NN, 'invertedpyramid', 'numtrainex' + str(data.hyper.num_train_ex), 'backgroundsize' + str(background_size)]))

            exp.hyper.background_size = background_size
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = 8
            exp.hyper.full_size = True

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)
            exp.dnn.num_input_channels = 5 

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)

            # exp.restart = True      # temporarily to retrain models with proper background_size calculation

            opt.append(exp)

            if False:
                print(idx - 9393)
            idx += 1


# FCN, vanilla, train fixed test fixed
# IDs=9717-9932 (0-215)
for background_size in background_sizes[:-1]:   # skip background_size=112
    for data in opt[:len(num_train_exs) - 4]:   # skip num_train_ex=512,1024,5000
        for learning_rate in learning_rates:
            
            name_NN = 'mnist_fcn'
            exp = Experiments(idx, '_'.join([name_NN, 'vanilla', 'numtrainex' + str(data.hyper.num_train_ex), 'backgroundsize' + str(background_size)]))
    
            exp.hyper.background_size = background_size
            exp.hyper.num_train_ex = data.hyper.num_train_ex
            exp.hyper.learning_rate = learning_rate
            exp.hyper.batch_size = 8
            exp.hyper.full_size = True	# so that it always gets resized up 

            exp.dnn.name = name_NN
            exp.dnn.set_num_layers(num_layers[name_NN])
            exp.dnn.neuron_multiplier.fill(3)

            exp.dataset.reuse_tfrecords(data)
            exp.hyper.max_num_epochs = int(max_epochs[name_NN])
            exp.dataset.max_num_epochs = int(max_epochs[name_NN])
            exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
            opt.append(exp)

            if False:
                print(idx - 9717)
                
            idx += 1


# FCN, vanilla, train random (test fixed will happen later)
# ID=9933-9968 (0-35)
for data in opt[:len(num_train_exs) - 4]:
    for learning_rate in learning_rates:
        name_NN = 'mnist_fcn'
        exp = Experiments(idx, '_'.join([name_NN, 'vanilla', 'numtrainex' + str(data.hyper.num_train_ex), 'backgroundsize' + 'random']))

        exp.hyper.background_size = 'random'
        exp.hyper.num_train_ex = data.hyper.num_train_ex
        exp.hyper.learning_rate = learning_rate
        exp.hyper.batch_size = 8
        exp.hyper.full_size = True

        exp.dnn.name = name_NN
        exp.dnn.set_num_layers(num_layers[name_NN])
        exp.dnn.neuron_multiplier.fill(3)

        exp.dataset.reuse_tfrecords(data)
        exp.hyper.max_num_epochs = int(max_epochs[name_NN])
        exp.dataset.max_num_epochs = int(max_epochs[name_NN])
        exp.hyper.num_epochs_per_decay = int(exp.hyper.num_epochs_per_decay)
        opt.append(exp)

        if False:
            print(idx)

        idx += 1


if __name__ == '__main__':
# print(calculate_IDs([False, True], [3, 14, 28], [16, 32], [40], learning_rates[:]))
# print(calculate_IDs(background_sizes[:], [8], [128], [0.1]))
# print(calculate_randombg_IDs(num_train_exs[:], [40], learning_rates[:]))
    # calculate_IDs([True], [7, 56], [8, 16, 32, 64, 128, 256], [40], learning_rates[:])
    pass

  
