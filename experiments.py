import numpy as np


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
        self.batch_size = 1 
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
        self.log_dir_base = '/om/user/sanjanas/eccentricity-data/models/' 
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

# neuron_multiplier = [0.25, 0.5, 1, 2, 4]
# crop_sizes = [28, 24, 20, 16, 12]
# training_data = [1]

name = ["Alexnet"]
num_layers = [5]
max_epochs = [100]
background_sizes = [int(28 * 2**i) for i in range(-3, 2)]	# total of 6 
num_train_exs = [2**i for i in range(3, 11)]	# total of 8 experiments


# idx = 0
# # Create base for TF records:
# opt += [Experiments(idx, "data")]
# opt[-1].hyper.max_num_epochs = 0
# idx += 1


idx = 0
for num_train_ex in num_train_exs:
    for background_size in background_sizes:
         data = Experiments(idx, 'numtrainex' + str(num_train_ex) + '_backgroundsize' + str(background_size))
         data.dataset.set_num_train_ex(num_train_ex)
         data.hyper.set_background_size(background_size)
         opt.append(data)
         opt[-1].hyper.max_num_epochs = 2	# TODO change to max_epochs[0]
         opt[-1].hyper.num_train_ex = num_train_ex
     
         idx += 1


for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):

    i = 0
    for background_size in background_sizes:
        # for data in opt[:len(num_train_exs)]:	# get tf records for each num_train_ex
        i += 1
        for data in [opt[12 + 0]]:	# just the 64-image, smallest-background-size-trained model
            opt += [Experiments(idx, name_NN + '_' + str(background_size) + '_' + str(num_train_ex))]
    
            opt[-1].hyper.max_num_epochs = max_epochs_NN
            opt[-1].hyper.background_size = background_size
            opt[-1].hyper.num_train_ex = data.hyper.num_train_ex 
            opt[-1].dnn.name = name_NN
            opt[-1].dnn.set_num_layers(num_layers_NN)
            opt[-1].dnn.neuron_multiplier.fill(3)
    
            opt[-1].dataset.reuse_tfrecords(data) # opt[0])
            opt[-1].hyper.max_num_epochs = int(max_epochs_NN)
            opt[-1].hyper.num_epochs_per_decay = \
                int(opt[-1].hyper.num_epochs_per_decay)   
     
            idx += 1

