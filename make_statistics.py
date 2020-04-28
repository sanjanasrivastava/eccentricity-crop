from experiments import calculate_IDs, opt
from experiments import learning_rates as exp_learning_rates

import itertools
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle  
from PIL import Image
import tensorflow as tf
# import seaborn as sns


PATH_TO_VIS_DATA = '/om/user/sanjanas/eccentricity-data/visualizations/'


def get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates, 
                      fixed_inv_pyr=False, 
                      fixed_inv_pyr_truncated=False,
                      inv_pyr_test_fixed=False, 
                      random_test_fixed=False,
                      invpyr_perfectcrops=False):
    '''
    Get the ideal network for a certain full_size/background_size/num_train_ex combo, save in a json.
      Each entry: (<full_size>, <background_size>, <num_train_ex>) : (batch_size, learning_rate)
    Return matrix of maximal accuracies for these trios for graphing.
    '''

    # Setup 
    FS = len(full_sizes)
    BG = len(background_sizes)
    NTE = len(num_train_exs)
    LR = len(learning_rates)
    bg_lookup = {background_sizes[i]: i for i in range(BG)}
    nte_lookup = {num_train_exs[i]: i for i in range(NTE)}
    lr_lookup = {learning_rates[i]: i for i in range(LR)}
    
    # Get various accuracies in matrices
    train_accuracies, val_accuracies, test_accuracies = (np.zeros((BG, NTE, LR)) for __ in range(3))
    if fixed_inv_pyr:
        IDs = list(range(9393,9716))           # num_train_ex 1-256, batch_size=8 instead of some being 8 and some being 40
#        IDs = list(range(9393, 9501))           # num_train_ex 1, 2, 4 come before larger num_train_ex
#         for mul in range(6):
#             IDs.extend([i + (mul * 60) + 8790 for i in range(0, 36)])
    elif fixed_inv_pyr_truncated:
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 54) + 9393 for i in range(18, 54)])
    elif inv_pyr_test_fixed:
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 7) + 8720 for i in range(6)])
    elif random_test_fixed:
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 7) + 8650 for i in range(6)])
    elif invpyr_perfectcrops:
        IDs = range(9210, 9390)
        IDs = []
        for mul in range(3):
            IDs.extend([i + (mul * 60) + 9210 for i in range(36)])
    else:
        IDs, __ = calculate_IDs(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)
    for ID in IDs:
        iopt = opt[ID]
        bg = bg_lookup[iopt.hyper.background_size]
        nte = nte_lookup[iopt.hyper.num_train_ex]
        lr = lr_lookup[iopt.hyper.learning_rate]
        try: 
            with open(iopt.log_dir_base + iopt.name + '/' + iopt.results_file, 'r') as modelf:
                results = json.load(modelf)
            train_accuracies[bg][nte][lr] = results['train_acc']
            val_accuracies[bg][nte][lr] = results['val_acc'] 
            test_accuracies[bg][nte][lr] = results['test_acc'] 
        except IOError:	
            print('File missing:', ID)
            print('FS:', iopt.hyper.full_size, '; BG:', iopt.hyper.background_size, '; NTE:', iopt.hyper.num_train_ex, '; LR:', iopt.hyper.learning_rate)
            train_accuracies[bg][nte][lr] = -1.
            val_accuracies[bg][nte][lr] = -1.
            test_accuracies[bg][nte][lr] = -1.

    # Evaluate based on validation accuracy (get index of best learning_rate for each background_size, and num_train_ex), report test accuracy
    choose_nets = np.argmax(val_accuracies, axis=-1)	# TODO make sure axis is correct when adding batch_size
    final_test_acc = np.choose(choose_nets, np.rollaxis(test_accuracies, 2, 0))
    print(choose_nets)

    # Make mapping dictionary
    with open(opt[0].log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
        optimal_models = pickle.load(ofile)
    for fs in range(FS):
        for bg in range(BG):
            for nte in range(NTE):
                model_key = (full_sizes[fs], background_sizes[bg], num_train_exs[nte]) 
                optimal_params = (batch_sizes[0], learning_rates[choose_nets[bg][nte]])		# This assumes we only get one batch size, like the rest of the code does.
                optimal_models[model_key] = optimal_params
    with open(opt[0].log_dir_base + 'optimal_models.pickle', 'wb') as ofile: 
        pickle.dump(optimal_models, ofile)

    # Return maximal accuracy matrix for graphing
    return final_test_acc	 


def background_use_statistics(full_size, background_size, num_train_ex):
    '''
    Plot histogram of feature maps across batch images grouped by variance in background 
    region pixels for given parameters. 
    One plot saved for each conv layer. 
    NOTE: not to be used with background_size == 0.
    '''

    # Select optimal model for each (num_train_ex, background_size) and retrieve npz of activations
    with open(opt[0].log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
        optimal_models = pickle.load(ofile)
    try:
        obatch_size, olearning_rate = optimal_models[(full_size, background_size, num_train_ex)]
    except KeyError:
        print('Ideal learning rate has not been established, cannot plot histogram for these parameters.')
        return
    ID, __ = calculate_IDs([full_size], [background_size], [num_train_ex], [obatch_size], [olearning_rate])
    iopt = opt[ID[0]]
    activations = np.load(iopt.log_dir_base + iopt.name + '/train_activations.npz')
   
    dirname = PATH_TO_VIS_DATA + os.path.join('fullsize_' + str(full_size),
                                              'backgroundsize_' + str(background_size), 
                                              'numtrainex_' + str(num_train_ex))
    if not os.path.exists(dirname):
        os.makedirs(dirname) 

    # Go through each layer in the activations
    for layer_name in activations:
        if 'conv' not in layer_name:		# NOTE this was not extended to fully-connected layers 
            continue			

        # Get size parameters to cut matrix	TODO check if integer division has caused incorrect measures
        initial_image_size = (2 * background_size) + iopt.hyper.image_size
        if full_size:	# This means the actual background size is scaled up 
            object_size = int((140 / initial_image_size) * iopt.hyper.image_size)		# Scale up object size accordingly
            total_image_size = 140
            bg = (total_image_size - object_size) // 2
        else:
            object_size = iopt.hyper.image_size
            total_image_size = initial_image_size
            bg = background_size        

        if layer_name == 'conv2':	# If conv2, all metrics are divided by 2
            bg, object_size = bg // 2, object_size // 2

        # Cut object out of layer and reform to remove object. Location and order don't 
        # matter for sttdev.
        layer = activations[layer_name]
        t = layer[:, :bg, :bg + object_size, :]
        r = layer[:, :bg + object_size, bg + object_size:, :]
        b = layer[:, bg + object_size:, bg:, :]
        l = layer[:, bg:, :bg, :]
        r, l = np.swapaxes(r, 1, 2), np.swapaxes(l, 1, 2)
        B = np.concatenate([t, r, b, l], axis=1)

        # Calculate stddev across images, height, and width, maintaining separation between
        # individual feature maps
        sigmas = np.std(B, axis=(0, 1, 2))
        fig = plt.figure()
        try:
            plt.hist(sigmas, bins=20)
        except AttributeError:
            return
        plt.savefig(os.path.join(dirname, '_'.join([layer_name, 'featuremap', 'stddevs']) + '.pdf'))



def visualize_separate_ip_activations():
    '''
    Get activations and filters for random-trained inverted pyramid tested on fixed background. 
    Convolve each crop of the inverted pyramid with its corresponding kernel channel.
    Visualize the results.

    Only conv1 is really relevant because conv2 is combining feature maps.
    '''

    for ID in range(8720, 8762):
        iopt = opt[ID]
        activations = np.load(iopt.log_dir_base + iopt.name + (('/train_activations_bg' + str(iopt.hyper.background_size) + '.npz')))
        sample_idx = np.array([15, 30])
        featuremap_idx = np.arange(0, 32, 2)
        dirname = PATH_TO_VIS_DATA + os.path.join('architecture_splitinvertedpyramid',
                                                  'fullsize_True',
                                                  'backgroundsize_' + str(iopt.hyper.background_size),
                                                  'num_train_ex' + str(iopt.hyper.num_train_ex))


        if not os.path.exists(dirname):
            os.makedirs(dirname)

        inputs = activations['inputs'][sample_idx, :, :, :]
        image1, image2 = inputs
        kernels = activations['kernel1'][:, :, :, featuremap_idx]        # [height, width, channels, maps]
        kernels = np.moveaxis(kernels, -1, 0)   # [maps, height, width, channels] so I can iterate through
        
        # Replace this (PyTorch-style TF mixed with NumPy)...
        if False:
            with tf.Session() as sess:
                i = 0
                for image in inputs:
                    j = 0
                    for kernel in kernels:
                        for channel_idx in range(iopt.dnn.num_input_channels):
                            M = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(image[:, :, channel_idx]), axis=0), axis=-1)
                            K = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(kernel[:, :, channel_idx]), axis=-1), axis=-1)
                            M_ = tf.nn.conv2d(M, K, [1, 1, 1, 1], padding='SAME')
                            conv_mat = np.squeeze(M_.eval(session=sess))
    
                            conv_mat -= np.min(conv_mat)
                            conv_mat *= 255. / np.max(conv_mat)
                            conv_mat = conv_mat.astype(np.uint8)
                            conv_image = Image.fromarray(conv_mat, mode='L')
    
                            conv_image.save(os.path.join(dirname, '_'.join(['im' + str(sample_idx[i]), 'map' + str(featuremap_idx[j]), 'crop' + str(channel_idx)])), 'JPEG')
    
                        j += 1
                    i += 1

        # ...with proper TF graph-building and executing.
        images = tf.placeholder(tf.float32, shape=(2, 28, 28, 5))         # tensor shape (2, 28, 28, 5)
        ipcrops = tf.unstack(images, axis=-1)                             # list len 5: tensors shape (2, 28, 28)
        ipcrops = [tf.expand_dims(ipcrop, axis=-1) for ipcrop in ipcrops] # list len 5: tensors shape (2, 28, 28, 1)
        kernels = tf.convert_to_tensor(kernels)                           # tensor shape (28, 28, 5, 16)
        kernels = tf.unstack(kernels, axis=2)                             # list len 5: tensors shape (28, 28, 16)
        kernels = [tf.expand_dims(kernel, axis=2) for kernel in kernels]  # list len 5: tensors shape (28, 28, 1, 16)

        conv_outputs = [tf.nn.conv2d(ipcrop, kernel) for ipcrop, kernel in zip(ipcrops, kernels)]   # list len 5: tensors shape (2, 28, 28, 16)
        conv_outputs = tf.stack(conv_outputs, axis=-2)                    # tensor shape (2, 28, 28, 5, 16)
        # TODO do I need to zero-adjust, or is it already nonnegative due to ReLU? Historically it has been nonnegative. If not nonnegative, am I misleading myself and others about negative (and therefore more meaninful) values by zero-adjusting? In that case I suspect it would be more gray than black, but who knows.
        norm_factors = tf.reduce_max(conv_outputs, axis=(1, 2, 3)) * 255.                           # tensor shape (2, 16)

        

        print(conv_outputs)
        crash




        


def visualize_mnist_activations(architecture, full_size, background_size, num_train_ex, test_bg=None, override_id=None):
    '''
    Save JPEG visualizations of activations from the optimal model parametrized here.
    Does one at a time because I don't expect to be using this heavily, as it's for vis.

    If visualizing an architecture tested under a changed condition from training, 
    test_bg holds the appropriate background_size. Assuming these are 
    full_size because otherwise how the fuck would you compare?????????????

    If using an ID that can't be accessed from experiments.calculate_IDs, put it in override_id.
 
    NOTE: Currently assumes 'mnist_cnn' architecture.
    '''

    with open(opt[0].log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
        optimal_models = pickle.load(ofile)
    try:
        obatch_size, olearning_rate = optimal_models[(full_size, background_size, num_train_ex)]
        print('OPTIMAL LEARNING RATE:', olearning_rate)
    except KeyError:
        print('Ideal learning rate has not been established, cannot visualize for these parameters.')
        return
    if override_id is not None:
        ID = override_id
    else:
        ID, __ = calculate_IDs([full_size], [background_size], [num_train_ex], [obatch_size], [olearning_rate])
        ID = ID[0]
    print('OPTIMAL ID:', ID)
    iopt = opt[ID]
    activations = np.load(iopt.log_dir_base + iopt.name + (('/train_activations_bg' + str(test_bg) + '.npz'))) 
    print(iopt.log_dir_base + iopt.name)

    sample_idx = np.array([2, 6]) if 9393 <= ID <= 9716 else np.array([15, 30])         # tinydata uses batch size = 8
    dirname = PATH_TO_VIS_DATA + os.path.join('architecture_' + str(architecture),
                                              'fullsize_' + str(full_size),
                                              'backgroundsize_' + str(background_size), 
                                              'numtrainex_' + str(num_train_ex))
    if not os.path.exists(dirname):
        os.makedirs(dirname) 
   
    # NOTE no longer saving inputs; currently not saving FC layers 
    for layer_name in activations:
        # TODO currently saving first kernel, could change
        layer = activations[layer_name]
        if layer_name == 'inputs':			# No inputs, but need this catch since they're in npz
            continue					
        elif layer.ndim == 4: 				# if conv layer, work with three non-batch dimensions
            feature_map_idx = np.arange(0, 32, 2) if layer_name == 'conv1' else np.arange(0, 64, 4)
            samples = layer[sample_idx, :, :, :]
            samples = samples[:, :, :, feature_map_idx]
        else:						# ow it's second fc layer, which won't be saved NOTE also not saving first fc layer, above lines would have saved it
            continue

        i = 0
        for sample in samples:
            if sample.ndim == 3:
                j = 0
                for m in sample.swapaxes(2, 0).swapaxes(1, 2):
                    m -= np.min(m)
                    m *= 255. / np.max(m)
                    m = m.astype(np.uint8)
                    im = Image.fromarray(m, mode='L')
                    im.save(os.path.join(dirname, '_'.join([layer_name, 'im' + str(sample_idx[i]), 'map' + str(feature_map_idx[j]), 'testbg' + str(test_bg)])), 'JPEG')
                    j += 1
            else:
                continue							# Not saving FC
                im = Image.fromarray(sample.reshape(32, 32), mode='L')		# TODO remove hardcoding

            i += 1


def accuracy_v_num_train_ex(suffix, full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates,
                            plot_title=None,
                            fixed_inv_pyr=False, 
                            fixed_inv_pyr_truncated=False,
                            inv_pyr_test_fixed=False, 
                            random_test_fixed=False,
                            invpyr_perfectcrops=False):
    '''
    Plot accuracy vs. num_train_ex, one curve per background_size. 
    NOTES
        Currently assumes one batch_size value. 
        Will always assume one full_size value (I think) because the two values denote two separate experiments. 
    '''
    
    # TODO change to get from file
    final_test_acc = get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates, fixed_inv_pyr=fixed_inv_pyr, fixed_inv_pyr_truncated=fixed_inv_pyr_truncated, inv_pyr_test_fixed=inv_pyr_test_fixed, random_test_fixed=random_test_fixed, invpyr_perfectcrops=invpyr_perfectcrops)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for bg in range(len(background_sizes)):
#     for results_per_bg in final_test_acc:
        results_per_bg = final_test_acc[bg]
        plt.errorbar(x=num_train_exs, y=100*results_per_bg, fmt='-o' if type(background_sizes[bg]) != int else '-o', label=background_sizes[bg] if background_sizes[bg] != 'random' else 'vanilla')
        plt.xscale('log')
        plt.xlim(8, 256)
        plt.ylim(0, 100)

    # plt.xticks(num_train_exs, [str(num_train_ex) for num_train_ex in num_train_exs])
    ax.set_xticks(num_train_exs)
    ax.set_xticklabels([str(num_train_ex) for num_train_ex in num_train_exs], visible=True)
    print(plt.xticks())
    print([str(label) for label in ax.get_xticklabels()])

    ax.legend(loc='lower right', frameon='True', title='Background size')
    ax.set_xlabel('Number of training examples (T)')
    ax.set_ylabel('% Accuracy on test set')

    # Hardcoded experiments 
    if fixed_inv_pyr:
        ax.set_title('Inverted pyramid trained and tested on fixed-background size inputs') 
    elif fixed_inv_pyr_truncated:
        ax.set_title('Inverted pyramid trained and tested on fixed-background size inputs (B >= 8)') 
    elif invpyr_perfectcrops:
        ax.set_title('Inverted pyramid trained and tested on fixed-background size inputs \n(one pyramid layer is always exactly the original object)')
    elif inv_pyr_test_fixed:
        ax.set_title('Inverted pyramid trained on random-background size inputs, \ntested on fixed-background size inputs')
    elif random_test_fixed:
        ax.set_title('Vanilla trained on random-background size inputs, \ntested on fixed-background size inputs')
    else:
        ax.set_title(plot_title)

    
    plt.savefig('./' + suffix + '.pdf')
    plt.close()



if __name__ == '__main__':

    visualize_separate_ip_activations()

    if False:
        for background_size in [7, 56]:
            for num_train_ex in [8, 256]:
                visualize_mnist_activations('vanilla', True, background_size, num_train_ex)

    # Running visualize_mnist_activations for IDs that cannot be sourced from calculate_IDs
    if False:
         override_ids = list(range(9393, 9717))
         i = 0
         for background_size in [0, 3, 7, 14, 28, 56]:
             for num_train_ex in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                 for learning_rate in exp_learning_rates:
     
                     # Limit which activations are pulled
                     if background_size not in [7, 56] or num_train_ex not in [8, 256]:
                         i += 1
                         continue
                     
                     # Get optimal model and parameters
                     with open(opt[0].log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
                         optimal_models = pickle.load(ofile)
                     try:
                         obatch_size, olearning_rate = optimal_models[(True, background_size, num_train_ex)]
                         if olearning_rate != learning_rate:
                             i += 1
                             continue
                     except KeyError:
                         print('Ideal learning rate has not been established, cannot visualize for these parameters.')
                         i += 1
                         continue
     
                     # Make and save visualization
                     visualize_mnist_activations('invertedpyramid', True, background_size, num_train_ex, override_id=override_ids[i])
                     i += 1


