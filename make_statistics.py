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
# import seaborn as sns


PATH_TO_VIS_DATA = '/om/user/sanjanas/eccentricity-data/visualizations/'


def get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates, fixed_inv_pyr=False, inv_pyr_test_fixed=False, random_test_fixed=False):
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
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 60) + 8790 for i in range(0, 36)])
    elif inv_pyr_test_fixed:
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 7) + 8720 for i in range(6)])
    elif random_test_fixed:
        IDs = []
        for mul in range(6):
            IDs.extend([i + (mul * 7) + 8650 for i in range(6)])
    else:
        IDs, __ = calculate_IDs(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)
    for ID in IDs:
        print('ID:', ID)
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

    for layer_name in activations:
        if 'conv' not in layer_name:		# TODO deal with FC layers
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
            print 'full size:', full_size, '; background size:', background_size, '; num train ex:', num_train_ex
            print sigmas
            return
        plt.savefig(os.path.join(dirname, '_'.join([layer_name, 'featuremap', 'stddevs']) + '.pdf'))



def visualize_mnist_activations(full_size, background_size, num_train_ex, test_bg=None):
    '''
    Save JPEG visualizations of activations from the optimal model parametrized here.
    Does one at a time because I don't expect to be using this heavily, as it's for vis.

    If visualizing an architecture tested under a changed condition from training, 
    test_bg holds the appropriate background_size. Assuming these are 
    full_size because otherwise how the fuck would you compare?????????????
 
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
    ID, __ = calculate_IDs([full_size], [background_size], [num_train_ex], [obatch_size], [olearning_rate])
    print('OPTIMAL ID:', ID)
    iopt = opt[ID[0]]
    activations = np.load(iopt.log_dir_base + iopt.name + (('/train_changed_condition_activations_bg' + str(test_bg) + '.npz') if test_bg is not None  else '/train_activations.npz'))
    print(iopt.log_dir_base + iopt.name)

    sample_idx = np.array([15, 30])
    dirname = PATH_TO_VIS_DATA + os.path.join('fullsize_' + str(full_size),
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
                    im.save(os.path.join(dirname, '_'.join([layer_name, 'im' + str(sample_idx[i]), 'map' + str(feature_map_idx[j]), (('testbg' + str(test_bg)) if test_bg is not None else '')])), 'JPEG')
                    j += 1
            else:
                continue							# Not saving FC
                im = Image.fromarray(sample.reshape(32, 32), mode='L')		# TODO remove hardcoding

            i += 1


def accuracy_v_num_train_ex(suffix, full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates, fixed_inv_pyr=False, inv_pyr_test_fixed=False, random_test_fixed=False):
    '''
    Plot accuracy vs. num_train_ex, one curve per background_size. 
    NOTES
        Currently assumes one batch_size value. 
        Will always assume one full_size value (I think) because the two values denote two separate experiments. 
    '''
    
    # TODO change to get from file
    final_test_acc = get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates, fixed_inv_pyr=fixed_inv_pyr, inv_pyr_test_fixed=inv_pyr_test_fixed, random_test_fixed=random_test_fixed)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for bg in range(len(background_sizes)):
#     for results_per_bg in final_test_acc:
        results_per_bg = final_test_acc[bg]
        plt.errorbar(x=num_train_exs, y=100*results_per_bg, fmt='--o' if type(background_sizes[bg]) != int else '-o', label=background_sizes[bg])
        plt.xscale('log')
        plt.xlim(8, 256)

    # plt.xticks(num_train_exs, [str(num_train_ex) for num_train_ex in num_train_exs])
    ax.set_xticks(num_train_exs)
    ax.set_xticklabels([str(num_train_ex) for num_train_ex in num_train_exs], visible=True)
    print(plt.xticks())
    print([str(label) for label in ax.get_xticklabels()])

    ax.legend(loc='lower right', frameon='True', title='Input type')
    
    plt.savefig('./test' + '_' + suffix + '.pdf')
    plt.close()



if __name__ == '__main__':
    # get_optimal_model([True], ['random', 'inverted_pyramid'], [16, 32, 64, 128], [40], exp_learning_rates[:])
    accuracy_v_num_train_ex('random_fixed_tests', [True], [0, 3, 7, 14, 28, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:], random_test_fixed=True)
    accuracy_v_num_train_ex('inverted_pyramid_fixed_tests', [True], [0, 3, 7, 14, 28, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:], inv_pyr_test_fixed=True)
    
#     for train_bg in ['inverted_pyramid']:
#         for num_train_ex in [8, 256]:
#             for test_bg in [7, 56]:
#                 visualize_mnist_activations(True, train_bg, num_train_ex, test_bg=test_bg)
    # visualize_mnist_activations(True, 'inverted_pyramid', 256)
    


