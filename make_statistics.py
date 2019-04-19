from experiments import calculate_IDs, opt
from experiments import learning_rates as exp_learning_rates

import itertools
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pickle  
# import seaborn as sns


def get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates):
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
    IDs, __ = calculate_IDs(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)
    for ID in IDs:
        iopt = opt[ID]
        bg = bg_lookup[iopt.hyper.background_size]
        nte = nte_lookup[iopt.hyper.num_train_ex]
        lr = lr_lookup[iopt.hyper.learning_rate]
        try: 
            with open(iopt.log_dir_base + iopt.name + '/results.json', 'r') as modelf:
                results = json.load(modelf)
            train_accuracies[bg][nte][lr] = results['train_acc']
            val_accuracies[bg][nte][lr] = results['val_acc'] 
            test_accuracies[bg][nte][lr] = results['test_acc'] 
        except FileNotFoundError:	
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
    with open(opt[0].log_dir_base + 'optimal_model.pickle', 'rb') as ofile:
        optimal_models = pickle.load(ofile)
    for fs in range(FS):
        for bg in range(BG):
            for nte in range(NTE):
                model_key = (full_sizes[fs], background_sizes[bg], num_train_exs[nte]) 
                optimal_params = (batch_sizes[0], learning_rates[choose_nets[bg][nte]])		# This assumes we only get one batch size, like the rest of the code does.
                optimal_models[model_key] = optimal_params
    with open(opt[0].log_dir_base + 'optimal_model.pickle', 'wb') as ofile: 
        pickle.dump(optimal_models, ofile)

    # Return maximal accuracy matrix for graphing
    return final_test_acc	 


def accuracy_v_num_train_ex(suffix, full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates):
    '''
    Plot accuracy vs. num_train_ex, one curve per background_size. 
    NOTES
        Currently assumes one batch_size value. 
        Will always assume one full_size value (I think) because the two values denote two separate experiments. 
    '''
    
    # Setup
    BG = len(background_sizes)
    NTE = len(num_train_exs)
    LR = len(learning_rates)
    bg_lookup = {background_sizes[i]: i for i in range(BG)}
    nte_lookup = {num_train_exs[i]: i for i in range(NTE)}
    lr_lookup = {learning_rates[i]: i for i in range(LR)}

    # Get various accuracies in matrices
    train_accuracies = np.zeros((BG, NTE, LR))
    val_accuracies = np.zeros((BG, NTE, LR))
    test_accuracies = np.zeros((BG, NTE, LR))
    IDs, __ = calculate_IDs(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)
    for ID in IDs:
        iopt = opt[ID]
        bg = bg_lookup[iopt.hyper.background_size]
        nte = nte_lookup[iopt.hyper.num_train_ex]
        lr = lr_lookup[iopt.hyper.learning_rate]
        try:
            with open(iopt.log_dir_base + iopt.name + '/results.json', 'r') as modelf:
                results = json.load(modelf)
            train_accuracies[bg][nte][lr] = results['train_acc'] 
            val_accuracies[bg][nte][lr] = results['val_acc'] 
            test_accuracies[bg][nte][lr] = results['test_acc'] 
        except FileNotFoundError:	# TODO deal with more elegantly? The only reason for lack of results file [that I am currently observing] is divergence during training. For now, I would say this just means those hyperparameters didn't work. It's usually the highest or second-highest learning rate. 
            print('File missing:', ID)
            print('FS:', iopt.hyper.full_size, '; BG:', iopt.hyper.background_size, '; NTE:', iopt.hyper.num_train_ex, '; LR:', iopt.hyper.learning_rate)
            train_accuracies[bg][nte][lr] = -1.
            val_accuracies[bg][nte][lr] = -1.
            test_accuracies[bg][nte][lr] = -1.
    
    # Evaluate based on validation accuracy (get index of best learning_rate for each background_size, and num_train_ex), report test accuracy
    choose_nets = np.argmax(val_accuracies, axis=-1)	# TODO make sure axis is correct when adding batch_size
    final_test_acc = np.choose(choose_nets, np.rollaxis(test_accuracies, 2, 0))
    print(final_test_acc)
#     return final_test_acc
 
    # Graph!
#     cc = itertools.cycle(sns.cubehelix_palette(8))

    final_test_acc = get_optimal_model(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for bg in range(BG):
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
    # res = accuracy_v_num_train_ex('small', [False], [0, 3, 7, 14, 28, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # res = accuracy_v_num_train_ex('full', [True], [0, 3, 7, 14, 28, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # res = accuracy_v_num_train_ex([True], ['random'], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # accuracy_v_num_train_ex('fullsize_fixedrandombg', [True], [0, 3, 7, 14, 28, 56, 'random'], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # accuracy_v_num_train_ex('all', [True], [0, 3, 7, 14, 28, 56, 'random', 'inverted_pyramid', 'random_small'], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # accuracy_v_num_train_ex('baselines', [False], [0, 'random', 'inverted_pyramid', 'random_small'], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # accuracy_v_num_train_ex('debug', [False], [0], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    get_optimal_model([True], [7, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])




