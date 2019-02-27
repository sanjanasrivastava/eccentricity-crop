from experiments import calculate_IDs, opt
from experiments import learning_rates as exp_learning_rates
print('imported opt')
import itertools
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

PATH_TO_DATA = '/om2/user/sanjanas/eccentricity-data/models/'


def model_folder_name(ID, background_size, num_train_ex):
    return 'ID' + str(ID) + '_mnist_cnn_backgroundsize' + str(background_size) + '_numtrainex' + str(num_train_ex) + '/'


def accuracy_v_num_train_ex(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates):
    '''
    Plot accuracy vs. num_train_ex, one curve per background_size
    NOTE: currently assumes one batch_size value. Will always assume one full_size value (I think) because the two values denote two separate experiments. 
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
    IDs, __, __, __ = calculate_IDs(full_sizes, background_sizes, num_train_exs, batch_sizes, learning_rates)
    for ID in IDs:
        iopt = opt[ID]
        bg = bg_lookup[iopt.hyper.background_size]
        nte = nte_lookup[iopt.hyper.num_train_ex]
        lr = lr_lookup[iopt.hyper.learning_rate]
        try:
            with open(PATH_TO_DATA + model_folder_name(ID, iopt.hyper.background_size, iopt.hyper.num_train_ex) + 'results.json', 'r') as modelf:
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
    first = True
    for results_per_bg in final_test_acc:
        if first:
            plt.errorbar(x=num_train_exs, y=100*results_per_bg, fmt='-o')
            plt.xscale('log')
            plt.xlim(8, 256)
#             first = False
#         else:
#             plt.errorbar(x=num_train_exs, y=100*results_per_bg, fmt='-o', color=next(cc))

    plt.xticks(num_train_exs, [str(num_train_ex) for num_train_ex in num_train_exs])
    plt.savefig('./test_full.pdf')


if __name__ == '__main__':
    res = accuracy_v_num_train_ex([True], [0, 3, 7, 14, 28, 56], [8, 16, 32, 64, 128, 256], [40], exp_learning_rates[:])
    # res = accuracy_v_num_train_ex([True], [0, 3, 7, 14, 28, 56], [8, 
    # print(res)






