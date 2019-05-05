import csv
import json
import os.path
import shutil
import sys
print('Path:', sys.argv[0])
print('Python version:', sys.version_info.major, sys.version_info.minor)
import numpy as np
import pickle
import tensorflow as tf
import experiments
from datasets import mnist_dataset 
from nets import nets
from util import summary


################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])
print(len(experiments.opt))
opt = experiments.opt[ID]


train_background_size = test_background_size = None

# Check if making model or getting activations
if len(sys.argv) > 2 and (sys.argv[2] == 'activations' or sys.argv[2] == 'changed_condition_activations'): 
    opt.test = True
    with open(opt.log_dir_base + 'optimal_models.pickle', 'rb') as ofile:
        optimal_models = pickle.load(ofile)
    try:
        if optimal_models[(opt.hyper.full_size, opt.hyper.background_size, opt.hyper.num_train_ex)][1] == opt.hyper.learning_rate:
            print('OPTIMAL LEARNING RATE')
            make_activations = True
        else:
            print('Suboptimal learning rate, exiting script.')
            sys.exit()
    except KeyError:
        print('Ideal learning rate has not been established, exiting script.')
        sys.exit()

    if sys.argv[2] == 'changed_condition_activations':
        train_background_size = opt.hyper.background_size
        test_background_size = 56		# TODO hardcoded; change as needed

else:
    make_activations = False

# Check if comparing in different conditions
if len(sys.argv) > 2 and sys.argv[2] == 'comparison_test':
    opt.test = True

if opt.dnn.num_input_channels == 5:
    train_background_size = 'inverted_pyramid'          # so that fixed-background image will be made pyramidal


# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print(opt.name)
print('ID:', ID)
print('Num training examples:', opt.hyper.num_train_ex)
print('Background size:', opt.hyper.background_size)
print('Log directory:', opt.log_dir_base)
################################################################################################


################################################################################################
# Define training and validation datasets through Dataset API
################################################################################################

# Initialize dataset and creates TF records if they do not exist
dataset = mnist_dataset.MNIST(opt)
print('NUM IMAGES EPOCH:', dataset.num_images_epoch)
# Repeatable datasets for training
train_dataset = dataset.create_dataset(augmentation=opt.hyper.augmentation, standarization=False, set_name='train', repeat=True)
val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)

# No repeatable dataset for testing
train_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
val_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
test_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)

# Hadles to switch datasets
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
# iterator = tf.contrib.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()
val_iterator = val_dataset.make_one_shot_iterator()

train_iterator_full = train_dataset_full.make_initializable_iterator()
val_iterator_full = val_dataset_full.make_initializable_iterator()
test_iterator_full = test_dataset_full.make_initializable_iterator()
################################################################################################


################################################################################################
# Declare DNN
################################################################################################

# Get data from dataset dataset
images_in, y_ = iterator.get_next()
images_in.set_shape([opt.hyper.batch_size, opt.hyper.image_size, opt.hyper.image_size, 1])
ims = tf.unstack(images_in, axis=0)


max_input_size = 140
make_background = True
standardization = True	# TODO toggle based on what's best 
if make_background:

    # If making changed_condition_activations, go to test_background_size to make appropriate test images
    if make_activations and sys.argv[2] == 'changed_condition_activations':
        opt.hyper.background_size = test_background_size
  
    # Transform each individual image 
    process_ims = []
    for im in ims:	
        imc = im
 
        # Either generate a random background_size that will, at most, fill up the image; or put the defined background_size into a constant tensor
        # background_size = tf.random_uniform([1], maxval=(max_input_size-opt.hyper.image_size)//2, dtype=tf.int32) if opt.hyper.background_size in ['random', 'inverted_pyramid', 'random_small'] else tf.constant([opt.hyper.background_size])

        if opt.hyper.background_size in ['random', 'inverted_pyramid', 'random_small']:
            background_size = tf.random_uniform([1], maxval=(max_input_size - opt.hyper.image_size) // 2, dtype=tf.int32)
        elif opt.hyper.background_size == 'different_conditions':
            background_size = tf.constant([14])
        else:
            background_size = tf.constant([opt.hyper.background_size])

        # If background_size is randomly generated, resize input image so that when concatenated with background matrices, it will fill up the max_input_size. This is happening before background addition so background pixels are truly not random and not products of interpolation.
        if opt.hyper.background_size == 'random' or opt.hyper.background_size == 'random_small':
            image_size = max_input_size - (2 * background_size)
            imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), tf.concat([image_size, image_size], axis=0)), axis=0)
        else:
            image_size = tf.constant([opt.hyper.image_size])

        # Make random background matrices and add them to the existing image. If training different_conditions comparison experiment, pad with zeros instead.
        if opt.hyper.background_size == 'different_conditions' and not opt.test:
            imc = tf.pad(imc, tf.constant([[14, 14], [14, 14], [0, 0]]))
        else:
            l = tf.random_uniform(tf.concat([image_size, background_size, tf.constant([1])], axis=0), maxval=255)
            r = tf.random_uniform(tf.concat([image_size, background_size, tf.constant([1])], axis=0), maxval=255)
            imc = tf.concat([l, imc, r], 1)
            t = tf.random_uniform(tf.concat([background_size, (background_size * 2) + image_size, tf.constant([1])], axis=0), maxval=255)
            b = tf.random_uniform(tf.concat([background_size, (background_size * 2) + image_size, tf.constant([1])], axis=0), maxval=255)
            imc = tf.concat([t, imc, b], 0)

        # If random-background image is meant to be original image_size, resize it down.
        if opt.hyper.background_size == 'random_small':
            imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), [opt.hyper.image_size, opt.hyper.image_size]), axis=0)

        # If the background_size is predetermined and the image is supposed to be full_size, resize the whole thing up. In this case, the background pixels ARE supposed to be products of interpolation along with the original MNIST image.
        if type(opt.hyper.background_size) == int and opt.hyper.full_size:
            imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), [max_input_size, max_input_size]), axis=0)

        # If we are taking an inverted pyramid approach, make the inverted pyramid. 
        # NOTE: also used for testing inverted_pyramid-trained models on any input. 
        if opt.hyper.background_size == 'inverted_pyramid' or train_background_size == 'inverted_pyramid':
            boxes = [[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.6, 0.6]]
            imc = tf.image.crop_and_resize(tf.expand_dims(imc, axis=0), boxes, [0 for __ in range(len(boxes))], [opt.hyper.image_size, opt.hyper.image_size], method='bilinear')
            imc = tf.transpose(imc, perm=[3, 1, 2, 0])
            imc = tf.squeeze(imc, axis=0)

        # NOTE: NOT A TRAINING CHANGE. If we are making changed_condition_activations for inverted_pyramid, resize the image down and concat it 5 times.
        if False:
        # if sys.argv[2] == 'changed_condition_activations' and train_background_size == 'inverted_pyramid':
            print('BEFORE RESIZE:', imc.get_shape())
            imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), [opt.hyper.image_size, opt.hyper.image_size]), axis=0)
            print('AFTER RESIZE:', imc.get_shape())
            imc = tf.concat([imc for __ in range(5)], axis=2)

        # Now that all that's over, standardize.
        if standardization:
            imc = tf.image.per_image_standardization(imc)
        process_ims.append(imc)

else:
    process_ims = [tf.image.per_image_standardization(im) if standardization else im for im in ims]

# If making changed_condition_activations, go back to original background_size to ensure correct set_shape code
if make_activations and sys.argv[2] == 'changed_condition_activations':
    opt.hyper.background_size = train_background_size

image = tf.stack(process_ims)
if opt.hyper.background_size == 'inverted_pyramid' or opt.hyper.background_size == 'random_small' or train_background_size == 'inverted_pyramid':
    image.set_shape([opt.hyper.batch_size, opt.hyper.image_size, opt.hyper.image_size, opt.dnn.num_input_channels])
elif opt.hyper.full_size:		# this covers all cases where opt.hyper.background_size == 'random'
    image.set_shape([opt.hyper.batch_size, max_input_size, max_input_size, 1])
elif opt.hyper.background_size == 'different_conditions':
    image.set_shape([opt.hyper.batch_size, opt.hyper.image_size + (14 * 2), opt.hyper.image_size + (14 * 2), 1])
else:
    image.set_shape([opt.hyper.batch_size, opt.hyper.image_size + (opt.hyper.background_size * 2), opt.hyper.image_size + (opt.hyper.background_size * 2), 1])

print('IMAGE:', image)

# Save some images for debugging since tensorboard doesn't work 
save_images = True	# TODO toggle off once unnecessary 
if save_images:
    pass


# Call DNN
dropout_rate = tf.placeholder(tf.float32)
to_call = getattr(nets, opt.dnn.name)
y, parameters, activations, inputs = to_call(image, dropout_rate, opt, dataset.list_labels)

# Loss function
with tf.name_scope('loss'):
    weights_norm = tf.reduce_sum(
        input_tensor=opt.hyper.weight_decay * tf.stack(
            [tf.nn.l2_loss(i) for i in parameters]
        ),
        name='weights_norm')
    tf.summary.scalar('weight_decay', weights_norm)

    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

    total_loss = weights_norm + cross_entropy
    tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
################################################################################################


################################################################################################
# Set up Training
################################################################################################

# Learning rate
num_batches_per_epoch = dataset.num_images_epoch / opt.hyper.batch_size
decay_steps = int(opt.hyper.num_epochs_per_decay)
lr = tf.train.exponential_decay(opt.hyper.learning_rate,
                                global_step,
                                decay_steps,
                                opt.hyper.learning_rate_factor_per_decay,
                                staircase=True)
tf.summary.scalar('learning_rate', lr)
tf.summary.scalar('weight_decay', opt.hyper.weight_decay)

# Accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

# tf.summary.image('input', tf.reshape(tf.cast(image, tf.float32), [-1, max_input_size, max_input_size, 1]))
#              tf.reshape(tf.cast(image, tf.float32), [-1, opt.hyper.image_size + 2 * opt.hyper.background_size, opt.hyper.image_size + 2 * opt.hyper.background_size]), 3))
################################################################################################


with tf.Session() as sess:

    ################################################################################################
    # Set up Gradient Descent
    ################################################################################################
    all_var = tf.trainable_variables()

    train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=opt.hyper.momentum).minimize(total_loss, var_list=all_var)
    inc_global_step = tf.assign_add(global_step, 1, name='increment')

    raw_grads = tf.gradients(total_loss, all_var)
    grads = list(zip(raw_grads, tf.trainable_variables()))

    for g, v in grads:
        summary.gradient_summaries(g, v, opt)
    ################################################################################################


    ################################################################################################
    # Set up checkpoints and data
    ################################################################################################

    saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

    # Automatic restore model, or force train from scratch
    flag_testable = False

    # Set up directories and checkpoints
    if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
        print('NO FILE:', opt.log_dir_base + opt.name + '/models/checkpoint')
        sess.run(tf.global_variables_initializer())
    elif opt.restart:
        print("RESTART")
        shutil.rmtree(opt.log_dir_base + opt.name + '/models/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/train/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/val/')
        sess.run(tf.global_variables_initializer())
    else:
        print("RESTORE")
        print('Log directory in restore:', opt.log_dir_base)
        saver.restore(sess, opt.log_dir_base + opt.name + '/models/model-19')
        flag_testable = True

    # datasets
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(val_iterator.string_handle())
    ################################################################################################

    ################################################################################################
    # RUN TRAIN
    ################################################################################################
    if not opt.test:

        # Prepare summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(opt.log_dir_base + opt.name + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(opt.log_dir_base + opt.name + '/val')

        ################################################################################################
        # Loop alternating between training and validation.
        ################################################################################################
        print('NUM EPOCHS:', opt.hyper.max_num_epochs)
        for iEpoch in range(int(sess.run(global_step)), opt.hyper.max_num_epochs):
            # Save metadata every epoch
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summ = sess.run([merged], feed_dict={handle: training_handle, dropout_rate: opt.hyper.drop_train},
                               options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iEpoch)
            saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)

            # Steps for doing one epoch
            for iStep in range(int(dataset.num_images_epoch/opt.hyper.batch_size)):

                # Epoch counter
                k = iStep*opt.hyper.batch_size + dataset.num_images_epoch*iEpoch

                # Print accuray and summaries + train steps
                if iStep == 0:
                    # !train_step
                    print("* epoch: " + str(float(k) / float(dataset.num_images_epoch)))
                    summ, acc_train = sess.run([merged, accuracy],
                                                    feed_dict={handle: training_handle,
                                                               dropout_rate: opt.hyper.drop_train})
                    train_writer.add_summary(summ, k)
                    print("train acc: " + str(acc_train))
                    sys.stdout.flush()

                    summ, acc_val = sess.run([merged, accuracy], feed_dict={handle: validation_handle,
                                                                            dropout_rate: opt.hyper.drop_test})
                    val_writer.add_summary(summ, k)
                    print("val acc: " + str(acc_val))
                    sys.stdout.flush()

                else:

                    sess.run([train_step], feed_dict={handle: training_handle,
                                                      dropout_rate: opt.hyper.drop_train})

            sess.run([inc_global_step])
            print("----------------")
            sys.stdout.flush()
            ################################################################################################

        flag_testable = True

        train_writer.close()
        val_writer.close()

    print('TRAIN COMPLETE')
    ################################################################################################
    # RUN TEST
    ################################################################################################

    if flag_testable:

        test_handle_full = sess.run(test_iterator_full.string_handle())
        validation_handle_full = sess.run(val_iterator_full.string_handle())
        train_handle_full = sess.run(train_iterator_full.string_handle())

        # Run one pass over a batch of the train dataset.
        sess.run(train_iterator_full.initializer)
        acc_tmp = 0.0
        for num_iter in range(1 if make_activations else int(dataset.num_images_epoch/opt.hyper.batch_size)):
            acc_val, train_activations, train_inputs = sess.run([accuracy, activations, inputs], feed_dict={handle: train_handle_full, dropout_rate: opt.hyper.drop_test})
            acc_tmp += acc_val

        conv2 = train_activations[1]
        print('CONV2 MAX:', np.max(conv2))
        print('CONV2 MIN:', np.min(conv2))

        if make_activations:
            np.savez(opt.log_dir_base + opt.name + '/train_' + sys.argv[2] + '_bg' + str(test_background_size),
                     conv1=train_activations[0], 
                     conv2=train_activations[1], 
                     fc1=train_activations[2], 
                     fc2=train_activations[3],
                     inputs=train_inputs[0]) 

        train_acc = acc_tmp / (1. if make_activations else float(dataset.num_images_epoch/opt.hyper.batch_size))
        print("Full train acc = " + str(train_acc))
        sys.stdout.flush()

        # Run one pass over a batch of the validation dataset.
        sess.run(val_iterator_full.initializer)
        acc_tmp = 0.0
        for num_iter in range(int(dataset.num_images_val/opt.hyper.batch_size)):
            acc_val = sess.run([accuracy], feed_dict={handle: validation_handle_full,
                                                      dropout_rate: opt.hyper.drop_test})
            acc_tmp += acc_val[0]

        val_acc = acc_tmp / float(dataset.num_images_val/opt.hyper.batch_size)
        print("Full val acc = " + str(val_acc))
        sys.stdout.flush()


        # Run one pass over a batch of the test dataset.
        sess.run(test_iterator_full.initializer)
        acc_tmp = 0.0
        for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size) + 1):
            acc_val = sess.run([accuracy], feed_dict={handle: test_handle_full,
                                                      dropout_rate: opt.hyper.drop_test})
            acc_tmp += acc_val[0]

        test_acc = acc_tmp / float(int(dataset.num_images_test / opt.hyper.batch_size) + 1)
        print("Full test acc: " + str(test_acc))

        # Record data	TODO uncomment after figuring out synchronization
        with open(opt.log_dir_base + opt.name + '/' + opt.results_file, 'w') as rf:
            results = {'background_size': opt.hyper.background_size,
                       'num_train_ex': opt.hyper.num_train_ex,
                       'batch_size': opt.hyper.batch_size,
                       'learning_rate': opt.hyper.learning_rate,
                       'train_acc': train_acc,
                       'val_acc': val_acc,
                       'test_acc': test_acc}
            if 'testbg' in opt.results_file:
                results['test_background_size'] = opt.hyper.background_size
                results['background_size'] = opt.hyper.train_background_size
            json.dump(results, rf)

        print('\n')
        print('BACKGROUND SIZE:', opt.hyper.background_size)
        print('NUM TRAIN EX:', opt.hyper.num_train_ex)
        print('BATCH SIZE:', opt.hyper.batch_size)
        print('LEARNING RATE:', opt.hyper.learning_rate)
       
        sys.stdout.flush()
        print(":)")

    else:
        print("MODEL WAS NOT TRAINED")

