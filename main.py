import os.path
import shutil
import sys
print('PAth:', sys.argv[0])
import numpy as np
import tensorflow as tf
import experiments
from datasets import mnist_dataset 
from nets import nets
from util import summary


################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])
opt = experiments.opt[ID]

print('ID:', ID)
print('Num training examples:', opt.hyper.num_train_ex)
print('Background size:', opt.hyper.background_size)

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print(opt.name)
################################################################################################


################################################################################################
# Define training and validation datasets through Dataset API
################################################################################################

# Initialize dataset and creates TF records if they do not exist
dataset = mnist_dataset.MNIST(opt)

# Repeatable datasets for training
train_dataset = dataset.create_dataset(augmentation=opt.hyper.augmentation, standarization=False, set_name='train', repeat=True)
val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)

# No repeatable dataset for testing
train_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=False)
val_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=False)
test_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=False)

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
ims = tf.unstack(images_in, num=opt.hyper.batch_size, axis=0)

skip_background = False # TODO toggle False after the mismatched input sizes issue 
standardization = False	# TODO toggle based on what's best 
if not skip_background:
    process_ims = []
    # Prepare images by adding correct-sized random background to each 
    for im in ims:	# Get each individual image 
        # imc = im * 255 / tf.reduce_max(im)
        l = r = tf.random_uniform([opt.hyper.image_size, opt.hyper.background_size, 1], maxval=255)
        imc = tf.concat([l, im, r], 1)
        t = b = tf.random_uniform([opt.hyper.background_size, opt.hyper.image_size + 2 * opt.hyper.background_size, 1], maxval=255)
        imc = tf.concat([t, imc, b], 0)
        if standardization:
            imc = tf.image.per_image_standardization(imc)
        process_ims.append(imc)
else:
    process_ims = [tf.image.per_image_standardization(im) if standardization else im for im in ims]

tf.summary.scalar('im max', tf.reduce_max(imc))
tf.summary.scalar('im min', tf.reduce_min(imc))
image = tf.stack(process_ims)

# Call DNN
dropout_rate = tf.placeholder(tf.float32)
to_call = getattr(nets, opt.dnn.name)
print('TO CALL:', to_call)
y, parameters, _ = to_call(image, dropout_rate, opt, dataset.list_labels)

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

tf.summary.image('input', tf.expand_dims(
            tf.reshape(tf.cast(image, tf.float32), [-1, opt.hyper.image_size + 2 * opt.hyper.background_size, opt.hyper.image_size + 2 * opt.hyper.background_size]), 3))
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
        sess.run(tf.global_variables_initializer())
    elif opt.restart:
        print("RESTART")
        shutil.rmtree(opt.log_dir_base + opt.name + '/models/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/train/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/val/')
        sess.run(tf.global_variables_initializer())
    else:
        print("RESTORE")
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
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
        for iEpoch in range(int(sess.run(global_step)), opt.hyper.max_num_epochs):
            print(iEpoch)
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

#     if flag_testable:
# 
#         test_handle_full = sess.run(test_iterator_full.string_handle())
#         validation_handle_full = sess.run(val_iterator_full.string_handle())
#         train_handle_full = sess.run(train_iterator_full.string_handle())
# 
#         # Run one pass over a batch of the train dataset.
#         sess.run(train_iterator_full.initializer)
#         acc_tmp = 0.0
#         for num_iter in range(15):
#             print('TRAIN HANDLE TYPE:', type(train_handle_full))
#             acc_val = sess.run([accuracy], feed_dict={handle: train_handle_full, dropout_rate: opt.hyper.drop_test})
#             acc_tmp += acc_val[0]
#  
#         val_acc = acc_tmp / float(15)
#         print("Full train acc = " + str(val_acc))
#         sys.stdout.flush()
# 
# 
#         # Run one pass over a batch of the validation dataset.
#         sess.run(val_iterator_full.initializer)
#         acc_tmp = 0.0
#         for num_iter in range(15):
#             acc_val = sess.run([accuracy], feed_dict={handle: validation_handle_full,
#                                                       dropout_rate: opt.hyper.drop_test})
#             acc_tmp += acc_val[0]
# 
#         val_acc = acc_tmp / float(15)
#         print("Full val acc = " + str(val_acc))
#         sys.stdout.flush()
# 
# 
#         # Run one pass over a batch of the test dataset.
#         sess.run(test_iterator_full.initializer)
#         acc_tmp = 0.0
#         for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)):
#             acc_val = sess.run([accuracy], feed_dict={handle: test_handle_full,
#                                                       dropout_rate: opt.hyper.drop_test})
#             acc_tmp += acc_val[0]
# 
#         val_acc = acc_tmp / float(int(dataset.num_images_test / opt.hyper.batch_size))
#         print("Full test acc: " + str(val_acc))
#         sys.stdout.flush()
# 
#         print(":)")
# 
#     else:
#         print("MODEL WAS NOT TRAINED")
# 
