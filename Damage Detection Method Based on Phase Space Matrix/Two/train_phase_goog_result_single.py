# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, optimizers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, \
    TensorBoard
import h5py

# %%
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 sub_dir='/tmp', sub_dir2='/tmp', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, sub_dir, sub_dir2, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, sub_dir, sub_dir2, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        # if TensorFlow version < 2.0, use FileWriter, if > 2.0, use tf.summary.create_file_writer instead.
        # self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            # the following 5 lines is in TensorFlow 1.x, use 'tf.summary.scalar(name, value, step=epoch)' instead in version 2.x
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value.item()
            # summary_value.tag = name
            # self.val_writer.add_summary(summary, epoch)
            tf.summary.scalar(name, value, step=epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


# %%
def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, input_tensor=None,
             classes=10, activation='tanh', filter_size=(3, 1), trans_filter=(2, 2),
             init_kernel=(5, 1), init_stride=(5, 1)):
    '''Instantiate the DenseNet architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'imagenet' (pre-training on ImageNet)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns
            A Keras model instance.
        '''

    if activation not in ['softmax', 'sigmoid', 'tanh']:
        raise ValueError('activation must be one of "softmax" or "sigmoid", "tanh"')
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation,
                           filter_size, trans_filter, init_kernel, init_stride)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='densenet')

    return model

# %%
def __conv_block(ip, nb_filter, filter_size, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)

    # x = layers.BatchNormalization()(ip)
    x = layers.Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = layers.Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = layers.BatchNormalization()(x)
        x = layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)

        x = layers.Activation('relu')(x)

    x = layers.Conv2D(nb_filter, filter_size, kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x

# %%
def __dense_block(x, nb_layers, nb_filter, growth_rate, filter_size, bottleneck=False, dropout_rate=None,
                  weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, filter_size, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = layers.concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

# %%
def __transition_block(ip, nb_filter, trans_filter=(2, 2), compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)

    # x = layers.BatchNormalization()(ip)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    #     x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    # x = AveragePooling2D((5, 1), strides=(5, 1))(x)
    x = layers.AveragePooling2D(trans_filter, strides=trans_filter)(x)

    return x

# %%
def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='tanh', filter_size=(3, 1), trans_filter=(2, 2),
                       init_kernel=(5, 1), init_stride=(5, 1)):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid' or 'tanh'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        # initial_kernel = (3, 3)
        initial_kernel = init_kernel
        initial_strides = init_stride

    x = layers.Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
                      strides=initial_strides, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(
        img_input)

    if subsample_initial_block:
        x = layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)

        # x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, filter_size,
                                     bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, trans_filter=trans_filter, compression=compression,
                               weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, filter_size, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = layers.BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)

    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    if include_top:
        x = layers.Dense(nb_classes, activation=activation)(x)

    return x

# %%
def ResNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
           bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
           include_top=True,
           classes=10, activation='tanh', filter_size=(3, 1), trans_filter=(2, 2),
           init_kernel=(5, 1), init_stride=(5, 1), cnn=False):
    if activation not in ['softmax', 'sigmoid', 'tanh']:
        raise ValueError('activation must be one of "softmax" or "sigmoid", "tanh"')
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    x = __create_res_net(classes, img_input, include_top, depth, nb_dense_block,
                         growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                         dropout_rate, weight_decay, subsample_initial_block, activation,
                         filter_size, trans_filter, init_kernel, init_stride, cnn)
    inputs = img_input
    # Create model.
    if cnn:
        model = models.Model(inputs, x, name='convnets')
    else:
        model = models.Model(inputs, x, name='resnets')

    return model

# %%
def __res_block(ip, nb_filter, filter_size, bottleneck=False, dropout_rate=None, weight_decay=1e-4, cnn=False):
    shortcut = ip
    print('shortcut:{}'.format(shortcut.shape[-1]))

    x = layers.BatchNormalization()(ip)
    x = layers.Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4
        print('inter_channel:{}'.format(inter_channel))
        x = layers.Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(int(shortcut.shape[-1]), filter_size, kernel_initializer='he_normal', padding='same',
                      use_bias=False)(x)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    # print(x)
    if cnn is False:
        x = layers.Add()([shortcut, x])
    return x

# %%
def __residual_block(x, nb_layers, nb_filter, growth_rate, filter_size, bottleneck=False, dropout_rate=None,
                     weight_decay=1e-4,
                     grow_nb_filters=True, cnn=False):
    for i in range(nb_layers):
        x = __res_block(x, growth_rate, filter_size, bottleneck, dropout_rate, weight_decay, cnn)

        if grow_nb_filters:
            nb_filter += growth_rate

    return x, nb_filter

# %%
def __transition_res_block(ip, nb_filter, trans_filter=(2, 2), compression=1.0, weight_decay=1e-4):
    x = layers.BatchNormalization()(ip)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.AveragePooling2D(trans_filter, strides=trans_filter)(x)

    return x

# %%
def __create_res_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                     nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                     subsample_initial_block=False, activation='tanh', filter_size=(3, 1), trans_filter=(2, 2),
                     init_kernel=(5, 1), init_stride=(5, 1), cnn=False):
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    if nb_layers_per_block == -1:
        assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
        count = int((depth - 4) / 3)

        if bottleneck:
            count = count // 2

        nb_layers = [count for _ in range(nb_dense_block)]
        final_nb_layer = count
    else:
        final_nb_layer = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    initial_kernel = init_kernel
    initial_strides = init_stride

    x = layers.Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
                      strides=initial_strides, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(
        img_input)

    if subsample_initial_block:
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __residual_block(x, nb_layers[block_idx], nb_filter, growth_rate, filter_size,
                                        bottleneck=bottleneck,
                                        dropout_rate=dropout_rate, weight_decay=weight_decay, cnn=cnn)
        # add transition_block
        x = __transition_res_block(x, nb_filter, trans_filter=trans_filter, compression=compression,
                                   weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __residual_block(x, final_nb_layer, nb_filter, growth_rate, filter_size, bottleneck=bottleneck,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay, cnn=cnn)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    if include_top:
        x = layers.Dense(nb_classes, activation=activation)(x)

    return x

# %%
class FrameDamageDetection2D_Acceleration():
    def __init__(self):
        # Define the working directory
        # self.dir = '/home/Deep_Learner/work/SHM/Frame-Acc/'
        # self.data_dir = '/home/Deep_Learner/work/SHM/Frame-Acc/'
        self.dir = os.path.abspath('.')
        self.data_dir = os.path.abspath('.')
        self.whitening = False
        self.mapminmax = False
        self.standardize = False
        self.process_y = True # process the output, MinMaxScaler
        self.depth = 40
        self.nb_dense_block = 3
        self.growth_rate = 12
        self.kernel = (2, 5)
        # Params of Frame data set
        self.img_rows = 14
        self.img_cols = 331
        self.channels = 9
        self.classes = 70
        self.bottleneck = True
        self.reduction = 0.0
        self.trans_filter = (2, 2)
        self.init_kernel = (1, 15)
        self.init_stride = (1, 5)
        # Initial hyper-parameters for training a model
        self.n_epoch = 500  # number of training epochs
        self.l2_wd = 0.0005  # l2 weight decay regularization
        self.l1_sr = 0  # l1 sparse regularizer
        self.lr = 0.0001  # learning rate
        self.batch_size = 128  # batch size
        self.activation = None
        self.max_pool = False
        self.y_sc = None
        self.tc = 'tc'  # Test Cases group name
        self.sub_tc = 'sub_tc'  # Single test case in the group
        self.noise_type = 'None'  # 'Clean' 'W10dB' 'W20dB' 'W30dB'  'Unc' 'Pink'
        self.data_type = 'Acc'  # 'Acc' 'Dis'
        self.duration = '0.5sec'
        self.num_sensors = 9
        self.damage_case = 'SingleE'  # 'SingleE' 'TwoE'
        self.split_test = True
        self.dr = 0.0

    def evaluate(self, model, X, y, mode=''):
        score = model.evaluate(X, y, verbose=0)
        print(mode + '_mse:', score[0])
        print(mode + '_R2:', score[1])
        return score

    def save_results(self, model, history, tr_score, val_score, te_score, tag=None):
        # f = open("{}history\\{}\\{}{}.pckl".format(self.dir, tag, self.tc, self.sub_tc), "wb")
        # h = history.history
        # pickle.dump(h, f)
        # f.close()
        filename = '{}{}.pckl'.format(self.tc, self.sub_tc)
        filepath = os.path.join(self.dir, 'history', tag, filename)
        
        f = open(filepath, 'wb')
        h = history.history
        pickle.dump(h, f)
        f.close()

        with open("{}/history/{}/{}.txt".format(self.dir, tag, self.tc), "a+") as text_file:
            if self.bottleneck is True:

                text_file.write("Deep DenseNet-Bottleneck-Compressed Network-%d-%d-%d created."
                                % (self.depth, self.growth_rate, self.nb_dense_block))
            else:
                text_file.write("Deep DenseNet Network-%d-%d-%d created."
                                % (self.depth, self.growth_rate, self.nb_dense_block))
            text_file.write("Test_Case: %s\n Training mse: %f R2: %f\n"
                            "Validation mse:%f R2:%f\n Testing mse:%f R2:%f\n"
                            "params:%s\n"
                            % (self.sub_tc, tr_score[0], tr_score[1], val_score[0],
                               val_score[1], te_score[0], te_score[1], self.__dict__))
            # model.summary(print_fn=lambda x: text_file.write(x + '\n'))
        text_file.close()

        with open("{}/history/{}/{}arch.txt".format(self.dir, tag, self.tc), "a+") as text_file:
            if self.bottleneck is True:

                text_file.write("Deep DenseNet-Bottleneck-Compressed Network-%d-%d-%d created."
                                % (self.depth, self.growth_rate, self.nb_dense_block))
            else:
                text_file.write("Deep DenseNet Network-%d-%d-%d created."
                                % (self.depth, self.growth_rate, self.nb_dense_block))
            text_file.write("Test_Case: %s\n params:%s\n" % (self.tc + self.sub_tc, self.__dict__))
            model.summary(print_fn=lambda x: text_file.write(x + '\n'))
        text_file.close()

    def R(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def R2(self, y_true, y_pred):
        SS_res = np.sum(np.square(y_true - y_pred))
        SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def save_ori_space_results(self, model, scaler, X, y, mode, tag=None):

        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y)
        # model = LinearRegression().fit(y_true, y_pred)
        # r_sq = model.score(y_true, y_pred)
        # print('coefficient of determination:', r_sq)
        R = self.R2(y_true, y_pred)
        mse = np.square(np.subtract(y_true, y_pred)).mean()
        with open("{}/history/{}/{}.txt".format(self.dir, tag, self.tc), "a+") as text_file:
            text_file.write("Original_Space: %s mse: %f R2: %f\n" % (mode, mse, R))
            # text_file.write("R2_Space: %s mse: %f R2: %f\n" % (mode, mse, r_sq))

        text_file.close()
        sio.savemat(self.dir + '/prediction/' + tag + '/' + self.tc + self.sub_tc + '_' + mode + '.mat',
                    {'y_true': y_true, 'y_pred': y_pred})

    def save_results_R2(self, model, X, y, mode, tag=None):

        y_pred = model.predict(X)
        model = LinearRegression().fit(y, y_pred)
        r_sq = model.score(y, y_pred)
        print('coefficient of determination:', r_sq)
        mse = np.square(np.subtract(y, y_pred)).mean()

        with open("{}/history/{}/{}.txt".format(self.dir, tag, self.tc), "a+") as text_file:
            text_file.write("R2_Space: %s mse: %f R2: %f\n" % (mode, mse, r_sq))
        text_file.close()

    def Vec_to_Img(self, X, include_fre=True):
        if include_fre:
            print('Frequency is included.')
            Freq = X[:, 0:7]
            MS = X[:, 7::]
            X = np.concatenate((np.expand_dims(Freq, 2), MS.reshape(X.shape[0], self.img_rows, self.img_cols)), 2)
            X = np.expand_dims(X, 3).astype('float32')

        else:
            X = X[:, 7::]
            X = X.reshape(X.shape[0], self.img_rows, self.img_cols)
            X = np.expand_dims(X, 3).astype('float32')

        return X

    def whiten_2(self, X):
        # Assume input data matrix X of size [N x D]
        mu = np.mean(X, axis=0)
        X -= mu  # zero-center the data (important)

        cov = np.dot(X.T, X) / X.shape[0]  # get the data covariance matrix
        U, S, V = np.linalg.svd(cov)
        Xrot = np.dot(X, U)  # decorrelate the data
        #         Xrot_reduced = np.dot(X, U) # Xrot_reduced becomes [N x 100]
        #         Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
        Xwhite = Xrot / np.sqrt(S + 1e-5)
        return Xwhite, mu, U, S

    def save_ori_space_pred(self, model, scaler, X, y, mode, tag=None):

        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y)
        sio.savemat(self.dir + '/prediction/' + tag + '/' + self.tc + self.sub_tc + '_' + mode + '.mat',
                    {'y_true': y_true, 'y_pred': y_pred})

    def save_pred(self, model, X, y, mode, tag=None):
        y_pred = model.predict(X)
        sio.savemat(self.dir + '/prediction/' + tag + '/' + self.tc + self.sub_tc + '_' + mode + '.mat',
                    {'y_true': y, 'y_pred': y_pred})

    def load_data(self, sensor_idx=None):
        if sensor_idx is not None:
            mat_fname = '{}/FEMP/{}{}{}Sensor{}{}.mat'.format(self.data_dir, self.damage_case, self.data_type,
                                                            self.num_sensors,
                                                            self.duration, sensor_idx)

        else:
            mat_fname = '{}/FEMP/{}{}{}Sensor{}.mat'.format(self.data_dir, self.damage_case, self.data_type,
                                                          self.num_sensors,
                                                          self.duration)
        print('Loading file: ' + mat_fname)

        if self.duration is '5sec':
            with h5py.File(mat_fname, 'r') as f:
                X, y = np.array(f['Acceleration']).T, np.array(f['Output70']).T
                del f
        else:
            # mat_contents = sio.loadmat(mat_fname)
            mat_contents = h5py.File(mat_fname, 'r')
            X, y = mat_contents['Acceleration'], mat_contents['Output70']
        
        # If only one channel, use the below script to add a channel axis
        # X = np.expand_dims(X, -1)
        return X, y
        
    def load_test_data(self, sensor_idx = None):
        if sensor_idx is not None:
            mat_fname = os.path.join(self.data_dir, 'FEMP', 'Testing_{}{}{}Sensor{}{}.mat'.format(self.damage_case, self.data_type,
                                                                                    self.num_sensors,
                                                                                    self.duration, sensor_idx))

        else:
            mat_fname = os.path.join(self.data_dir, 'FEMP', 'Testing_{}{}{}Sensor{}.mat'.format(self.damage_case, self.data_type,
                                                                                    self.num_sensors,
                                                                                    self.duration))
        print('Loading file: ' + mat_fname)

        if self.duration is '5sec':
            with h5py.File(mat_fname, 'r') as f:
                X, y = np.array(f['Acceleration']).T, np.array(f['Output70']).T
                del f
        else:
            # mat_contents = sio.loadmat(mat_fname)
            mat_contents = h5py.File(mat_fname, 'r')
            X, y = mat_contents['Acceleration'], mat_contents['Output70']
        
        # If only one channel, use the below script to add a channel axis
        # X = np.expand_dims(X, -1)
        X = np.transpose(X, (3, 2, 1, 0))
        y = np.transpose(y, (1, 0))
        return X, y
        
    def train(self, tag=None, opt_str='sgd', model_str=None):
        seed = 42
        # #-------Loading Dataset----------------------------------------------------------------------------------------#
        X, y = self.load_data()
        X = np.transpose(X, (3, 2, 1, 0))
        y = np.transpose(y, (1, 0))
        print('X shape:', X.shape)
        print('y shape:', y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
        # #-------Pre-processing Dataset----------------------------------------------------------------------------------------#
        if self.process_y:
            y_sc = MinMaxScaler((-1, 1))
            y_train = y_sc.fit_transform(y_train)
            y_val = y_sc.transform(y_val)
            y_test = y_sc.transform(y_test)
        else:
            y_sc = None

        if self.whitening:
            X_train, mu, U, S = self.whiten_2(X_train)
            X_val = np.dot(X_val - mu, U) / np.sqrt(S + 1e-5)
            X_test = np.dot(X_test - mu, U) / np.sqrt(S + 1e-5)
        if self.mapminmax:
            X_sc = MinMaxScaler()
            print(X_train.shape)
            X_train = X_sc.fit_transform(X_train)
            X_val = X_sc.transform(X_val)
            X_test = X_sc.transform(X_test)
        if self.standardize:
            X_sc = StandardScaler()
            X_train = X_sc.fit_transform(X_train)
            X_val = X_sc.transform(X_val)
            X_test = X_sc.transform(X_test)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples', 'min:', np.min(X_train), 'max:', np.max(X_train))
        print(X_val.shape[0], 'val samples', 'min:', np.min(X_val), 'max:', np.max(X_val))
        print(X_test.shape[0], 'test samples', 'min:', np.min(X_test), 'max:', np.max(X_test))

        print('y_train shape:', y_train.shape)
        print(y_train.shape[0], 'train samples', 'min:', np.min(y_train), 'max:', np.max(y_train))
        print(y_val.shape[0], 'val samples', 'min:', np.min(y_val), 'max:', np.max(y_val))
        print(y_test.shape[0], 'test samples', 'min:', np.min(y_test), 'max:', np.max(y_test))

        # #-------Building Training Model----------------------------------------------------------------------------------------#
        if model_str is 'DenseNets':

            model = DenseNet((self.img_rows, self.img_cols, self.channels), depth=self.depth,
                             nb_dense_block=self.nb_dense_block,
                             growth_rate=self.growth_rate, bottleneck=self.bottleneck, reduction=self.reduction,
                             classes=self.classes,
                             dropout_rate=self.dr, filter_size=self.kernel, weight_decay=self.l2_wd,
                             trans_filter=self.trans_filter, init_kernel=self.init_kernel, init_stride=self.init_stride)


        elif model_str is 'ResNets':
            model = ResNet((self.img_rows, self.img_cols, self.channels), depth=self.depth,
                           nb_dense_block=self.nb_dense_block,
                           growth_rate=self.growth_rate, bottleneck=self.bottleneck, reduction=self.reduction,
                           classes=self.classes,
                           dropout_rate=self.dr, filter_size=self.kernel, weight_decay=self.l2_wd,
                           trans_filter=self.trans_filter, init_kernel=self.init_kernel, init_stride=self.init_stride)

        elif model_str is 'ConvNets':
            model = ResNet((self.img_rows, self.img_cols, self.channels), depth=self.depth,
                           nb_dense_block=self.nb_dense_block,
                           growth_rate=self.growth_rate, bottleneck=self.bottleneck, reduction=self.reduction,
                           classes=self.classes,
                           dropout_rate=self.dr, filter_size=self.kernel, weight_decay=self.l2_wd,
                           trans_filter=self.trans_filter, init_kernel=self.init_kernel, init_stride=self.init_stride,
                           cnn=True)

        model.summary()
        # opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        if opt_str is 'sgd':
            opt = optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                             patience=20,
                                             verbose=1,
                                             factor=0.5,
                                             min_lr=1e-4)
        else:

            opt = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                             patience=5,
                                             verbose=1,
                                             factor=0.5,
                                             min_lr=1e-7)
        # You are using the triangular learning rate policy and
        #  base_lr (initial learning rate which is the lower boundary in the cycle) is 0.1
        #         clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001,mode='triangular')

        # opt = AdaBound(lr=self.lr,
        #                 final_lr=0.01,
        #                 gamma=1e-03,
        #                 weight_decay=0.,
        #                 amsbound=True)
        model.compile(loss=['mean_squared_error'],
                      optimizer=opt,
                      metrics=[self.R])

        tensorboard = TrainValTensorBoard(sub_dir=self.tc, sub_dir2=self.sub_tc)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
        # filepath = self.dir + 'savedmodel/' + self.tc + self.sub_tc + 'best_val_weights.hdf5'
        # filepath_temp = os.path.join(self.dir, 'savedmodel', self.tc, self.sub_tc, 'best_val_weights.hdf5')
        filename = 'best_val_weights.hdf5'
        filepath = os.path.join('.', filename)
        checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                       save_best_only=True, save_weights_only=False,
                                       mode='auto', period=1)
        # lr_reduction = ReduceLROnPlateau(monitor='val_loss',
        #                                  patience=10,
        #                                  verbose=1,
        #                                  factor=0.5,
        #                                  min_lr=1e-7)
        #         lr_reduction = ReduceLROnPlateau(monitor='val_loss',
        #                                          patience=10,
        #                                          verbose=1,
        #                                          factor=0.9,
        #                                          min_lr=1e-7)
        # todo plot history tune learning rate
        # -------Training The Model----------------------------------------------------------------------------------------#

        history = model.fit(X_train, y_train,
                            batch_size=self.batch_size,
                            validation_data=(X_val, y_val),
                            epochs=self.n_epoch,
                            callbacks=[lr_reduction, tensorboard, checkpointer], verbose=1)

        model.load_weights(filepath)
        # model.save('{}savedmodel/{}/{}{}.h5'.format(self.dir, self.damage_case, self.tc, self.sub_tc))
        output_directory = os.path.join(self.dir, 'savedmodel', self.damage_case)
        os.makedirs(output_directory, exist_ok=True)
        filepath = os.path.join(self.dir, 'savedmodel', self.damage_case, '{}{}.h5'.format(self.tc, self.sub_tc))
        model.save(filepath)
        # os.remove(filepath)

        # -------Evaluating The Model----------------------------------------------------------------------------------------#

        tr_score = self.evaluate(model, X_train, y_train, mode='Training')
        val_score = self.evaluate(model, X_val, y_val, mode='Validation')
        te_score = self.evaluate(model, X_test, y_test, mode='Testing')

        self.save_results(model, history, tr_score, val_score, te_score, tag)
        if self.process_y:
            self.save_ori_space_results(model, y_sc, X_train, y_train, 'Training', tag)
            self.save_ori_space_results(model, y_sc, X_val, y_val, 'Validation', tag)
            self.save_ori_space_results(model, y_sc, X_test, y_test, 'Testing', tag)

        else:
            self.save_pred(model, X_test, y_test, 'Testing', tag)
            
        X_ex_test, y_ex_test = self.load_test_data()
        # MinMaxScaler
        y_ex_test_sc = y_sc.transform(y_ex_test)

        ex_score = self.evaluate(model, X_ex_test, y_ex_test_sc, mode='Ex_Testing')

        y_pred_scaled = model.predict(X_ex_test)  # 假设使用模型进行预测得到的预测结果为 y_pred_scaled
        y_pred = y_sc.inverse_transform(y_pred_scaled)
        print('y_ex_test_sclaed:', y_ex_test_sc)
        print('y_pred_sclaed:', y_pred_scaled)
        print('y_pred:', y_pred)

# %%
if __name__ == '__main__':
    FM = FrameDamageDetection2D_Acceleration()
    model_str = 'ConvNets'
    FM.tc = FM.damage_case + model_str

    FM.train(tag=FM.damage_case, opt_str='adam', model_str=model_str)

# %%

