import sys
sys.path.append('../')
sys.path.append('../../')

import datetime
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv3D, \
                                    Input, concatenate, Dropout, \
                                    BatchNormalization, ReLU, MaxPool3D
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

from DataPrep import DATA

# Set random state
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Set test size -- then do not touch
TRAIN_TEST_SPLIT = 0.8

class ROI_DNN:
    def __init__(self, ROI_nums):
        self.ROI_nums = ROI_nums
        self.len_ROI = len(ROI_nums)
        self.model = None

    #def get_data(self, balanced=1, batch_size=20, tra_val_split=0.8):
    def get_data(self, balanced=1, batch_size=20, tra_val_split=0.8, use_validation=True):
        self.use_validation = use_validation
        Data = DATA()
        Data.Fetch_OASIS(balanced=balanced)
        Data.Train_Test(TRAIN_TEST_SPLIT, random=RANDOM_SEED)
        selectors = self.ROI_nums
        Data.Add_MRI(selectors)
        Data.Split_Data()

        # GET TRAINING AND TEST SETS
        X = Data.features_train
        y = Data.labels_train
        y[y>0] = 1
        y[y<=0] = 0
        y.shape = (len(y), 1)

        X_test = Data.features_test
        y_test = Data.labels_test
        y_test[y_test>0] = 1
        y_test[y_test<=0] = 0
        y_test.shape = (len(y_test), 1)

        self.y_test = y_test
        self.X_test = X_test

        # SPLIT TRAINING INTO TRAINING/VALIDATION
        len_yt = y.shape[0]
        if use_validation:
            training_size = floor(tra_val_split*len_yt)
        else:
            training_size = len_yt - 1
        y_tra = y[:training_size]
        X_tra = X[:training_size, ...]

        y_val = y[training_size:]
        X_val = X[training_size:, ...]

        # CREATE TENSORFLOW DATASETS
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_tra, y_tra)).shuffle(3000).batch(batch_size)
        if use_validation:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        else:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(1)

    def data_augmentation(self):
        pass

    def build_model(self, small_dense, big_dense, activation='relu'):
        len_ROI = self.len_ROI
        self.model = None

        class Variable_ROI_DNN(Model):
            def __init__(self, len_ROI, small_dense, big_dense):
                super().__init__()

                denses = []
                denses.append(Dense(small_dense, activation=activation))
                denses.append(Dense(big_dense, activation=activation))
                denses.append(Dense(small_dense, activation=activation))
                denses.append(Dense(10, activation=activation))

                dense_out = Dense(2) # output digit

                self.denses = denses
                self.dense_out = dense_out
        
            def call(self, x):
                denses = self.denses
                dense_out = self.dense_out

                for j in range(len(denses)):
                    x = denses[j](x)

                out = self.dense_out(x)

                return out

        self.model = Variable_ROI_DNN(len_ROI, small_dense, big_dense)

    def run(self, lr=1e-5, epochs=500):
        train_ds = self.train_ds
        val_ds = self.val_ds

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        @tf.function
        def train_step(model, optimizer, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
            train_loss(loss)
            train_accuracy(labels, predictions)
        
        @tf.function
        def val_step(model, images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
        
            val_loss(t_loss)
            val_accuracy(labels, predictions)

        model = self.model
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs_roi_dnn/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs_roi_dnn/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        EPOCHS = epochs
        
        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(model, optimizer, images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
            for val_images, val_labels in val_ds:
                val_step(model, val_images, val_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  val_loss.result(),
                                  val_accuracy.result() * 100))

        self.model = model

        return(train_loss.result().numpy(), 
               train_accuracy.result().numpy(), 
               val_loss.result().numpy(), 
               val_accuracy.result().numpy())

    def test(self):
        model = self.model
        X_test = self.X_test
        y_test = self.y_test

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        predictions = model(X_test, training=False)
        t_loss = loss_object(y_test, predictions)
        
        test_loss(t_loss)
        test_accuracy(y_test, predictions)

        return(test_loss.result().numpy(),
               test_accuracy.result().numpy())


class ROI_CNN:
    def __init__(self, ROI_nums):
        self.ROI_nums = ROI_nums
        self.len_ROI = len(ROI_nums)
        self.model = None

    def get_data(self, balanced=1, tra_val_split=0.8, use_validation=True):
        self.use_validation = use_validation
        Data = DATA()
        Data.Fetch_OASIS(balanced=balanced)
        Data.Train_Test(TRAIN_TEST_SPLIT, random=RANDOM_SEED)
        Data.Split_Data()

        # GET TRAINING AND TEST SETS
        features_train = Data.features_train
        y = Data.labels_train
        y[y>0] = 1
        y[y<0] = 0
        y.shape = (len(y), 1)

        features_test = Data.features_test
        y_test = Data.labels_test
        y_test[y_test>0] = 1
        y_test[y_test<=0] = 0
        y_test.shape = (len(y_test), 1)

        Data.load_images()
        selectors = self.ROI_nums
        Data.get_3D_ROI(selectors)
        ROIs_3D_gm = Data.ROIs_3D_gm
        ROIs_3D_wm = Data.ROIs_3D_wm

        idx_train = Data.idx_train
        idx_test = Data.idx_test

        ROIs_3D_gm_train = []
        ROIs_3D_gm_test = []
        ROIs_3D_wm_train = []
        ROIs_3D_wm_test = []
        for k in range(self.len_ROI):
            ROIs_3D_gm_train.append(ROIs_3D_gm[k][idx_train, ...])
            ROIs_3D_gm_test.append(ROIs_3D_gm[k][idx_test, ...])
            ROIs_3D_wm_train.append(ROIs_3D_wm[k][idx_train, ...])
            ROIs_3D_wm_test.append(ROIs_3D_wm[k][idx_test, ...])

        # SPLIT TRAINING INTO TRAINING/VALIDATION
        len_yt = y.shape[0]
        if use_validation:
            train_size = floor(tra_val_split*len_yt)
        else:
            train_size = len_yt-1
        y_tra = y[:train_size]
        features_tra = features_train[:train_size, ...]

        y_val = y[train_size:]
        features_val = features_train[train_size:, ...]
        train_ROIs_3D_gm = []
        val_ROIs_3D_gm = []
        train_ROIs_3D_wm = []
        val_ROIs_3D_wm = []
        for k in range(self.len_ROI):
            train_ROIs_3D_gm.append(ROIs_3D_gm_train[k][:train_size, ...])
            val_ROIs_3D_gm.append(ROIs_3D_gm_train[k][train_size:, ...])

            train_ROIs_3D_wm.append(ROIs_3D_wm_train[k][:train_size, ...])
            val_ROIs_3D_wm.append(ROIs_3D_wm_train[k][train_size:, ...])

        X_tra = train_ROIs_3D_gm + train_ROIs_3D_wm + [features_tra]
        X_val = val_ROIs_3D_gm + val_ROIs_3D_wm + [features_val]
        X_test = ROIs_3D_gm_test + ROIs_3D_wm_test  + [features_test]

        self.y_test = y_test
        self.X_test = X_test

        self.y_tra = y_tra
        self.X_tra = X_tra

        self.y_val = y_val
        self.X_val = X_val

    def set_tf_datasets(self, batch_size=20):
        X_tra = self.X_tra
        X_tra = tuple(X_tra)
        y_tra = self.y_tra

        X_val = self.X_val
        X_val = tuple(X_val)
        y_val = self.y_val

        X_test = self.X_test
        X_test = tuple(X_test)
        y_test = self.y_test

        self.train_ds = tf.data.Dataset.from_tensor_slices((X_tra, y_tra)).shuffle(5000).batch(batch_size)
        #self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        if self.use_validation:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        else:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(1)
        #self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def data_augmentation(self, kwargs, num=10):
        y_tra = self.y_tra
        X_tra = self.X_tra

        for key, value in kwargs.items():
            if key == 'rotation':
                X_rot, y_rot =  ROI_CNN.random_rotate_ROIs(X_tra, y_tra, value, num=num)
            #elif key == 'translation':
            #    X_trans, y_trans = ROI_CNN.random_translate_ROIs(X_train, y_train, value, num=num)
            #elif key == 'noise':
            #    X_noise, y_noise = ROI_CNN.random_noise_ROIs(X_train, y_train, sigma=value, num=num)

        for i in range(len(X_tra)):
            X_tra[i] = np.concatenate((X_tra[i], X_rot[i]), axis=0)
            #X_train[i] = np.concatenate((X_train[i], X_trans[i]), axis=0)
            #X_train[i] = np.concatenate((X_train[i], X_noise[i]), axis=0)


        y_tra = np.concatenate((y_tra, y_rot), axis=0)
        #y_train = np.concatenate((y_train, y_trans), axis=0)
        #y_train = np.concatenate((y_train, y_noise), axis=0)

        self.y_tra = y_tra
        self.X_tra = X_tra

    @staticmethod
    def random_rotate_ROIs(X, y, angle_range, num=4):
        imgs = X[:-1]
        features = X[-1]
        features_out = np.copy(features)
        y_out = np.copy(y)
        
        n = y.shape[0]

        final_ROIs = []

        for ROI in imgs:
            ROI_out = np.zeros([num*n]+list(ROI.shape[1:]))
            count=0
            for j in range(num):
                angles = angle_range*(2*np.random.random(size=n)-1)
                for i in range(n):
                    angle = angles[i]
                    axis = np.random.randint(0, high=3)
                    if axis == 0:
                        axes = (1,0)
                    elif axis == 1:
                        axes = (2,0)
                    elif axis == 2:
                        axes = (2,1)

                    Xn = ROI[i, ...]

                    Xi = rotate(Xn, angle, axes=axes, reshape=False)
                    #Xi.shape = [1] + list(Xi.shape)
                    ROI_out[count, ...] = Xi
                    count += 1

            final_ROIs.append(ROI_out)

        if num > 1:
            for j in range(num-1):
                features_out = np.concatenate((features_out, features), axis=0)
                y_out = np.concatenate((y_out, y), axis=0)

        X_out = final_ROIs + [features_out]

        return X_out, y_out

    @staticmethod
    def random_noise_ROIs(X, y, sigma=0.01, num=4):
        imgs = X[:-1]
        features = X[-1]
        features_out = np.copy(features)
        y_out = np.copy(y)
        
        n = y.shape[0]

        final_ROIs = []
        mean = 0.0

        for ROI in imgs:
            ROI_out = np.zeros([num*n]+list(ROI.shape[1:]))
            count=0
            for j in range(num):
                noise = np.random.normal(mean, sigma, ROI.shape)
                ROI_out[j*n:(j+1)*n] = ROI + noise

            final_ROIs.append(ROI_out)

        if num > 1:
            for j in range(num-1):
                features_out = np.concatenate((features_out, features), axis=0)
                y_out = np.concatenate((y_out, y), axis=0)

        X_out = final_ROIs + [features_out]

        return X_out, y_out

    @staticmethod
    def random_translate_ROIs(X, y, pixel_range, num=4):
        imgs = X[:-1]
        features = X[-1]
        features_out = np.copy(features)
        y_out = np.copy(y)
        
        n = y.shape[0]

        final_ROIs = []

        for ROI in imgs:
            ROI_out = np.zeros([num*n]+list(ROI.shape[1:]))
            count=0
            for j in range(num):
                pixels = np.random.randint(-pixel_range, high=pixel_range+1, size=n)
                for i in range(n):
                    pixel = pixels[i]
                    axis = np.random.randint(0, high=3)

                    Xn = ROI[i, ...]

                    Xi = np.roll(Xn, pixel, axis=axis)
                    ROI_out[count, ...] = Xi
                    count += 1


            final_ROIs.append(ROI_out)
        if num > 1:
            for j in range(num-1):
                features_out = np.concatenate((features_out, features), axis=0)
                y_out = np.concatenate((y_out, y), axis=0)

        X_out = final_ROIs + [features_out]

        return X_out, y_out
        

    def build_model(self, small_filter, big_filter):
        len_ROI = self.len_ROI
        self.model = None

        class Variable_ROI_CNN(Model):
            def __init__(self, len_ROI, small_filter, big_filter):
                super().__init__()
                convs = []
                convs2 = []
                #maxpools = []
                #maxpools2 = []
                flattens = []
                for i in range(len_ROI):
                    convs.append(Conv3D(small_filter, 3, activation='relu', data_format='channels_last'))
                    convs2.append(Conv3D(big_filter, 3, activation='relu', data_format='channels_last'))
                    #maxpools.append(MaxPool3D())
                    #convs2.append(Conv3D(big_filter, 3, activation='relu', data_format='channels_last'))
                    #maxpools2.append(MaxPool3D())
                    flattens.append(Flatten())
                dense_features = Dense(5, activation='relu')

                dense1 = Dense(50, activation='relu')
                dense2 = Dense(20, activation='relu')
                dense_out = Dense(2) # output digit

                self.convs = convs
                #self.maxpools = maxpools
                self.convs2 = convs2
                #self.maxpools2 = maxpools2

                self.flattens = flattens
                self.dense_features = dense_features

                self.dense1 = dense1
                self.dense2 = dense2
                self.dense_out = dense_out
        
            def call(self, x):
                xouts = []

                for i in range(len(self.convs)):
                    xc = x[i]
                    xc = self.convs[i](xc)
                    #xc = self.maxpools[i](xc)
                    xc = self.convs2[i](xc)
                    #xc = self.maxpools2[i](xc)
                    xc = self.flattens[i](xc)
                    xouts.append(xc)

                xouts.append(self.dense_features(x[-1]))
        
                x = concatenate(xouts)
                x = self.dense1(x)
                x = self.dense2(x)
                out = self.dense_out(x)

                return out

        self.model = Variable_ROI_CNN(len_ROI, small_filter, big_filter)

    def run(self, lr=1e-4, epochs=50):
        train_ds = self.train_ds
        val_ds = self.val_ds

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        @tf.function
        def train_step(model, optimizer, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
            train_loss(loss)
            train_accuracy(labels, predictions)
        
        @tf.function
        def val_step(model, images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
        
            val_loss(t_loss)
            val_accuracy(labels, predictions)

        model = self.model
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs_ROI_cnn/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs_ROI_cnn/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        EPOCHS = epochs
        
        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(model, optimizer, images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
            for val_images, val_labels in val_ds:
                val_step(model, val_images, val_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  val_loss.result(),
                                  val_accuracy.result() * 100))

        self.model = model

        return(train_loss.result().numpy(), 
               train_accuracy.result().numpy(), 
               val_loss.result().numpy(), 
               val_accuracy.result().numpy())

    def test(self):
        model = self.model
        test_ds = self.test_ds
        X_test = self.X_test
        y_test = self.y_test

        #X_test, y_test = test_ds

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        predictions = model(X_test, training=False)
        t_loss = loss_object(y_test, predictions)
        
        test_loss(t_loss)
        test_accuracy(y_test, predictions)

        return(test_loss.result().numpy(),
               test_accuracy.result().numpy())
        

class CNN_SUBJECT_LEVEL:
    def __init__(self):
        self.model = None

    def get_data(self, balanced=1, tra_val_split=0.8, use_validation=True):
        self.use_validation = use_validation
        Data = DATA()
        Data.Fetch_OASIS(balanced=balanced)
        Data.Train_Test(TRAIN_TEST_SPLIT, random=RANDOM_SEED)
        Data.Split_Data()

        # GET TRAINING AND TEST SETS
        features_train = Data.features_train
        y = Data.labels_train
        y[y>0] = 1
        y[y<0] = 0
        y.shape = (len(y), 1)

        features_test = Data.features_test
        y_test = Data.labels_test
        y_test[y_test>0] = 1
        y_test[y_test<=0] = 0
        y_test.shape = (len(y_test), 1)

        Data.load_images()
        gm_imgs_3D = Data.gm_imgs_3D
        wm_imgs_3D = Data.wm_imgs_3D

        gm_imgs_3D = gm_imgs_3D[..., np.newaxis]
        wm_imgs_3D = wm_imgs_3D[..., np.newaxis]

        idx_train = Data.idx_train
        idx_test = Data.idx_test

        gm_imgs_3D_train = gm_imgs_3D[idx_train, ...]
        wm_imgs_3D_train = wm_imgs_3D[idx_train, ...]

        gm_imgs_3D_test = gm_imgs_3D[idx_test, ...]
        wm_imgs_3D_test = wm_imgs_3D[idx_test, ...]

        # SPLIT TRAINING INTO TRAINING/VALIDATION
        len_yt = y.shape[0]
        if use_validation:
            train_size = floor(tra_val_split*len_yt)
        else:
            train_size = len_yt-1

        y_tra = y[:train_size]
        features_tra = features_train[:train_size, ...]

        y_val = y[train_size:]
        features_val = features_train[train_size:, ...]
        
        gm_imgs_3D_val = gm_imgs_3D_train[train_size:, ...]
        wm_imgs_3D_val = wm_imgs_3D_train[train_size:, ...]

        gm_imgs_3D_tra = gm_imgs_3D_train[:train_size, ...]
        wm_imgs_3D_tra = wm_imgs_3D_train[:train_size, ...]

        X_tra = [gm_imgs_3D_tra] + [wm_imgs_3D_tra] + [features_tra]
        X_val = [gm_imgs_3D_val] + [wm_imgs_3D_val] + [features_val]
        X_test = [gm_imgs_3D_test] + [wm_imgs_3D_test] + [features_test]

        self.y_test = y_test
        self.X_test = X_test

        self.y_tra = y_tra
        self.X_tra = X_tra

        self.y_val = y_val
        self.X_val = X_val

    def set_tf_datasets(self, batch_size=20):
        X_tra = self.X_tra
        X_tra = tuple(X_tra)
        y_tra = self.y_tra

        X_val = self.X_val
        X_val = tuple(X_val)
        y_val = self.y_val

        X_test = self.X_test
        X_test = tuple(X_test)
        y_test = self.y_test

        self.train_ds = tf.data.Dataset.from_tensor_slices((X_tra, y_tra)).shuffle(5000).batch(batch_size)
        #self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        if self.use_validation:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        else:
            self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(1)
        #self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def data_augmentation(self, kwargs, num=1):
        y_tra = self.y_tra
        X_tra = self.X_tra

        for key, value in kwargs.items():
            if key == 'rotation':
                X_rot, y_rot =  CNN_SUBJECT_LEVEL.random_rotate(X_tra, y_tra, value, num=num)
            #elif key == 'translation':
            #    X_trans, y_trans = ROI_CNN.random_translate_ROIs(X_train, y_train, value, num=num)
            #elif key == 'noise':
            #    X_noise, y_noise = ROI_CNN.random_noise_ROIs(X_train, y_train, sigma=value, num=num)

        for i in range(len(X_tra)):
            X_tra[i] = np.concatenate((X_tra[i], X_rot[i]), axis=0)
            #X_train[i] = np.concatenate((X_train[i], X_trans[i]), axis=0)
            #X_train[i] = np.concatenate((X_train[i], X_noise[i]), axis=0)


        y_tra = np.concatenate((y_tra, y_rot), axis=0)
        #y_train = np.concatenate((y_train, y_trans), axis=0)
        #y_train = np.concatenate((y_train, y_noise), axis=0)

        self.y_tra = y_tra
        self.X_tra = X_tra

    @staticmethod
    def random_rotate(X, y, angle_range, num=4):
        imgs = X[:-1]
        features = X[-1]
        features_out = np.copy(features)
        y_out = np.copy(y)
        
        n = y.shape[0]

        final_imgs = []

        for img in imgs:
            img_out = np.zeros([num*n]+list(img.shape[1:]))
            count=0
            for j in range(num):
                angles = angle_range*(2*np.random.random(size=n)-1)
                for i in range(n):
                    angle = angles[i]
                    axis = np.random.randint(0, high=3)
                    if axis == 0:
                        axes = (1,0)
                    elif axis == 1:
                        axes = (2,0)
                    elif axis == 2:
                        axes = (2,1)

                    Xn = img[i, ...]

                    Xi = rotate(Xn, angle, axes=axes, reshape=False)
                    #Xi.shape = [1] + list(Xi.shape)
                    img_out[count, ...] = Xi
                    count += 1

            final_imgs.append(img_out)

        if num > 1:
            for j in range(num-1):
                features_out = np.concatenate((features_out, features), axis=0)
                y_out = np.concatenate((y_out, y), axis=0)

        X_out = final_imgs + [features_out]

        return X_out, y_out
        

    def build_model(self):
        self.model = None

        class Variable_CNN(Model):
            def __init__(self):
                super().__init__()
                convs1 = []
                batchnorms1 = []
                ReLUs1 = []
                convs2 = []
                batchnorms2 = []
                ReLUs2 = []
                flattens = []
                for i in range(2):
                    #convs1.append(Conv3D(1, 3, activation='linear', data_format='channels_last'))
                    convs1.append(Conv3D(2, 3, activation='linear', data_format='channels_last'))
                    batchnorms1.append(BatchNormalization())
                    ReLUs1.append(ReLU())

                    #convs2.append(Conv3D(2, 3, activation='linear', data_format='channels_last'))
                    convs2.append(Conv3D(4, 3, activation='linear', data_format='channels_last'))
                    batchnorms2.append(BatchNormalization())
                    ReLUs2.append(ReLU())

                    flattens.append(Flatten())
                dense_features = Dense(5, activation='relu')

                dense1 = Dense(150, activation='relu')
                #dense2 = Dense(20, activation='relu')
                dense_out = Dense(2) # output digit

                self.convs1 = convs1
                self.batchnorms1 = batchnorms1
                self.ReLUs1 = ReLUs1

                self.convs2 = convs2
                self.batchnorms2 = batchnorms2
                self.ReLUs2 = ReLUs2

                self.flattens = flattens
                self.dense_features = dense_features

                self.dense1 = dense1
                #self.dense2 = dense2
                self.dense_out = dense_out
        
            def call(self, x):
                xouts = []

                for i in range(len(self.convs1)):
                    xc = x[i]
                    xc = self.convs1[i](xc)
                    xc = self.batchnorms1[i](xc)
                    xc = self.ReLUs1[i](xc)
                    xc = self.convs2[i](xc)
                    xc = self.batchnorms2[i](xc)
                    xc = self.ReLUs2[i](xc)
                    xc = self.flattens[i](xc)
                    xouts.append(xc)

                xouts.append(self.dense_features(x[-1]))
        
                x = concatenate(xouts)
                x = self.dense1(x)
                #x = self.dense2(x)
                out = self.dense_out(x)

                return out

        self.model = Variable_CNN()

    def run(self, lr=1e-4, epochs=50):
        train_ds = self.train_ds
        val_ds = self.val_ds

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        @tf.function
        def train_step(model, optimizer, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
            train_loss(loss)
            train_accuracy(labels, predictions)
        
        @tf.function
        def val_step(model, images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
        
            val_loss(t_loss)
            val_accuracy(labels, predictions)

        model = self.model
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs_cnn/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs_cnn/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        EPOCHS = epochs
        
        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(model, optimizer, images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
            for val_images, val_labels in val_ds:
                val_step(model, val_images, val_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  val_loss.result(),
                                  val_accuracy.result() * 100))


        self.model = model

        return(train_loss.result().numpy(), 
               train_accuracy.result().numpy(), 
               val_loss.result().numpy(), 
               val_accuracy.result().numpy())

    def test(self):
        model = self.model
        test_ds = self.test_ds
        X_test = self.X_test
        y_test = self.y_test

        #X_test, y_test = test_ds

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        predictions = model(X_test, training=False)
        t_loss = loss_object(y_test, predictions)
        
        test_loss(t_loss)
        test_accuracy(y_test, predictions)

        return(test_loss.result().numpy(),
               test_accuracy.result().numpy())


if __name__ == '__main__':
    pass
