import os
import sys
import keras
import json
import csv
import numpy as np
import keras.backend as K
from Adam_mult import AdamWarmup, calc_train_steps
from checkpoint_loader import load_model
from keras.callbacks import  ModelCheckpoint


def metric_cor (y_true, y_pred):
        n = K.sum(K.ones_like(y_true))
        sum_x = K.sum(y_true)
        sum_y = K.sum(y_pred)
        sum_x_sq = K.sum(K.square(y_true))
        sum_y_sq = K.sum(K.square(y_pred))
        psum = K.sum(y_true * y_pred)
        num = psum - (sum_x * sum_y / n)
        den = K.sqrt((sum_x_sq - K.square(sum_x) / n) *  (sum_y_sq - K.square(sum_y) / n))
        return (num / den)


class training:

        def __init__(self, 
                     BERT_CONFIG_PATH=None,
                     CHECKPOINT_PATH=None, 
                     NUM_CLASSES=2, 
                     SEQ_LEN=128,
                     EPOCHS=5,
                     BATCH_SIZE=64,
                     TASK=None,
                     OUTPUT_DIR=None,
                     train_data=None,
                     dev_data=None,
                     LOGFILE_PATH=None):

                self.BERT_CONFIG_PATH = BERT_CONFIG_PATH
                self.CHECKPOINT_PATH = CHECKPOINT_PATH
                self.NUM_CLASSES = NUM_CLASSES
                self.SEQ_LEN = SEQ_LEN
                self.EPOCHS = EPOCHS
                self.BATCH_SIZE = BATCH_SIZE
                self.TASK = TASK
                self.OUTPUT_DIR = OUTPUT_DIR
                self.train_data = train_data
                self.dev_data = dev_data
                self.LOGFILE_PATH = LOGFILE_PATH
                self.NUM_TRAIN = train_data[1].shape[0]

                self.loss = 'sparse_categorical_crossentropy'
                self.metric = ['sparse_categorical_accuracy']
                self.validation_metric = 'val_sparse_categorical_accuracy'

                if self.TASK == "sts-b":
                        self.loss = 'mean_squared_error'
                        self.metric = [metric_cor,'mae']
                        self.validation_metric = 'val_metric_cor'


        def fine_tuning_step(self,
                             LR_BERT=0.00003):

                fine_tuned_model = load_model(
                    self.BERT_CONFIG_PATH,
                    self.CHECKPOINT_PATH,
                    FLAG_BERT_PRETRAINED=True,
                    output_dim=self.NUM_CLASSES,
                    seq_len=self.SEQ_LEN,
                    FLAG_EXTRACT_LAYER=0,
                    TASK=self.TASK)
                decay_steps, warmup_steps = calc_train_steps(
                    self.NUM_TRAIN,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS,
                )
                fine_tuned_model.compile(
                    AdamWarmup(decay_steps=decay_steps, 
                                warmup_steps=warmup_steps, 
                                lr=LR_BERT, 
                                lr_mult=None),
                    loss=self.loss,
                    metrics=self.metric,
                )
                print ("Fine-tuned model summary: ", fine_tuned_model.summary())
                
                SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"finetune.hdf5")
                checkpoint = ModelCheckpoint(SAVE_CP_PATH, monitor=self.validation_metric, verbose=1, save_best_only=True, mode='max')
                history = fine_tuned_model.fit(self.train_data[0], 
                                     self.train_data[1], 
                                     batch_size=self.BATCH_SIZE, 
                                     epochs=self.EPOCHS, 
                                     validation_data=(self.dev_data[0], self.dev_data[1], None),
                                     verbose=1, 
                                     callbacks=[checkpoint])
                with open(self.LOGFILE_PATH, 'a') as fp:
                        fp.write("\n Fine-tuned model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

                keras.backend.clear_session()

                return SAVE_CP_PATH


        def configuration_search_step(self, 
                                      fine_tuned_model_path=None, 
                                      LAMBDA=0.0001, 
                                      LR_BERT=0.00003, 
                                      LR_SOFT_EXTRACT=0.0001):

                ## Define a PoWER-BERT model containing Soft-Extract Layers
                configuration_search_model = load_model(
                    self.BERT_CONFIG_PATH,
                    self.CHECKPOINT_PATH,
                    FLAG_BERT_PRETRAINED=True,
                    output_dim=self.NUM_CLASSES,
                    seq_len=self.SEQ_LEN,
                    LAMBDA=LAMBDA,
                    FLAG_EXTRACT_LAYER=1,
                    TASK=self.TASK)
                configuration_search_model.load_weights(fine_tuned_model_path, by_name=True)

                decay_steps, warmup_steps = calc_train_steps(
                    self.NUM_TRAIN,
                    batch_size=self.BATCH_SIZE,
                    epochs=5, #self.EPOCHS,
                )

                ## Set different Learning rates for original BERT parameters and the retnetion parameters fo the Soft-Extract Layers
                lr_mult = {}
                for layer in configuration_search_model.layers:
                        if 'Extract' in layer.name:
                                lr_mult[layer.name] = 1.0
                        else:
                                lr_mult[layer.name] = LR_BERT/LR_SOFT_EXTRACT

                configuration_search_model.compile(
                    AdamWarmup(decay_steps=decay_steps, 
                                warmup_steps=warmup_steps, 
                                lr=LR_SOFT_EXTRACT, 
                                lr_mult=lr_mult),
                    loss=self.loss,
                    metrics=self.metric,
                )
                print ("Configuration Search model summary: ", configuration_search_model.summary())
                
                ## Train the model
                configuration_search_model.fit(self.train_data[0],
                                               self.train_data[1],
                                               batch_size=self.BATCH_SIZE, 
                                               epochs=5, #self.EPOCHS, 
                                               validation_data=(self.dev_data[0], self.dev_data[1], None),
                                               verbose=1)
                SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,'configuration_search_model.hdf5')
                configuration_search_model.save(os.path.join(SAVE_CP_PATH))

                ## Obtain the retention configuration by calculating the mass of each encoder layer
                retention_configuration = self.get_configuration(configuration_search_model)
                with open(self.LOGFILE_PATH, 'a') as fp:
                        fp.write("\n Retention Configuration :" + str(retention_configuration))

                keras.backend.clear_session()
        
                return SAVE_CP_PATH, retention_configuration


        def get_configuration(self, configuration_search_model):

                retention_configuration = []
                for layer in configuration_search_model.layers:
                    for i in range(1,13):
                        if layer.name == 'Encoder-' + str(i) + '-MultiHeadSelfAttention-Soft-Extract':
                            ww = layer.get_weights()
                            weight_sum = int(np.sum(ww[0]))
                            if weight_sum == 0:
                                weight_sum = 1
                            if len(retention_configuration) == 0 or weight_sum < retention_configuration[-1]:
                                retention_configuration.append(weight_sum)
                            else:
                                retention_configuration.append(retention_configuration[-1])
                print ("Retention Configuration :", retention_configuration, np.sum(retention_configuration))

                return retention_configuration


        def retraining_step(self, 
                            configuration_search_model_path=None, 
                            retention_configuration=[], 
                            LR_BERT=0.00003):   

                ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
                retrained_model = load_model(
                    self.BERT_CONFIG_PATH,
                    self.CHECKPOINT_PATH,
                    FLAG_BERT_PRETRAINED=True,
                    output_dim=self.NUM_CLASSES,
                    seq_len=self.SEQ_LEN,
                    retention_configuration=retention_configuration,
                    FLAG_EXTRACT_LAYER=2,
                    TASK=self.TASK)
                decay_steps, warmup_steps = calc_train_steps(
                    self.NUM_TRAIN,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS,
                )
                retrained_model.load_weights(configuration_search_model_path, by_name=True)

                retrained_model.compile(
                    AdamWarmup(decay_steps=decay_steps, 
                                warmup_steps=warmup_steps, 
                                lr=LR_BERT, 
                                lr_mult=None),
                    loss=self.loss,
                    metrics=self.metric,
                )
                print ("Re-trained model summary: ", retrained_model.summary())

                SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.hdf5")
                checkpoint = ModelCheckpoint(SAVE_CP_PATH, monitor=self.validation_metric, verbose=1, save_best_only=True, mode='max')
                history = retrained_model.fit(self.train_data[0],
                                    self.train_data[1],
                                    batch_size=self.BATCH_SIZE, 
                                    epochs=self.EPOCHS, 
                                    validation_data=(self.dev_data[0], self.dev_data[1], None),
                                    verbose=1,
                                    callbacks=[checkpoint])
                with open(self.LOGFILE_PATH, 'a') as fp:
                        fp.write("\n Re-trained model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

                return retrained_model

