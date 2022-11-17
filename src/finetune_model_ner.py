# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:08:09 2021

@author: luol2
"""
import tensorflow as tf
from represent_ner import Hugface_RepresentationLayer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tf_crf2 import CRF
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam
from transformers import TFBertModel, BertConfig,TFElectraModel,TFAutoModel
import numpy as np
import sys


class LRSchedule_LINEAR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=5e-5,
        init_warmup_lr=0.0,
        final_lr=5e-7,
        warmup_steps=0,
        decay_steps=0,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.init_warmup_lr=init_warmup_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        """ linear warm up - linear decay """
        if self.warmup_steps>0:
            warmup_lr = (self.init_lr - self.init_warmup_lr)/self.warmup_steps * step+self.init_warmup_lr
        else:
            warmup_lr=1000.0
        #print('\n.......warmup_lr:',warmup_lr)
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.init_lr - (step - self.warmup_steps)/self.decay_steps*(self.init_lr - self.final_lr)
        )
        #print('\n.....decay_lr:',decay_lr)
        return tf.math.minimum(warmup_lr,decay_lr)


        
class HUGFACE_NER(): #huggingface transformers
    def __init__(self, model_files):
        self.model_type='HUGFACE'
        self.maxlen = 256  #sent 256 doc-512,pretrain-sent 128
        self.checkpoint_path = model_files['checkpoint_path']
        self.label_file=model_files['labelfile']
        self.lowercase=model_files['lowercase']
        self.rep = Hugface_RepresentationLayer(self.checkpoint_path, self.label_file, lowercase=self.lowercase)
        
            
    def build_encoder(self):
        print('...vocab len:',self.rep.vocab_len)
        plm_model = TFAutoModel.from_pretrained(self.checkpoint_path, from_pt=True)
        plm_model.resize_token_embeddings(self.rep.vocab_len) 
        x1_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='input_ids')
        x2_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='token_type_ids')
        x3_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='attention_mask')
        x = plm_model(x1_in, token_type_ids=x2_in, attention_mask=x3_in)[0]
        #dense = TimeDistributed(Dense(512, activation='relu'), name='dense1')(x)
        self.encoder = Model (inputs=[x1_in,x2_in,x3_in], outputs=x,name='hugface_encoder')
        self.encoder.summary()
        
    def build_softmax_decoder(self):

        x1_in = Input(shape=(self.maxlen,),dtype=tf.int32)
        x2_in = Input(shape=(self.maxlen,),dtype=tf.int32)
        x3_in = Input(shape=(self.maxlen,),dtype=tf.int32)  
        features = self.encoder([x1_in,x2_in,x3_in])
        #features = Dropout(0.4)(features)
        features = TimeDistributed(Dense(128, activation='relu'), name='dense2')(features)
        features= Dropout(0.1)(features)
        output = TimeDistributed(Dense(self.rep.label_table_size, activation='softmax'), name='softmax_new')(features)
        self.model = Model(inputs=[x1_in,x2_in,x3_in], outputs=output, name="hugface_softmax")

        lr_schedule=LRSchedule_LINEAR(
            init_lr=2e-5,
            init_warmup_lr=1e-7,
            final_lr=1e-5,
            warmup_steps=0,
            decay_steps=400)

        opt = Adam(learning_rate = lr_schedule)
        #opt = Adam(lr=5e-6) 
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        self.model.summary()
        
    def build_crf_decoder(self):

        x1_in = Input(shape=(self.maxlen,),dtype=tf.int32)
        x2_in = Input(shape=(self.maxlen,),dtype=tf.int32)
        x3_in = Input(shape=(self.maxlen,),dtype=tf.int32)  
        features = self.encoder([x1_in,x2_in,x3_in])
        #features = Dropout(0.4)(features)
        features = TimeDistributed(Dense(128, activation='relu'), name='dense2')(features)
        features= Dropout(0.1)(features)
        crf=CRF(self.rep.label_table_size,name='crf_layer_new')
        output = crf(features)
        self.model = Model(inputs=[x1_in,x2_in,x3_in], outputs=output, name="hugface_crf")

        lr_schedule=LRSchedule_LINEAR(
            init_lr=2e-5,
            init_warmup_lr=0.0,
            final_lr=1e-5,
            warmup_steps=0,
            decay_steps=400)
        opt = Adam(learning_rate = lr_schedule)
        self.model.compile(
            optimizer=opt,
            loss=crf.get_loss,
            metrics=['accuracy'],
        )
        self.model.summary()
        
    def load_model(self,model_file):
        self.model.load_weights(model_file, by_name=True)
        self.model.summary()  
        print('load HUGFACE model done!')
