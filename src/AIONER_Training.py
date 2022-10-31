# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:49:23 2021

@author: luol2
"""

import os
import sys
import argparse
import random
from model_ner import HUGFACE_NER
from processing_data import ml_intext,out_BIO_BERT_softmax,out_BIO_BERT_crf
from evaluation_BIO import NER_Evaluation_fn
from tensorflow.keras import callbacks
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)
#tf.compat.v1.disable_eager_execution()


class NERCallback_PLM(callbacks.Callback):
    def __init__(self, temp_files):
        super(NERCallback_PLM, self).__init__()
        self.tempout = temp_files['infiles']
        self.index_2_label=temp_files['index_2_label']
        self.model_out=temp_files['model_out']
        self.dev_set=temp_files['dev_set']
        self.decoder_type=temp_files['decoder_type']
        
    def on_train_begin(self, logs=None):
        self.max_dev=0.0
        self.max_dev_epoch=0
        self.max_train=0.0
        self.max_train_epoch=0
        self.patein_es=0

    def on_epoch_end(self, epoch, logs=None):
        #_lr=0
        current_acc = logs.get("accuracy")
        #print(current_acc)
        self.patein_es+=1
        #print(self.model.optimizer._decayed_lr(tf.float32))
        _lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        if self.dev_set!=[]:
            print('......dev performance:')
            _dev_predict = self.model.predict(self.dev_set[0])
            #print(_dev_predict)
            if self.decoder_type=='crf':
                out_BIO_BERT_crf(self.tempout['devtemp'],_dev_predict,self.dev_set[1],self.index_2_label)
            elif self.decoder_type=='softmax':
                out_BIO_BERT_softmax(self.tempout['devtemp'],_dev_predict,self.dev_set[1],self.index_2_label)
    
            dev_f1=NER_Evaluation_fn(self.tempout['devtemp'])
            
            if dev_f1>self.max_dev:
                self.max_dev=dev_f1
                self.max_dev_epoch=epoch+1
                self.model.save_weights(self.model_out['BEST'])
        
        if current_acc >self.max_train:
            self.max_train = current_acc
            self.max_train_epoch = epoch+1
            self.model.save_weights(self.model_out['ES'])
            self.patein_es=0
        if self.patein_es>5:
            self.model.stop_training = True
        
        if self.dev_set!=[]:
            print('\nmax_train_acc=',self.max_train,'max_epoch:',self.max_train_epoch,'max_dev_f1=',self.max_dev,'max_epoch:',self.max_dev_epoch, 'lr:',_lr,'cur_epoch:',epoch+1)
        else:
            print('\nmax_train_acc=',self.max_train,'max_epoch:',self.max_train_epoch,'lr:',_lr,'cur_epoch:',epoch+1)


    
def Hugface_training(infiles,vocabfiles,model_out,decoder_type='softmax'):
    

    #build model
    plm_model=HUGFACE_NER(vocabfiles)
    plm_model.build_encoder() #PubmedBERT,ELECTRA
    if decoder_type=='crf': 
        plm_model.build_crf_decoder()
    elif decoder_type=='softmax':
        plm_model.build_softmax_decoder()
    else:
        print('decoder type is error!')
        sys.exit()
    

    #load dataset
    print('loading dataset......')  
    trainfile=infiles['trainfile']
    train_list = ml_intext(trainfile)
    
        
    print('numpy dataset......')
    if decoder_type=='crf': 
        train_x, train_y,train_bert_text_label = plm_model.rep.load_data_hugface(train_list,word_max_len=plm_model.maxlen,label_type='crf') #softmax
        if infiles['devfile']!='':
            devfile=infiles['devfile']
            dev_list = ml_intext(devfile)
            dev_x, dev_y,dev_bert_text_label = plm_model.rep.load_data_hugface(dev_list,word_max_len=plm_model.maxlen,label_type='crf')
    elif decoder_type=='softmax':
        train_x, train_y,train_bert_text_label = plm_model.rep.load_data_hugface(train_list,word_max_len=plm_model.maxlen,label_type='softmax') #softmax
        if infiles['devfile']!='':
            devfile=infiles['devfile']
            dev_list = ml_intext(devfile)
            dev_x, dev_y,dev_bert_text_label = plm_model.rep.load_data_hugface(dev_list,word_max_len=plm_model.maxlen,label_type='softmax')
    #print(train_x)
    #print(train_y)

    #train model
    if infiles['devfile']!='':
        temp_files={'infiles':infiles,
                    'index_2_label':plm_model.rep.index_2_label,
                    'model_out':model_out,
                    'dev_set':[dev_x,dev_bert_text_label],
                    'decoder_type':decoder_type
                    }
    else:
         temp_files={'infiles':infiles,
                    'index_2_label':plm_model.rep.index_2_label,
                    'model_out':model_out,
                    'dev_set':[],
                    'decoder_type':decoder_type
                    }   
    plm_model.model.fit(train_x,train_y, batch_size=32, epochs=50,verbose=2,callbacks=[NERCallback_PLM(temp_files)])
                             

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='train NER model, python NER_Training.py -t trainfile -v valfile -e encoder -d decoder -o outpath')
    parser.add_argument('--trainfile', '-t', help="the training set file",default='../data/conll/Project_All-AIO_entity.conll')
    parser.add_argument('--valfile', '-v', help="the validation set file",default='')
    parser.add_argument('--encoder', '-e', help="encoder (bioformer or pubmedbert?)",default='bioformer')
    parser.add_argument('--decoder', '-d', help="decoder (crf or softmax?)",default='softmax')
    parser.add_argument('--outpath', '-o', help="the model output folder",default='../models/')
    args = parser.parse_args()
    if args.outpath[-1]!='/':
        args.outpath+='/'
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
   
    
    infiles={'trainfile':args.trainfile,
             'devfile':args.valfile,
             'devtemp':args.outpath+str(random.randint(10000,50000))+'_tmp_ner.conll',
             }


    
    if args.encoder=='pubmedbert':
        vocabfiles={'labelfile':'../vocab/AIO_label.vocab',
                    'checkpoint_path':'../pretrained_models/BiomedNLP-PubMedBERT-base-uncased-abstract/',
                    'lowercase':True,
                    }
     
    elif args.encoder=='bioformer':
        vocabfiles={'labelfile':'../vocab/AIO_label.vocab',
                    'checkpoint_path':'../pretrained_models/bioformer-cased-v1.0/',
                    'lowercase':False,
                    }

    model_out={'BEST':args.outpath+args.encoder+'-'+args.decoder+'-best-AIO.h5',
               'ES':args.outpath+args.encoder+'-'+args.decoder+'-es-AIO.h5'}        


    Hugface_training(infiles,vocabfiles,model_out,decoder_type=args.decoder)
        
    if os.path.exists(infiles['devtemp']):  #delete tmp file
        os.remove(infiles['devtemp'])
                             
