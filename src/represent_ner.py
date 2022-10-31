# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:54:17 2021

@author: luol2
"""



import os, sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

class Hugface_RepresentationLayer(object):
    
    
    def __init__(self, tokenizer_name_or_path, label_file,lowercase=True):
        

        #load vocab
        #self.bert_vocab_dict = {}
        #self.cased=cased
        #self.load_bert_vocab(vocab_path,self.bert_vocab_dict)
        self.model_type='bert'
        #self.model_type='roberta'
        if self.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True,do_lower_case=lowercase)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True,do_lower_case=lowercase)
        
        self.tokenizer.add_tokens(["<Chemical>","</Chemical>","<Disease>","</Disease>","<CellLine>","</CellLine>","<Gene>","</Gene>","<Species>","</Species>","<Variant>","</Variant>","<ALL>","</ALL>"])
        
        #load label
        self.label_2_index={}
        self.index_2_label={}
        self.label_table_size=0
        self.load_label_vocab(label_file,self.label_2_index,self.index_2_label)
        self.label_table_size=len(self.label_2_index)
        self.vocab_len=len(self.tokenizer)
       
    def load_label_vocab(self,fea_file,fea_index,index_2_label):
        
        fin=open(fea_file,'r',encoding='utf-8')
        all_text=fin.read().strip().split('\n')
        fin.close()
        for i in range(0,len(all_text)):
            fea_index[all_text[i]]=i
            index_2_label[str(i)]=all_text[i]
            
    
    def generate_label_list(self,ori_tokens,labels,word_index): #tonly first subtoken is B, other is I
        label_list=['O']*len(word_index)

        label_list_index=[]
        old_new_token_map=[]
        ori_i=0
        first_index=-1
        i=0
        while i <len(word_index):
            if word_index[i]==None:
                label_list_index.append(self.label_2_index[label_list[i]])
                i+=1
            else:
                first_index=word_index[i]
                if first_index==ori_i:
                    old_new_token_map.append(i)
                    ori_i+=1
                label_list[i]=labels[word_index[i]]
                label_list_index.append(self.label_2_index[label_list[i]])
                i+=1
                while word_index[i]==first_index and word_index[i]!=None:
                    #print(first_index)
                    if labels[first_index].startswith("B-"):
                        label_list[i]='I-'+labels[first_index][2:]
                        label_list_index.append(self.label_2_index[label_list[i]])
                    else:
                        label_list[i]=labels[word_index[i]]
                        label_list_index.append(self.label_2_index[label_list[i]])
                    i+=1
                        
        bert_text_label=[]
        #print(len(old_new_token_map))
        for i in range(0,len(ori_tokens)):
            if i<len(old_new_token_map):
                bert_text_label.append([ori_tokens[i],labels[i],old_new_token_map[i]])
            else: # after token > max len
                break
        return label_list_index,bert_text_label
    
    def load_data_hugface(self,instances,  word_max_len=100, label_type='crf'):
    
        x_index=[]
        x_seg=[]
        x_mask=[]
        y_list=[]
        bert_text_labels=[]
        max_len=0
        over_num=0
        maxT=word_max_len
        ave_len=0

        
        
        for sentence in instances:                           
            sentence_text_list=[]
            label_list=[]
            for j in range(0,len(sentence)):
                sentence_text_list.append(sentence[j][0])
                label_list.append(sentence[j][-1])

            token_result=self.tokenizer(
                sentence_text_list,
                max_length=word_max_len,
                truncation=True,is_split_into_words=True)
            
            bert_tokens=self.tokenizer.convert_ids_to_tokens(token_result['input_ids'])
            word_index=token_result.word_ids(batch_index=0)
            ave_len+=len(bert_tokens)
            if len(sentence_text_list)>max_len:
                max_len=len(sentence_text_list)
            if len(bert_tokens)==maxT:
                over_num+=1

            x_index.append(token_result['input_ids'])
            if self.model_type in {"gpt2", "roberta"}:
                x_seg.append([0]*len(token_result['input_ids']))
            else:
                x_seg.append(token_result['token_type_ids'])
            x_mask.append(token_result['attention_mask'])
            
            #print('\nsentence_text_list:',len(sentence_text_list),sentence_text_list)
            #print('\nlabel:',len(label_list),label_list)
            #print('\nword_index:',len(word_index),word_index)
            #print('\nbert_tokens:',len(bert_tokens),bert_tokens)
            label_list,bert_text_label=self.generate_label_list(sentence_text_list,label_list,word_index) # the label list after bert token, ori token/lable/new index
            #print('\nlabel list:',len(label_list),label_list)
            #print('\nbert_text_label:',len(bert_text_label),bert_text_label)
            #sys.exit()
            y_list.append(label_list)
            #print(y_list)
            bert_text_labels.append(bert_text_label)

        
        x1_np = pad_sequences(x_index, word_max_len, value=0, padding='post',truncating='post')  # right padding
        x2_np = pad_sequences(x_seg, word_max_len, value=0, padding='post',truncating='post')
        x3_np = pad_sequences(x_mask, word_max_len, value=0, padding='post',truncating='post')
        y_np = pad_sequences(y_list, word_max_len, value=0, padding='post',truncating='post')

        if label_type=='softmax':
            y_np = np.expand_dims(y_np, 2)
        elif label_type=='crf':
            pass
        
        return [x1_np, x2_np,x3_np], y_np,bert_text_labels  
         

if __name__ == '__main__':
    pass
    
 
            
