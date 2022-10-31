# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:34:12 2020

@author: luol2
"""
import numpy as np
import io
import sys
#read ner text (word\tlabel), generate the list[[[w1,label],[w2,label]]]
def ml_intext(file):
    fin=open(file,'r',encoding='utf-8')
    alltexts=fin.read().strip().split('\n\n')
    fin.close()
    data_list=[]

    for sents in alltexts:
        lines=sents.split('\n')
        temp_sentece=[]
        for i in range(0,len(lines)):
            seg=lines[i].split('\t')
            temp_sentece.append(seg[:])
        
        data_list.append(temp_sentece)
    #print(data_list)
    #print(label_list)
    return data_list

def ml_intext_fn(ml_input):
    fin=io.StringIO(ml_input)
    alltexts=fin.read().strip().split('\n\n')
    fin.close()
    data_list=[]

    for sents in alltexts:
        lines=sents.split('\n')
        temp_sentece=[]
        for i in range(0,len(lines)):
            seg=lines[i].split('\t')
            temp_sentece.append(seg[:])
        
        data_list.append(temp_sentece)
    #print(data_list)
    #print(label_list)
    return data_list


def out_BIO_BERT_softmax(file,raw_pre,raw_input,label_set):
    fout=open(file,'w',encoding='utf-8')
    for i in range(len(raw_input)):
        for j in range(len(raw_input[i])):
            if raw_input[i][j][-1]<len(raw_pre[i]):
                # label_id = raw_pre[i][j]
                label_id = np.argmax(raw_pre[i][raw_input[i][j][-1]])
                label_tag = label_set[str(label_id)]
            else:
                label_tag='O'
            fout.write(raw_input[i][j][0]+'\t'+raw_input[i][j][1]+'\t'+label_tag+'\n')
        fout.write('\n')
    fout.close() 

def out_BIO_BERT_softmax_fn(raw_pre,raw_input,label_set):
    fout=io.StringIO()
    for i in range(len(raw_input)):
        for j in range(len(raw_input[i])):
            if raw_input[i][j][-1]<len(raw_pre[i]):
                #label_id = raw_pre[i][j]
                label_id = np.argmax(raw_pre[i][raw_input[i][j][-1]])
                label_tag = label_set[str(label_id)]
            else:
                label_tag='O'
            fout.write(raw_input[i][j][0]+'\t'+raw_input[i][j][1]+'\t'+label_tag+'\n')
        fout.write('\n')
    return fout.getvalue()

def out_BIO_BERT_crf(file,raw_pre,raw_input,label_set):
    fout=open(file,'w',encoding='utf-8')
    for i in range(len(raw_input)):
        
        for j in range(len(raw_input[i])):
            if raw_input[i][j][-1]<len(raw_pre[i]):
                label_id = raw_pre[i][raw_input[i][j][-1]] 
                label_tag = label_set[str(label_id)]
            else:
                label_tag='O'
            fout.write(raw_input[i][j][0]+'\t'+raw_input[i][j][1]+'\t'+label_tag+'\n')
        fout.write('\n')
    fout.close() 

def out_BIO_BERT_crf_fn(raw_pre,raw_input,label_set):
    fout=io.StringIO()
    for i in range(len(raw_input)):
        
        for j in range(len(raw_input[i])):
            if raw_input[i][j][-1]<len(raw_pre[i]):
                label_id = raw_pre[i][raw_input[i][j][-1]] 
                label_tag = label_set[str(label_id)]
            else:
                label_tag='O'
            fout.write(raw_input[i][j][0]+'\t'+raw_input[i][j][1]+'\t'+label_tag+'\n')
        fout.write('\n')
    return fout.getvalue()                  





