# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:02:20 2022

@author: luol2
"""


import argparse
import os
import time
import re
import io
import bioc

from model_ner import HUGFACE_NER
from processing_data import ml_intext_fn,out_BIO_BERT_crf_fn,out_BIO_BERT_softmax_fn
from restore_index import NN_restore_index_fn

import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)


import stanza
nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'},package='None') #package='craft'


def pre_token(sentence):
    sentence=re.sub("([\=\/\(\)\<\>\+\-\_])"," \\1 ",sentence)
    sentence=re.sub("[ ]+"," ",sentence);
    return sentence

def ssplit_token(in_text,entity_type,max_len=400):
    #print('max_len:',max_len)
    fout=io.StringIO()

    in_text=in_text.strip()
    in_text=pre_token(in_text)
    doc_stanza = nlp(in_text)
    strlen=0
    for sent in doc_stanza.sentences:
        fout.write('<'+entity_type+'>\tO\n')
        for word in sent.words:
            strlen+=1
            fout.write(word.text+'\tO\n')
            if strlen>=max_len:
                # print('long sentence:',strlen)
                fout.write('\n')
                strlen=0
        fout.write('</'+entity_type+'>\tO\n')
        fout.write('\n')
        strlen=0           
    return fout.getvalue()

def ml_tagging(ml_input,nn_model,decoder_type='crf'):

    
    test_list = ml_intext_fn(ml_input)
    if decoder_type=='crf': 
        test_x,test_y, test_bert_text_label=nn_model.rep.load_data_hugface(test_list,word_max_len=nn_model.maxlen,label_type='crf')
    elif decoder_type=='softmax': 
        test_x,test_y, test_bert_text_label=nn_model.rep.load_data_hugface(test_list,word_max_len=nn_model.maxlen,label_type='softmax')

    test_pre = nn_model.model.predict(test_x,batch_size=64)
    if decoder_type=='crf':
        test_decode_temp=out_BIO_BERT_crf_fn(test_pre,test_bert_text_label,nn_model.rep.index_2_label)
    elif decoder_type=='softmax':
        test_decode_temp=out_BIO_BERT_softmax_fn(test_pre,test_bert_text_label,nn_model.rep.index_2_label)

    return test_decode_temp

# only machine learning-based method
def ML_Tag(text,ml_model,decoder_type='crf',entity_type='ALL'):

    conll_in=ssplit_token(text, entity_type, max_len=ml_model.maxlen)
    #print(ssplit_token) 
#    print('ssplit token:',time.time()-startTime)
    
#    startTime=time.time()
    ml_tsv=ml_tagging(conll_in,ml_model,decoder_type=decoder_type)
    #print('ml_tsv:\n',ml_tsv)
#    print('ml ner:',time.time()-startTime)
   
    final_result= NN_restore_index_fn(text,ml_tsv)
    # print('final ner:',time.time()-startTime)
    
    return final_result

def NER_PubTator(infile,outfile,nn_model,para_set):

    with open(infile, 'r',encoding='utf-8') as fin:
        with open(outfile,'w', encoding='utf8') as fout:
            title=''
            abstract=''
            all_text=fin.read().strip().split('\n\n')
            Total_n=len(all_text)
            print('Total number of sub-documents:', Total_n)
            doc_num=0
            for doc in all_text:
                print("Processing:{0}%".format(round(doc_num * 100 / Total_n)), end="\r")
                doc_num+=1
                lines = doc.split('\n')
                seg=lines[0].split('|t|')
                pmid=seg[0]
                title=seg[1]
                seg=lines[1].split('|a|')
                abstract=seg[1]
                
                intext=title+' '+abstract
                tag_result=ML_Tag(intext,nn_model,decoder_type=para_set['decoder_type'],entity_type=para_set['entity_type'])
                fout.write(lines[0]+'\n'+lines[1]+'\n')
                for ele in tag_result:
                    ent_start = ele[0]
                    ent_last = ele[1]
                    ent_mention = intext[int(ele[0]):int(ele[1])]
                    ent_type=ele[2]
                    fout.write(pmid+"\t"+ent_start+"\t"+ent_last+"\t"+ent_mention+"\t"+ent_type+"\n")
                fout.write('\n')
                title=''
                abstract=''

def NER_BioC(infile,outfile,nn_model,para_set):

    with open(infile, 'r',encoding='utf-8') as fin:
        with open(outfile,'w', encoding='utf-8') as fout:
            collection = bioc.load(fin)
            
            Total_n=len(collection.documents)
            print('Total number of sub-documents:', Total_n)
            doc_num=0
            for document in collection.documents:
                print("Processing:{0}%".format(round(doc_num * 100 / Total_n)), end="\r")
                doc_num+=1
                # print(document.id)
                mention_num=0
                for passage in document.passages:
                    if passage.text!='' and (not passage.text.isspace()) and passage.infons['type']!='ref': # have text and is not ref
                        passage_offset=passage.offset
                        tag_result=ML_Tag(passage.text,nn_model,decoder_type=para_set['decoder_type'],entity_type=para_set['entity_type'])
                        for ele in tag_result:
                            bioc_note = bioc.BioCAnnotation()
                            bioc_note.id = str(mention_num)
                            mention_num+=1
                            bioc_note.infons['type'] = ele[2]
                            start = int(ele[0])
                            last = int(ele[1])
                            loc = bioc.BioCLocation(offset=str(passage_offset+start), length= str(last-start))
                            bioc_note.locations.append(loc)
                            bioc_note.text = passage.text[start:last]
                            passage.annotations.append(bioc_note)
            bioc.dump(collection, fout, pretty_print=True)
    
   

def NER_main_path(inpath, para_set, outpath, modelfile):
    
    #check if file exsited
    files_exsited=0
    for infile in os.listdir(inpath):
        if os.path.isfile(outpath+infile):
            pass
            # print(infile+' has exsited.')
        else:
            files_exsited=1
            break
    if files_exsited==0:
        print('All files have exsited.')
    else:
    
        print('loading models........')    
        
        
        if para_set['encoder_type']=='pubmedbert':
            vocabfiles={'labelfile':para_set['vocabfile'],
                        'checkpoint_path':'../pretrained_models/BiomedNLP-PubMedBERT-base-uncased-abstract/',
                        'lowercase':True,
                        } 
        elif para_set['encoder_type']=='bioformer':
            vocabfiles={'labelfile':para_set['vocabfile'],
                        'checkpoint_path':'../pretrained_models/bioformer-cased-v1.0/',
                        'lowercase':False,
                        }    
        
        nn_model=HUGFACE_NER(vocabfiles)
        nn_model.build_encoder()
        if para_set['decoder_type']=='crf': 
            nn_model.build_crf_decoder()
        elif para_set['decoder_type']=='softmax':
            nn_model.build_softmax_decoder()
            
        nn_model.load_model(modelfile)
            
        #tagging text
        print("begin tagging........")
        start_time=time.time()
        
        for infile in os.listdir(inpath):
            if os.path.isfile(outpath+infile):
                print(infile+' has exsited.')
            else:
                print('processing:',infile)             
                fin = open(inpath+infile, 'r',encoding='utf-8')
                input_format=""
                for line in fin:
                    pattern_bioc = re.compile('.*<collection>.*')
                    pattern_pubtator = re.compile('^([^\|]+)\|[^\|]+\|(.*)')
                    if pattern_pubtator.search(line):
                        input_format="PubTator"
                        break
                    elif pattern_bioc.search(line):
                        input_format="BioC"
                        break
                fin.close()
                if(input_format == "PubTator"):
                    NER_PubTator(inpath+infile,outpath+infile,nn_model,para_set)
                elif(input_format == "BioC"):
                    NER_BioC(inpath+infile,outpath+infile,nn_model,para_set)    
        
        print('tag done:',time.time()-start_time)
    


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='run AIONER, python NER_Tagging.py -i input -m NERmodel -e ALL -o output')
    parser.add_argument('--inpath', '-i', help="input path",default='../example/input/')
    parser.add_argument('--model', '-m', help="trained deep learning NER model file",default='../pretrained_models/AIONER/Bioformer-softmax-AIONER.h5')
    parser.add_argument('--entity', '-e', help="predict entity type (Gene, Chemical, Disease, Variant, Species, CellLine, ALL)",default='ALL')
    parser.add_argument('--vocabfile', '-v', help="vocab file with BIO label",default='../vocab/AIO_label.vocab')
    parser.add_argument('--outpath', '-o', help="output path to save the NER tagged results",default='../example/output/')
    args = parser.parse_args()
    
    if args.outpath[-1]!='/':
        args.outpath+='/'
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    print('==============\n| AIONER |\n==============')
    
    model_paras=args.model.split('/')[-1].split('-')
    para_set={
              'encoder_type':model_paras[0].lower(), # pubmedbert or bioformer
              'decoder_type':model_paras[1].lower(),# crf or softmax
              'entity_type':args.entity,
              'vocabfile':args.vocabfile
              }
    print('run parameters:', para_set)

    NER_main_path(args.inpath, para_set, args.outpath, args.model)
    

