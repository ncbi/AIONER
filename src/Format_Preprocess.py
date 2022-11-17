# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:58:03 2022

@author: luol2
"""



import stanza
import sys
import os
import io
import json
import re
import argparse

def pre_token(sentence):
    sentence=re.sub("([\=\/\(\)\<\>\+\-\_])"," \\1 ",sentence)
    sentence=re.sub("[ ]+"," ",sentence);
    return sentence


#由于之前默认实体索引是降序，但是发现有bug不一定降序，所以先排序去重
def pubtator_entitysort(infile,entity_map):
    
    fin=open(infile,'r',encoding='utf-8')    
    # fout=open(path+'LitCoin/sort/Train_sort.PubTator','w',encoding='utf-8')
    fout=io.StringIO()
    all_in=fin.read().strip().split('\n\n')
    fin.close()
    error_dict={} #use to debug error
    for doc in all_in:
        entity_dict={}
        lines=doc.split('\n')
        fout.write(lines[0]+'\n'+lines[1]+'\n')
        for i in range(2,len(lines)):
            segs=lines[i].split('\t')
            if len(segs)>4 and (segs[4] in entity_map.keys()):
                if lines[i] not in entity_dict.keys():
                    entity_dict[lines[i]]=int(segs[1])
                else:
                    print('entity have in',lines[i])
                    if segs[0] not in error_dict.keys():
                        error_dict[segs[0]]=[lines[i]]
                    else:
                        if lines[i] not in error_dict[segs[0]]:
                            error_dict[segs[0]].append(lines[i])

        entity_sort=sorted(entity_dict.items(), key=lambda kv:(kv[1]), reverse=False)
        for ele in entity_sort:
            fout.write(ele[0]+'\n')
        fout.write('\n')
    return fout

def filter_overlap(infile): #nonest

    fin=io.StringIO(infile.getvalue())
    fout=io.StringIO()
    
    documents=fin.read().strip().split('\n\n')
    fin.close()
    total_entity=0
    over_entity=0
    nest_entity=0
    for doc in documents:
        lines=doc.split('\n')
        entity_list=[]
        if len(lines)>2:
            first_entity=lines[2].split('\t')
            nest_list=[first_entity]
            max_eid=int(first_entity[2])
            total_entity+=len(lines)-2
            for i in range(3,len(lines)):
                segs=lines[i].split('\t')
                if int(segs[1])> max_eid:
                    if len(nest_list)==1:
                        entity_list.append(nest_list[0])
                        nest_list=[]
                        nest_list.append(segs)
                        if int(segs[2])>max_eid:
                            max_eid=int(segs[2])
                    else:
                        # print(nest_list)
                        nest_entity+=len(nest_list)-1
                        tem=find_max_entity(nest_list)#find max entity
                        # if len(tem)>1:
                        #     print('max nest >1:',tem)
                        entity_list.extend(tem)
                        nest_list=[]
                        nest_list.append(segs)
                        if int(segs[2])>max_eid:
                            max_eid=int(segs[2])
                        
                else:
                    nest_list.append(segs)
                    if int(segs[2])>max_eid:
                        max_eid=int(segs[2])
            if nest_list!=[]:
                if len(nest_list)==1:
                    entity_list.append(nest_list[0])

                else:
                    tem=find_max_entity(nest_list)#find max entity
                    # if len(tem)>1:
                    #     print('max nest >1:',tem)
                    entity_list.extend(tem)
        fout.write(lines[0]+'\n'+lines[1]+'\n')
        for ele in entity_list:
            fout.write('\t'.join(ele)+'\n')
        fout.write('\n')
    # print(total_entity,over_entity, nest_entity)
    return fout
def find_max_entity(nest_list): #longest entity
    max_len=0
    final_tem=[]
    max_index=0
    for i in range(0, len(nest_list)):
        cur_len=int(nest_list[i][2])-int(nest_list[i][1])
        if cur_len>max_len:
            max_len=cur_len
            max_index=i

    final_tem.append(nest_list[max_index])
    return final_tem    
                
# change ori pubtator format to labeled text , entity begin with " SSSS", end with 'EEEE '
def pubtator_to_labeltext(infile,entity_map):
    
    fin=io.StringIO(infile.getvalue())
    all_context=fin.read().strip().split('\n\n')
    fin.close()
    fout=io.StringIO()
    ori_label={}
    
    for doc in all_context:
        lines=doc.split('\n')
        ori_text=lines[0].split('|t|')[1]+' '+lines[1].split('|a|')[1]
        pmid=lines[0].split('|t|')[0]
        s_index=0
        e_index=0
        new_text=''
        for i in range(2,len(lines)):
            segs=lines[i].split('\t')
            ori_label[entity_map[segs[4]].lower()]=entity_map[segs[4]]

            e_index=int(segs[1])
            # new_text+=ori_text[s_index:e_index]+' ssss'+type_label.lower()+' '+ori_text[int(segs[1]):int(segs[2])]+' eeee'+type_label.lower()+' '  
            new_text+=ori_text[s_index:e_index]+' ssss'+entity_map[segs[4]].lower()+' '+ori_text[int(segs[1]):int(segs[2])]+' eeee'+entity_map[segs[4]].lower()+' '  

            s_index=int(segs[2])
            if ori_text[int(segs[1]):int(segs[2])]!=segs[3]:
                print('error(ori,label):',ori_text[int(segs[1]):int(segs[2])],segs[3])

        new_text+=ori_text[s_index:] 
        fout.write(pmid+'\t'+' '.join(new_text.strip().split())+'\n')
    return fout,ori_label

  
# labeltext to conll format (BIO), a token (including features) per line. sentences are split by '\n', or docs are split by '\n'
def labeltext_to_conll(infile,type_label,ori_label):
    
    fin=io.StringIO(infile.getvalue())
    all_context=fin.read().strip().split('\n')
    fin.close()
    fout=[]
    if type_label=='ALL':
        O_TAG='O'
    else:
        O_TAG="O-"+type_label
    
    nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'},package='None') #package='craft'
    # nlp = stanza.Pipeline(lang='en', processors='tokenize',package='craft') #package='craft'
    doc_i=0
    for doc in all_context:

        pmid=doc.split('\t')[0]
        # fout.append('#pmid:'+pmid+'\n')
        fout.append('<'+type_label+'>\t'+O_TAG+'\n')
        doc_text=doc.split('\t')[1]
        doc_text=pre_token(doc_text)
        doc_stanza = nlp(doc_text)
        doc_i+=1
        # print(doc_i)
        inentity_flag=0
        last_label=O_TAG
        
        for sent in doc_stanza.sentences:
            temp_sent=[]
            word_num=0
            for word in sent.words:
                word_num+=1
                # print(word.text)
                if word.text.strip()=='':
                    continue
                temp_sent.append(word.text)
                if word.text.startswith('ssss')==True:
                    last_label=word.text
                    inentity_flag=1
                elif word.text.startswith('eeee')==True:
                    last_label=word.text
                    inentity_flag=0                    
                else:
                    if last_label==O_TAG:
                        now_label=O_TAG
                    elif last_label.startswith('ssss')==True:
                        now_label='B-'+ori_label[last_label[4:]]
                        
                    elif last_label.startswith('B-')==True:
                        now_label='I-'+last_label[2:]
                    elif last_label.startswith('I-')==True:
                        now_label='I-'+last_label[2:]  
                    elif last_label.startswith('eeee')==True:
                        now_label=O_TAG
                        
                    fout.append(word.text+'\t'+now_label+'\n')
                    last_label=now_label
            if inentity_flag==1 : # only output error, have connate the sentence
                # print('sentence error!!!')
                # print(word.text,word_num)
                # print(temp_sent)
                pass
            else:
                fout.append('</'+type_label+'>\t'+O_TAG+'\n')
                fout.append('\n')
                fout.append('<'+type_label+'>\t'+O_TAG+'\n')
        fout.pop()
        # fout.append('\n')
    return fout
    

def pubtator_to_conll(infile,entity_map,type_label):
    
    #1.entity sort 
    input_sort=pubtator_entitysort(infile,entity_map)
    #print(input_sort.getvalue())
    
    #2. no overlap, if overlap get longest entity
    input_nonest=filter_overlap(input_sort)
    # print('......sort.....\n',input_sort.getvalue())
    
    #3. pubtator to label text
    input_labtext,ori_label=pubtator_to_labeltext(input_nonest,entity_map)
    # print('......label.....\n',input_labtext.getvalue())
    #print(label_dic)
    
    #4. label text to conll
    output = labeltext_to_conll(input_labtext,type_label,ori_label)
    # print('......output.....\n',output.getvalue())
    # fout=open(outfile,'w',encoding='utf-8')
    # fout.write(input_nonest.getvalue())
    # fout.close()
    return output

#pubtator to conll, keep doc id
def run_pubtator_conll(inpath,outpath,entity_map_file):
    fin=open(entity_map_file,'r',encoding='utf-8')
    all_in=fin.read().strip().split('\n')
    fin.close()
    
    
    for doc in all_in:
        segs=doc.split('\t')
        filename=segs[0]
        type_label=segs[1]
        print(filename)
        entity_map={}
        for i in range(2,len(segs)):
            _eles=segs[i].split(':')
            new_type=_eles[0]
            old_types=_eles[1].split('|')
            for _old in old_types:
                entity_map[_old]=new_type
        print(entity_map)
        output=pubtator_to_conll(inpath+filename,entity_map,type_label)
        
        fout=open(outpath+filename.split('.')[0]+'-'+type_label+'.conll','w',encoding='utf-8')
        fout.write(''.join(output[:]))
        fout.close()
        


        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='preprocesss, python Format_Preprocess -i input -m mapfile -o output')
    parser.add_argument('--inpath', '-i', help="input path",default='./data/AIONER/ori_pubtator/')
    parser.add_argument('--mapfile', '-m', help="the mapfile to coversion",default='./data/AIONER/list_file.txt')
    parser.add_argument('--outpath', '-o', help="output path to save the conll results",default='./data/AIONER/conll/')
    args = parser.parse_args()
    
   

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        
    run_pubtator_conll(args.inpath,args.outpath,args.mapfile)


    
    
        