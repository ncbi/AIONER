# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:40:08 2021

@author: luol2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:19:02 2020

@author: luol2
"""

import io
import sys

# from BIO format to entity,list line is sentence, follwing the entity(start, end, text, entity, type)
def NN_BIO_tag_entity(pre_BIO):
    sentences=pre_BIO.strip().split('\n\n')

    pre_result=[]
    for sent in sentences:
        tokens=sent.split('\n')
        pre_entity=[]
        pre_start,pre_end=0,0
        sent_text=''
        for i in range(1,len(tokens)-1):
            segs=tokens[i].split('\t')
            sent_text+=segs[0]+' '
            # generate prediction entity            
            if segs[2].startswith('B-')>0:
                pre_start=i
                pre_type=segs[2][2:]
                if i+1>=len(tokens)-1: # the last word
                    pre_end=i
                    pre_entity.append([pre_start-1,pre_end-1,pre_type])
                else: # non last word
                    next_seg=tokens[i+1].split('\t')
                    if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                        pre_end=i
                        pre_entity.append([pre_start-1,pre_end-1,pre_type])
                    elif next_seg[2].startswith('I-')>0:
                        pass
            elif segs[2].startswith('I-')>0:
                if i==1 and i+1<len(tokens)-1: # the first word and not only a word
                    pre_start=i
                    pre_type=segs[2][2:]
                    next_seg=tokens[i+1].split('\t')
                    if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                        pre_end=i
                        pre_entity.append([pre_start-1,pre_end-1,pre_type])
                    elif next_seg[2].startswith('I-')>0:
                        pass
                elif i==1 and i+1==len(tokens)-1:# only one word:
                    pre_start=i
                    pre_type=segs[2][2:]
                    pre_end=i
                    pre_entity.append([pre_start-1,pre_end-1,pre_type])
                elif i+1>=len(tokens)-1: # the last word
                    last_seg=tokens[i-1].split('\t')
                    if last_seg[2].startswith('O')>0:
                        pre_start=i
                        pre_type=segs[2][2:]                
                    pre_end=i
                    pre_entity.append([pre_start-1,pre_end-1,pre_type])
                elif i+1< len(tokens)-1: # non last word
                    next_seg=tokens[i+1].split('\t')
                    last_seg=tokens[i-1].split('\t')
                    if last_seg[2].startswith('O')>0:
                        pre_start=i
                        pre_type=segs[2][2:]  
                    if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                        pre_end=i
                        pre_entity.append([pre_start-1,pre_end-1,pre_type])
                    elif next_seg[2].startswith('I-')>0:
                        pass
            elif segs[2].startswith('O')>0:
                pass        
        pre_result.append([sent_text.rstrip(),pre_entity])


        # print(pre_entity)
    return pre_result

def NN_restore_index_fn(ori_text,file_pre):

    input_result=NN_BIO_tag_entity(file_pre)
    #print(input_result)
    
    
    new_sentence=''
    restore_result=[]
    
    sentence_ori=ori_text.lower()

    for sent_ele in input_result:

        #print(pre_lines)
#        print(sentence_ori)
        if len(sent_ele[1])>0:
            #print(pre_lines)
            sentence_pre=sent_ele[0].lower()
            sentence_pre=sentence_pre.split()
            
            pre_result=sent_ele[1]

            
            restore_sid=0
            restore_eid=0
            each_word_id=[]
            
            for i in range(0,len(sentence_pre)):

                temp_id=sentence_ori.find(sentence_pre[i])
                if temp_id<0:
                        #print('ori:',sentence_ori)
                        print('resotr index error:',sentence_pre[i])
                new_sentence+=sentence_ori[0:temp_id]
                
                restore_sid=len(new_sentence)
                restore_eid=len(new_sentence)+len(sentence_pre[i])
                each_word_id.append([str(restore_sid),str(restore_eid)])
                new_sentence+=sentence_ori[temp_id:temp_id+len(sentence_pre[i])]
                sentence_ori=sentence_ori[temp_id+len(sentence_pre[i]):]
#            print('each_word:',each_word_id)    
            for pre_ele in pre_result:
                temp_pre_result=[each_word_id[int(pre_ele[0])][0],each_word_id[int(pre_ele[1])][1],pre_ele[2]]
                if temp_pre_result not in restore_result:
                    restore_result.append(temp_pre_result)
        else:
            sentence_pre=sent_ele[0].lower()
            sentence_pre=sentence_pre.split()
           
            for i in range(0,len(sentence_pre)):

                temp_id=sentence_ori.find(sentence_pre[i])
                if temp_id<0:
                    print('resotr index error:',sentence_pre[i])
                new_sentence+=sentence_ori[0:temp_id]
                new_sentence+=sentence_ori[temp_id:temp_id+len(sentence_pre[i])]
                sentence_ori=sentence_ori[temp_id+len(sentence_pre[i]):]
    #print('resotre:',restore_result)
    return restore_result

