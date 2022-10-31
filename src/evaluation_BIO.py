# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:33:54 2021

@author: luol2
"""
# from BIO format to entity
def BIO_tag(tokens):
    gold_entity={}
    pre_entity={}
    gold_start,gold_end=0,0
    pre_start,pre_end=0,0
    for i in range(0,len(tokens)):
        segs=tokens[i].split('\t')
        
        # generate gold entity
        if segs[1].startswith('B-')>0:
            gold_start=i
            gold_type=segs[1][2:]
            if i+1>=len(tokens): # the last word
                gold_end=i
                if gold_type in gold_entity.keys():
                    gold_entity[gold_type].append([gold_start,gold_end])
                else:
                    gold_entity[gold_type]=[[gold_start,gold_end]]
            else: # non last word
                next_seg=tokens[i+1].split('\t')
                if next_seg[1].startswith('B-')>0 or next_seg[1].startswith('O')>0:
                    gold_end=i
                    if gold_type in gold_entity.keys():
                        gold_entity[gold_type].append([gold_start,gold_end])
                    else:
                        gold_entity[gold_type]=[[gold_start,gold_end]]
                elif next_seg[1].startswith('I-')>0:
                    pass
        elif segs[1].startswith('I-')>0:
            if i+1>=len(tokens): # the last word
                gold_end=i
                if gold_type in gold_entity.keys():
                    gold_entity[gold_type].append([gold_start,gold_end])
                else:
                    gold_entity[gold_type]=[[gold_start,gold_end]]
            else: # non last word
                next_seg=tokens[i+1].split('\t')
                if next_seg[1].startswith('B-')>0 or next_seg[1].startswith('O')>0:
                    gold_end=i
                    if gold_type in gold_entity.keys():
                        gold_entity[gold_type].append([gold_start,gold_end])
                    else:
                        gold_entity[gold_type]=[[gold_start,gold_end]]
                elif next_seg[1].startswith('I-')>0:
                    pass
        elif segs[1].startswith('O')>0:
            pass
        
        # generate prediction entity            
        if segs[2].startswith('B-')>0:
            pre_start=i
            pre_type=segs[2][2:]
            if i+1>=len(tokens): # the last word
                pre_end=i
                if pre_type in pre_entity.keys():
                    pre_entity[pre_type].append([pre_start,pre_end])
                else:
                    pre_entity[pre_type]=[[pre_start,pre_end]]
            else: # non last word
                next_seg=tokens[i+1].split('\t')
                if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                    pre_end=i
                    if pre_type in pre_entity.keys():
                        pre_entity[pre_type].append([pre_start,pre_end])
                    else:
                        pre_entity[pre_type]=[[pre_start,pre_end]]
                elif next_seg[2].startswith('I-')>0:
                    pass
        elif segs[2].startswith('I-')>0:
            if i==0 and i+1<len(tokens): # the first word and not only a word
                pre_start=i
                pre_type=segs[2][2:]
                next_seg=tokens[i+1].split('\t')
                if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                    pre_end=i
                    if pre_type in pre_entity.keys():
                        pre_entity[pre_type].append([pre_start,pre_end])
                    else:
                        pre_entity[pre_type]=[[pre_start,pre_end]]
                elif next_seg[2].startswith('I-')>0:
                    pass
            elif i==0 and i+1==len(tokens):# only one word:
                pre_start=i
                pre_type=segs[2][2:]
                pre_end=i
                if pre_type in pre_entity.keys():
                    pre_entity[pre_type].append([pre_start,pre_end])
                else:
                    pre_entity[pre_type]=[[pre_start,pre_end]]
            elif i+1>=len(tokens): # the last word
                last_seg=tokens[i-1].split('\t')
                if last_seg[2].startswith('O')>0:
                    pre_start=i
                    pre_type=segs[2][2:]                
                pre_end=i
                if pre_type in pre_entity.keys():
                    pre_entity[pre_type].append([pre_start,pre_end])
                else:
                    pre_entity[pre_type]=[[pre_start,pre_end]]
            elif i+1< len(tokens): # non last word
                next_seg=tokens[i+1].split('\t')
                last_seg=tokens[i-1].split('\t')
                if last_seg[2].startswith('O')>0:
                    pre_start=i
                    pre_type=segs[2][2:]  
                if next_seg[2].startswith('B-')>0 or next_seg[2].startswith('O')>0:
                    pre_end=i
                    if pre_type in pre_entity.keys():
                        pre_entity[pre_type].append([pre_start,pre_end])
                    else:
                        pre_entity[pre_type]=[[pre_start,pre_end]]
                elif next_seg[2].startswith('I-')>0:
                    pass
        elif segs[2].startswith('O')>0:
            pass        
    # print(tokens)
    # print(gold_entity)
    # print(pre_entity)
    return gold_entity,pre_entity
        
# input: token \t Gold \t Prediction\n, sentence is split "\n"
def NER_Evaluation():
    path='//panfs/pan1/bionlp/lulab/luoling/OpenBioIE_project/models/Kfold/BiLSTM-CRF/'
    fin=open(path+'dev_pre.conll_all','r',encoding='utf-8')
    all_sentence=fin.read().strip().split('\n\n')
    fin.close()
    Metrics={} #{'entity_type':[TP,gold_num,pre_num]}

    for sentence in all_sentence:
        tokens=sentence.split('\n')
        gold_entity,pre_entity=BIO_tag(tokens)
        # print(tokens)
        for entity_type in gold_entity.keys():
            if entity_type not in Metrics.keys():
                Metrics[entity_type]=[0,len(gold_entity[entity_type]),0]
            else:
                Metrics[entity_type][1]+=len(gold_entity[entity_type])
        for entity_type in pre_entity.keys():
            if entity_type not in Metrics.keys():
                Metrics[entity_type]=[0,0,len(pre_entity[entity_type])]
            else:
                Metrics[entity_type][2]+=len(pre_entity[entity_type])
                for mention in pre_entity[entity_type]:
                    if entity_type in gold_entity.keys():
                        if mention in gold_entity[entity_type]:
                            Metrics[entity_type][0]+=1
    print(Metrics)
    TP,Gold_num,Pre_num=0,0,0
    for ele in Metrics.keys():
        if Metrics[ele][2]==0:
            p=0
        else:
            p=Metrics[ele][0]/Metrics[ele][2]
        if Metrics[ele][1]==0:
            r=0
        else:
            r=Metrics[ele][0]/Metrics[ele][1]
        if p+r==0:
            f1=0
        else:
            f1=2*p*r/(p+r)
        TP+=Metrics[ele][0]
        Gold_num+=Metrics[ele][1]
        Pre_num+=Metrics[ele][2]
        print(ele+': P=%.5f, R=%.5f, F1=%.5f' % (p,r,f1))
        # break
    if Pre_num==0:
        P=0
    else:
        P=TP/Pre_num
    R=TP/Gold_num
    F1=2*P*R/(P+R)
    print("Overall: P=%.5f, R=%.5f, F1=%.5f"% (P,R,F1))
    
def NER_Evaluation_fn(file):
    
    fin=open(file,'r',encoding='utf-8')
    all_sentence=fin.read().strip().split('\n\n')
    fin.close()
    Metrics={} #{'entity_type':[TP,gold_num,pre_num]}
    # breai=0
    for sentence in all_sentence:
        # breai+=1
        # if breai>5000:
        #     break
        tokens=sentence.split('\n')
        gold_entity,pre_entity=BIO_tag(tokens)
        # print(tokens)
        for entity_type in gold_entity.keys():
            if entity_type not in Metrics.keys():
                Metrics[entity_type]=[0,len(gold_entity[entity_type]),0]
            else:
                Metrics[entity_type][1]+=len(gold_entity[entity_type])
        for entity_type in pre_entity.keys():
            if entity_type not in Metrics.keys():
                Metrics[entity_type]=[0,0,len(pre_entity[entity_type])]
            else:
                Metrics[entity_type][2]+=len(pre_entity[entity_type])
                for mention in pre_entity[entity_type]:
                    if entity_type in gold_entity.keys():
                        if mention in gold_entity[entity_type]:
                            Metrics[entity_type][0]+=1
    print(Metrics)
    TP,Gold_num,Pre_num=0,0,0
    for ele in Metrics.keys():
        if Metrics[ele][2]==0:
            p=0
        else:
            p=Metrics[ele][0]/Metrics[ele][2]
        if Metrics[ele][1]==0:
            r=0
        else:
            r=Metrics[ele][0]/Metrics[ele][1]
        if p+r==0:
            f1=0
        else:
            f1=2*p*r/(p+r)
        TP+=Metrics[ele][0]
        Gold_num+=Metrics[ele][1]
        Pre_num+=Metrics[ele][2]
        print(ele+': P=%.5f, R=%.5f, F1=%.5f' % (p,r,f1))
        # break
    if Pre_num==0:
        P=0
    else:
        P=TP/Pre_num
    R=TP/Gold_num
    if P+R==0:
        F1=0
    else:
        F1=2*P*R/(P+R)
    print("Overall: P=%.5f, R=%.5f, F1=%.5f"% (P,R,F1))
    return F1
            
if __name__=='__main__':
    NER_Evaluation()
