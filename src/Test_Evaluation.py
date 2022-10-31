# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:32:01 2021

@author: luol2

Pubtator file evaluation, used for test file evaluation
"""
import copy
import argparse

#if overlap 
def strict_mention_metric(pre_result, gold_result):

    var_test={}
    
    strict_Metrics={} # {'type':[tp,gold_num,pre_num],...}
    relaxed_Metrics={} # {'type':[ptp,gold_num,pre_num]}

    temp_gold_result=  copy.deepcopy(gold_result)    
    for pmid in pre_result.keys():
        
        for entity_type in gold_result[pmid].keys():
            #count gold num
            if entity_type not in strict_Metrics.keys():
                strict_Metrics[entity_type]=[0,len(gold_result[pmid][entity_type]),0]
                relaxed_Metrics[entity_type]=[0,len(gold_result[pmid][entity_type]),0]
            else:
                strict_Metrics[entity_type][1]+=len(gold_result[pmid][entity_type])
                relaxed_Metrics[entity_type][1]+=len(gold_result[pmid][entity_type])
                
          
        for entity_type in pre_result[pmid].keys():
            #count pre num
            if entity_type not in strict_Metrics.keys():
                strict_Metrics[entity_type]=[0,0,len(pre_result[pmid][entity_type])]
                relaxed_Metrics[entity_type]=[0,0,len(pre_result[pmid][entity_type])]
            else:
                strict_Metrics[entity_type][2]+=len(pre_result[pmid][entity_type])
                relaxed_Metrics[entity_type][2]+=len(pre_result[pmid][entity_type])
                
                for pre_seg in pre_result[pmid][entity_type]:
                    if entity_type in gold_result[pmid].keys():
                        #strict tp
                        if pre_seg in gold_result[pmid][entity_type]:
                            strict_Metrics[entity_type][0]+=1
                        #relaxed tp: 
                        for gold_seg in temp_gold_result[pmid][entity_type]:
                            if max(int(pre_seg[0]),int(gold_seg[0])) <= min(int(pre_seg[1]),int(gold_seg[1])):
                                relaxed_Metrics[entity_type][0]+=1
                                temp_gold_result[pmid][entity_type].remove(gold_seg)
                                break
        
    print('........strict metrics.........')            
    print(strict_Metrics)
    TP,Gold_num,Pre_num=0,0,0
    for ele in strict_Metrics.keys():
        if strict_Metrics[ele][2]==0:
            p=0
        else:
            p=strict_Metrics[ele][0]/strict_Metrics[ele][2]
        if strict_Metrics[ele][1]==0:
            r=0
        else:
            r=strict_Metrics[ele][0]/strict_Metrics[ele][1]
        if p+r==0:
            f1=0
        else:
            f1=2*p*r/(p+r)
        TP+=strict_Metrics[ele][0]
        Gold_num+=strict_Metrics[ele][1]
        Pre_num+=strict_Metrics[ele][2]
        # print(ele+': P/R/F=%.5f/%.5f/%.5f' % (p,r,f1))
        print(ele+': P/R/F=%.2f/%.2f/%.2f' % (p*100,r*100,f1*100))
        
        # break
    if Pre_num==0:
        strict_P=0
    else:
        strict_P=TP/Pre_num
    strict_R=TP/Gold_num
    strict_F1=2*strict_P*strict_R/(strict_P+strict_R)
    # print("Overall: P/R/F=%.5f/%.5f/%.5f"% (strict_P,strict_R,strict_F1))
    print("Overall: P/R/F=%.2f/%.2f/%.2f"% (strict_P*100,strict_R*100,strict_F1*100))
    
    
    
    print('........relaxed metrics.........')            
    print(relaxed_Metrics)
    pTP,rTP,Gold_num,Pre_num=0,0,0,0
    for ele in relaxed_Metrics.keys():
        if relaxed_Metrics[ele][2]==0:
            p=0
        else:
            p=relaxed_Metrics[ele][0]/relaxed_Metrics[ele][2]
        if relaxed_Metrics[ele][1]==0:
            r=0
        else:
            r=relaxed_Metrics[ele][0]/relaxed_Metrics[ele][1]
        if p+r==0:
            f1=0
        else:
            f1=2*p*r/(p+r)
        pTP+=relaxed_Metrics[ele][0]
        rTP+=relaxed_Metrics[ele][0]
        Gold_num+=relaxed_Metrics[ele][1]
        Pre_num+=relaxed_Metrics[ele][2]
        # print(ele+': P/R/F=%.5f/%.5f/%.5f' % (p,r,f1))
        print(ele+': P/R/F=%.2f/%.2f/%.2f' % (p*100,r*100,f1*100))
        # break
    if Pre_num==0:
        relaxed_P=0
    else:
        relaxed_P=pTP/Pre_num
    relaxed_R=rTP/Gold_num
    relaxed_F1=2*relaxed_P*relaxed_R/(relaxed_P+relaxed_R)
    # print("Overall: P/R/F=%.5f/%.5f/%.5f"% (relaxed_P,relaxed_R,relaxed_F1))
    print("Overall: P/R/F=%.2f/%.2f/%.2f"% (relaxed_P*100,relaxed_R*100,relaxed_F1*100))

def pubtatorfile_eva(goldfile,prefile):
   
    gold_fin=open(goldfile,'r',encoding='utf-8')
    pre_fin=open(prefile,'r',encoding='utf-8')
    gold_all=gold_fin.read().strip().split('\n\n')
    gold_fin.close()
    pre_all=pre_fin.read().strip().split('\n\n')
    pre_fin.close() 
    
    gold_result={} # {pmid:{'gene':[[start1,end1],[start2,end2]...],'chemical':[[start1,end1],...]}}
    for doc in gold_all:
        temp_result={}
        lines=doc.split('\n')
        pmid=lines[0].split('|t|')[0]
        for i in range(2,len(lines)):
            seg=lines[i].split('\t')
                
            if seg[4].lower() not in temp_result.keys():
                temp_result[seg[4].lower()]=[[seg[1],seg[2]]]
            else:
                if [seg[1],seg[2]] in temp_result[seg[4].lower()]:
                    print('same entity', seg)
                temp_result[seg[4].lower()].append([seg[1],seg[2]])

                
        gold_result[pmid]=temp_result
        
    pre_result={} # {pmid:{'gene':[[start1,end1],[start2,end2]...],'chemical':[[start1,end1],...]}}
    for doc in pre_all:
        temp_result={}
        lines=doc.split('\n')
        pmid=lines[0].split('|t|')[0]
        for i in range(2,len(lines)):
            seg=lines[i].split('\t')
            if seg[4].lower() not in temp_result.keys():
                temp_result[seg[4].lower()]=[[seg[1],seg[2]]]
            else:
                temp_result[seg[4].lower()].append([seg[1],seg[2]])
        pre_result[pmid]=temp_result

    # mention_metric(pre_result, gold_result)
    strict_mention_metric(pre_result, gold_result)
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Evaluation, python Test_Evaluation.py -g goldfile -p predfile ')
    parser.add_argument('--gold', '-g', help="gold standard file",default='')
    parser.add_argument('--pred', '-p', help="prediction file",default='')
    args = parser.parse_args()
    pubtatorfile_eva(args.gold, args.pred)