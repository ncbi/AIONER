# AIONER
***
Biomedical named entity recognition (BioNER) seeks to automatically recognize biomedical entities in natural language text, serving as a necessary foundation for downstream text mining tasks and applications. In this work, we propose AIONER, a new BioNER tagger that takes full advantage of various existing datasets for recognizing multiple entities simultaneously, through a novel all-in-one (AIO) scheme. This repo contains the source code and dataset for the AIONER.


## Content
- [Dependency package](#package)
- [Introduction of folders](#intro)
- [Pre-trained model preparation](#preparation)
- [Running AIONER](#tagging)
- [Training a new AIONER model](#training)
- [Fine-tune for a new NER task](#app)
- [Format conversion](#preprocess)



## Dependency package
<a name="package"></a>
The codes have been tested using Python 3.7 on CentOS and uses the following main dependencies on a CPU and GPU:
- [TensorFlow 2.3.0](https://www.tensorflow.org/)
- [Transformer 4.18.0](https://huggingface.co/docs/transformers/installation)
- [stanza 1.4.0](stanfordnlp.github.io/stanza/)


To install all dependencies automatically using the command:

    $ pip install -r requirements.txt


## Introduction of folders
<a name="intro"></a>

- data
	- pubtator: the datasets in pubtator format
	- conll: the datasets in CoNLL format
- example: input example files, BioC and PubTator formats (abstract or full text)
- src: the codes for AIONER
	- AIONER_Run.py: the script for running AIONER
	- AIONER_Training.py: the script for training AIONER model
	- AIONER_FineTune.py: the script for fine-tuning AIONER model for the new NER task
	- Format_Preprocess.py: the preprocesing script to covert the pubtator format to conll format
- vocab: label files for NER


## Pre-trained model preparation
<a name="preparation"></a>

To run this code, you need to first download [the model file](https://ftp.ncbi.nlm.nih.gov/pub/lu/AIONER/pretrained_models.zip) ( it includes the files for two original pre-trained models and two AIONER models), then unzip and put the model folder into the AIONER folder.

- bioformer-cased-v1.0: the original bioformer model files
- BiomedNLP-PubMedBERT-base-uncased-abstract: the original PubMedBERT model files
- AIONER:
	- PubmedBERT-CRF-AIONER.h5: the PubMedBERT-CRF AIONER model
	- Bioformer-softmax-AIONER.h5: the Bioformer-softmax AIONER model



## Running AIONER
<a name="tagging"></a>
Use our trained models (i.e., PubmedBERT/Bioformer) for running AIONER by *AIONER_Run.py*.

The file has 5 parameters:

- --inpath, -i, help="input folder"
- --model, -m, help="trained AIONER model file"
- --vocabfile, -v, help="vocab file with BIO label"
- --entity, -e, help="predict entity type (Gene, Chemical, Disease, Mutation, Species, CellLine, ALL)"
- --outpath, -o, help="output folder to save the AIONER tagged results"

The input file format is [BioC(xml)](bioc.sourceforge.net) or [PubTator](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html) format. You can find some input examples in the /example/input/ folder .



Run Example:

    $ python AIONER_Run.py -i ../example/input/ -m ../pretrained_models/AIONER/Bioformer-softmax-AIONER.h5 -v ../vocab/AIO_label.vocab -e ALL -o ../example/output/




## Training a new AIONER model
<a name="training"></a>


You can train a new AIONER model using the *AIONER_Training.py* file.

The file has 5 parameters:

- --trainfile, -t, help="the training set file in CoNLL format"
- --valfile, -v, help="the validation set file in CoNLL format (optional)"
- --encoder, -e, help="the encoder of model (bioformer or pubmedbert)?"
- --decoder, -d, help="the decoder of model (crf or softmax)?"
- --outpath, -o, help="the model output folder"



Note that --valfile is an optional parameter. When the validation set is provided, the model training will early stop by the performance on the validation. If no, the model training will early stop by the accuracy of training set. 

Run Example:

    $ python AIONER_Training.py -t ../data/conll/Merged_All-AIO.conll  -e bioformer -d softmax -o ../models/

After the training is finished, the trained model (e.g., *bioformer-softmax-es-AIO.h5*) will be generated in the output folder. If the development set is provided, two trained models (*bioformer-softmax-es-AIO.h5* for early stopping by the accuracy of training set; *bioformer-softmax-best-AIO.h5* for early stopping by the performance on the validation set) will be generated in the output folder.



## Fine-tune for a new NER task
<a name="app"></a>
Use our pretrained AIONER models for fine-tuning a new NER task.

First, you need to fine-tune the model using the new training set by *AIONER_FineTune.py* file.

The file has 5 parameters:

- --trainfile, -t, help="the training set file"
- --devfile, -d, help="the development set file"
- --vocabfile, -v, help="vocab file with BIO label"
- --modeltype, -m, help="deep learning model (bioformer or pubmedbert?)"
- --outpath, -o, help="the fine-tuned model output folder"

Note that, the input file is conll format with adding <ALL></ALL> tags. You can covert the pubtator format to conll format using the *Format_Preprocess.py* file. Moreover, --devfile is an optional parameter. When the development set is provided, the model training will early stop by the performance on the development. If no, the model training will early stop by the accuracy of training set. 

Run Example:

    $ python AIONER_FineTune.py -t ../data/conll/AnEM_train.conll -v ../vocab/AnEM_label.vocab -m bioformer -o ../models/

After the training is finished, the trained model (e.g., *bioformer-softmax-es-finetune.h5*) will be generated in the output folder. If the development set is provided, two trained models (*bioformer-softmax-es-finetune.h5* for early stopping by the accuracy of training set; *bioformer-softmax-best-finetune.h5* for early stopping by the performance on the validation set) will be generated in the output folder.


Then you can use the fine-tune model for tagging by *AIONER_Run.py* file.


Run Example:

    $ python AIONER_Run.py -i ../example/input/ -m bioformer-softmax-es-finetune.h5 -v ../vocab/AnEM_label.vocab -e ALL -o example/output/


## Format conversion
<a name="preprocess"></a>


You can covert the pubtator format to conll format using the *Format_Preprocess.py* file.

The file has 3 parameters:

- --inpath, -i, help="the input folder of training set files in Pubtator format"
- --mapfile, -m, help="the mapfile to coversion"
- --outpath, -o, help="the output folder of the files in CoNLL format"



Note that --mapfile is a file to guide the entity type and label. For example, in the file of "list_file.txt":

"NCBIdisease.PubTator	Disease	Disease:DiseaseClass|SpecificDisease|CompositeMention|Modifier" denotes the input pubtator file is "NCBIdisease.PubTator", the AIONER-label is "Disease", and the entities with entity types of "DiseaseClass, SpecificDisease, CompositeMention, Modifier" are used and changed to a new entity type is "Disease".


Run Example:

    $ python Format_Preprocess.py -i ../data/pubtator/ -m ../data/list_train.txt -o ../data/conll/

After the conversion, the all conll files used for training need to merge into a file.

## Acknowledgments
This research was supported by the Intramural Research Program of the National Library of Medicine (NLM), National Institutes of Health.



## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

***
