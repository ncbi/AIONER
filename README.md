# AIONER
***
Biomedical named entity recognition (BioNER) is a critical task of text mining research in the biomedical field. The task aims to automatically recognize the entities of the specific biomedical concepts in the text. In this work, we present AIONER, a deep learning-based NER method that successfully integrates multiple manually curated datasets and recognizes multiple biomedical concepts at once by a novel all-in-one (AIO) scheme..


## Content
- [Dependency package](#package)
- [Introduction of folders](#intro)
- [Running Gene NER and Species Assignment](#pipeline)
- [Training Gene NER](#GeneNER)
- [Training Species Assignment](#SpeAss)



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
	- Train_tmvar1-AIO.conll: all data (tmvar1 version) for AIONER training in CoNLL format.
	- Train_tmvar3-AIO.conll: all data (tmvar3 version) for AIONER training in CoNLL format
- example: input example files, BioC and PubTator formats (abstract or full text)
- src_python
	- AIONER: the codes for AIONER
- AIONER_Run.py: the script for running AIONER
- AIONER_Training.py: the script for training AIONER model
- AIONER_trained_models: pre-trianed models and trained NER models
	- bioformer-cased-v1.0: the original bioformer model
	- BiomedNLP-PubMedBERT-base-uncased-abstract: the original pubmedbert model
	- AIONER
		- Bioformer-Softmax-BEST-AIO_tmvar1(or 3).h5: the bioformer-softmax models in tmvar1(or 3) version
		- PubmedBERT-CRF-BEST-AIO_tmvar1(or 3).h5: the pubmedbert-crf models in tmvar1(or 3) version
- vocab: label files for NER

## Running AIONER
<a name="pipeline"></a>
Use our trained models (i.e., PubmedBERT/Bioformer) for running AIONER by *AIONER_Run.py*.

The file has 4 parameters:

- --inpath, -i, help="input folder"
- --model, -m, help="trained AIONER model file"
- --entity, -e, help="predict entity type (Gene, Chemical, Disease, Mutation, Species, CellLine, ALL)"
- --outpath, -o, help="output folder to save the AIONER tagged results"

The input file format is [BioC(xml)](bioc.sourceforge.net) or [PubTator](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html) format. You can find some input examples in the /example/input/ folder .



Run Example:

    $ python AIONER_Run.py -i example/input/ -m AIONER_trained_models/AIONER/Bioformer-Softmax-BEST-AIO_tmvar1.h5 -e ALL -o example/output/




## Training a new AIONER model
<a name="GeneNER"></a>


You can train a new AIONER model using the */AIONER_Training.py** file.

The file has 5 parameters:

- --trainfile, -t, help="the training set file in CoNLL format"
- --valfile, -v, help="the validation set file in CoNLL format (optional)"
- --encoder, -e, help="the encoder of model (bioformer or pubmedbert)?"
- --decoder, -d, help="the decoder of model (crf or softmax)?"
- --outpath, -o, help="the model output folder"



Note that --valfile is an optional parameter. When the validation set is provided, the model training will early stop by the performance on the validation. If no, the model training will early stop by the accuracy of training set. 

Run Example:

    $ python NER_Training.py -t ./data/AIONER/Train_tmvar1-AIO.conll -v ./data/AIONER/BioRED_Test_refined-ALL.conll -e bioformer -d softmax -o ./models/

After the training is finished, the trained model (e.g., *bioformer-softmax-es-AIO.h5*) will be generated in the output folder. If the development set is provided, two trained models (*bioformer-softmax-es-AIO.h5* for early stopping by the accuracy of training set; *bioformer-softmax-best-AIO.h5* for early stopping by the performance on the validation set) will be generated in the output folder.




## Acknowledgments
This research was supported by the Intramural Research Program of the National Library of Medicine (NLM), National Institutes of Health.