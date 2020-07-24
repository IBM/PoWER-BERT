
# PoWER-BERT

**PoWER-BERT** (**P**r**o**gressive **W**ord-vector **E**limination for inference time **R**eduction of **BERT**) is a novel scheme for improving BERT inference time for sentence classification tasks.

  

## Introduction

  
PoWER-BERT is based on identifying a new type of redundancy within the BERT model pertaining to the word-vectors. As part of the scheme, we design strategies for determining how many and which word-vectors to eliminate at each transformer block.

For more information, please refer to our [ICML 2020 paper](https://proceedings.icml.cc/static/paper_files/icml/2020/6722-Paper.pdf), which describes PoWER-BERT in detail and provides full results on a number of datasets for the sentence classification task.

Here are the results on the test sets of GLUE datasets <sup>1</sup>. 



||Method|CoLA|RTE|QQP|MRPC|SST-2|MNLI-m|MNLI-mm|QNLI|STS-B|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Test Accuracy|BERT-Base|52.5| 68.1| 71.2| 88.7| 93.0| 84.6| 84.0| 91.0| 85.8| 
|Test Accuracy|POWER-BERT|52.3| 67.4| 70.2| 88.1| 92.1| 83.8| 83.1| 90.1| 85.1| 
|Inference Time (ms)|BERT-Base|898| 3993| 1833| 1798 |905 |1867 |1881 |1848| 881 |
|Inference Time (ms)|POWER-BERT|201| 1189 |405 |674 |374| 725 |908| 916 |448 |
|Speedup||(4.5x) |(3.4x) |(4.5x)| (2.7x)| (2.4x)| (2.6x)| (2.1x)| (2.0x)| (2.0x)| 


<sup>1</sup> Inference time reported  on a K80 GPU for batch size of 128. We limit the accuracy loss for PoWER-BERT to be within 1%. Matthew’s Correlation reported for CoLA; F1-score for QQP and MRPC; Spearman Correlation for STS-B; Accuracy for the rest.

## Setup

This code base has been tested with Python3.7, Tensorflow1.15 and Keras2.3.1. 
All experiments presented here can be run on a GPU that has a least 12GB of RAM (this includes K80, P100, V100). There are few specific dependencies to be installed for use of the code base, you can install them with the command
 `pip install -r requirements.txt`. 

## Sentence (and sentence-pair) classification tasks

Before running this example you must download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory `$GLUE_DIR`. Next, download the [`BERT-Base`](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) checkpoint and unzip it to some directory `$BERT_BASE_DIR`. Note that our approach is generic and can work on other  BERT models as well.


### Training of PoWER-BERT

There are three steps involved in training the PoWER-BERT. We describe each of these steps below. This example code accelerates inference of `BERT-Base` for the Microsoft Research Paraphrase Corpus (MRPC) corpus, which  contains around 3,600 examples and can be completed in a few minutes to an hour depending on the GPU type. 

#### Fine-tuning step

This step does fine-tuning of `BERT-Base` on the downstream task.

#### Configuration-search step
This step introduces *Soft-Extract* layer after the self-attention module of each Transformer block. This is essential to generate the retention configuration which determines the number of  word-vectors to be eliminated from each Transformer block.

#### Retraining step
This step replaces the *Soft-Extract* layers with *Extract* layers and retrains the model outputted from the previous step  using the obtained retention configuration.

---------
The above three steps are executed in sequence using the following commands.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export OUTPUT_DIR=/path/to/output/directory

mkdir -p $OUTPUT_DIR
python main.py --TASK "mrpc" \
--BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
--CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
--VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
--DATA_DIR $GLUE_DIR/MRPC \
--EPOCH 3 \
--BATCH_SIZE 64 \
--LR_SOFT_EXTRACT 0.003 \
--LR_BERT 0.00006 \
--LAMBDA 0.003 \
--OUTPUT_DIR $OUTPUT_DIR 
```

We describe below each of the arguments in a little more detail.

``TASK``   Fine-tuning/downstream task. Valid values are: *cola, rte, sst-2, qnli, mrpc, mnli-m, mnli-mm, qqp, sts-b*. 
``BERT_CONFIG_PATH`` Path to the configuration file containing parameters pertaining to the BERT architecture.
``CHECKPOINT_PATH``  Path to the pre-trained BERT checkpoint. <br  />
``VOCAB_PATH`` Path to the BERT vocabulary. <br  />
``CASED`` Set the flag if the BERT model is cased. <br  />
``DATA_DIR`` Directory containing the dataset files - there must be three files; *train.tsv, dev.tsv* and *test.tsv* inside this directory. <br  />
``EPOCH``  Number of epochs for fine-tuning and re-training steps (for GLUE datasets typically 3-5 epochs are sufficient). <br  />
``BATCH_SIZE``  Batch size to be used for training, validation and prediction - can  be set to one of  {4, 8, 16 ,32, 64}. <br  />
``LR_SOFT_EXTRACT``  Learning rate for Retention parameters for Soft-Extract layers -  range: [10<sup>-4</sup>, 10<sup>-2</sup>].   <br  /> 
``LR_BERT``  Learning rate for rest of the parameters pertaining to the BERT model -range: [2×10<sup>-5</sup>, 6×10<sup>-5</sup>].<br  />
``LAMBDA`` Regularizer parameter to control trade-off between accuracy and inference time -  range: [10<sup>-4</sup>, 10<sup>-3</sup>]. <br  /> 
``OUTPUT_DIR`` Output directory to store the trained model and log files. After successful completion of the script this shall contain (i) best fine-tuned checkpoint (ii) best re-trained checkpoint (iii) log file containing the validation accuracies and retention configuration..

Note: For *MNLI* dataset downloaded from GLUE, you may have to accordingly rename the test/dev files for the above script to run.

The task specific hyper-parameters for uncased English BERT used to obtain the results presented above are specified below:


|Parameter|CoLA|RTE|QQP|MRPC|SST-2|MNLI-m|MNLI-mm|QNLI|STS-B|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|LR_BERT|2e-5| 3e-5| 3e-5| 6e-5| 3e-5| 3e-5| 3e-5| 3e-5| 3e-5| 
|LR_SOFT_EXTRACT|1.5e-3| 3e-3| 1e-4| 3e-3| 5e-4| 2e-4| 1e-4| 2e-4| 3e-3| 
|LAMBDA|7e-3| 1e-3| 3e-4| 3e-3| 2e-4| 1e-4| 1e-4| 1.5e-4| 1e-3| 
|BATCH_SIZE|32| 16| 64| 64| 64| 64| 64| 16| 64| 

### Evaluation/Prediction using PoWER-BERT

Evalutation using the PoWER-BERT can be carried out with the following command.

```
python main.py --TASK "mrpc" \
--BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
--CHECKPOINT_PATH /best/re-trained/checkpoint/obtained/above \
--VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
--CASED \
--EVAL_ONLY \
--MODEL_FORMAT "HDF5" \
--RETENTION_CONFIG /retention/configuration/obtained/above \
--DATA_DIR $GLUE_DIR/MRPC \
--BATCH_SIZE 128 \
--OUTPUT_DIR $OUTPUT_DIR
```

We describe below addtional arguments used for Evaluation/Prediction.

``EVAL_ONLY/PREDICT_ONLY`` Set the flag to carry out evaluation/prediction on the dev.tsv/test.tsv file present in the ``DATA_DIR``

``MODEL_FORMAT`` Format of the model present at the ``CHECKPOINT_PATH`` to be used for evaluation/prediction. Allowed values are {``CKPT`` , ``HDF5``}. ``CKPT`` is used for the ``.ckpt`` TensorFlow model and ``HDF5`` is used for the ``.hdf5`` Keras model. 

``RETENTION_CONFIG`` For a transformer with *N* blocks, this should be specified as *N* monotonically non-increasing comma separated values - the first value is generally the *maximum sequence length* used for the task. For example, for MRPC, to use all word-vectors at each of the 12 transformer blocks (as used for ``BERT-Base``), we can specify RETENTION_CONFIG="128,128,128,128,128,128,128,128,128,128,128,128". Another feasible setting is, for example, RETENTION_CONFIG="128,128,128,128,64,64,64,64,32,32,32,32". The optimum retention configuration for the task is however obtained from the training logs as stated above.

``PRED_GLUE`` Set this flag during prediction (i.e. when ``PREDICT_ONLY`` flag is set) if the dataset belongs to the GLUE Benchmark. This flag will save the predictions to be submitted to the GLUE evaluation server in the ``OUTPUT_DIR``.
 
NOTE:  For both evalutation and prediction, ``RETENTION_CONFIG`` is a mandatory argument that needs to be provided to the above python call. If not provided then a default value equal to Sequence length shall be used for evaluation/prediction.

Note: For evaluation mode the log file containing validation loss and accuracy is written in the ``OUTPUT_DIR``.. 


## Citation

If you find the resource useful, you should cite the following paper:

  

```

@misc{goyal2020powerbert,
    title={PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination},
    author={Saurabh Goyal and Anamitra R. Choudhury and Saurabh M. Raje and Venkatesan T. Chakaravarthy and Yogish Sabharwal and Ashish Verma},
    year={2020},
    eprint={2001.08950},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}


```

