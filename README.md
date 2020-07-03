# PoWER-BERT

PoWER-BERT is a novel scheme for improving BERT inference time. It is based on exploiting a new type of redundancy within the BERT model pertaining to the word-vectors. As part of the scheme, we design strategies for determining how many and which word-vectors to eliminate at each encoder.

PoWER-BERT was implemented by modifying the standard Keras codebase for BERT (https: //github.com/CyberZHG/keras-bert)

SETUP:

- Installation using requirement.txt

TRAINING:

- There are 3 steps involved in training the PoWER-BERT:
        - Fine-tuning step: Train BERT on downstream task
        - Configuration-search step: Introduce Soft-Extract layer afterthe self-attention module of each encoder layer. This step generates the retention configuration          which determines how many wrod-vectors to eliminate from each encoder layer.
        - Re-training step: Replace the Soft-Extract layers with Extract layers and retrain the previosu model by using the obtained retention configuration.

- Install the essential packages to run the PoWER-BERT code using the requirements.txt file included in the Code folder
        - pip install -r requirements.txt

- In order to train the model use the "run.sh" script. The script consists of following user defined parameters:

        - CHECKPOINT_PATH = Path to the pre-trained BERT (uncased) checkpoint Eg. "./uncased_L-12_H-768_A-12/bert_model.ckpt'

        - DATA_DIR = Directory containing the dataset files. NOTE: There must be 3 files; train.tsv, dev.tsv and test.tsv inside the DATA_DIR. Eg. '../GLUE/CoLA'
          The data loaders have been implemented for 9 GLUE datasets in the "data_parser.py" script. These datasets were downloaded directly from the GLUE Benchmark
          website (https://gluebenchmark.com/tasks)

        - VOCAB_PATH = Path to the BERT vocabulary. Eg. './uncased_L-12_H-768_A-12/vocab.txt'

        - BERT_CONFIG_PATH = Path to the configuration file containing parameters pertaining to the BERT architecture. Eg. './uncased_L-12_H-768_A-12/bert_config.json'

        - OUTPUT_DIR = An output directory will be created to store the trained model and log files. Eg. './Output'

        - TASK = Fine-tuning/Downstream TASK. Data loaders for 9 GLUE datasets have been implemented. For other tasks, corresponding data loader has to be implemented           Valid values are: cola, rte, sst-2, qnli, mrpc, mnli-m, mnli-mm, qqp, sts-b.

        - EPOCH = Number of epochs to be used for fine-tuning and re-training steps. For GLUE datasets typically 3-5 epochs are sufficient

        - HYPER-PARAMETERS:

                - BATCH_SIZE = Batch size to be used for training, validation and prediction. Ranges: {4, 8, 16 ,32, 64}

                - LR_SOFT_EXTRACT = Learning rate for Retention parameters for Soft-Extract layers. Range: [10^−4, 10^−2]

                - LR_BERT = Learning rate for rest of the parameters pertaining to the BERT model. Range: [2×10^−5, 6×10^−5]

                - LAMBDA = The regularizer parameter that controls the trade-off between Accuracy and Inference time. Range: [10^−4, 10^−3]


- The OUTPUT_DIR contains:
        - best fine-tuned checkpoint
        - best re-trained checkpoint
        - log file containing the validation accuracies and retention configuration
        - Prediction file for TEST dataset (to be submitted to the GLUE evaluation server)

NOTE: The TASK specific hyper-parameters to obtain the results in Table:2 has been listed in the Supplementary material

