import os
import keras
import json
import argparse
import numpy as np
import warnings
from training import training
from utils.data_parser import data_parser
from evaluation import eval
from prediction import predict, save_pred_glue
from utils.retention_parser import retention_config_parser
from model.checkpoint_loader import build_model_from_config, load_model, load_checkpoint


if __name__ == "__main__":


        parser = argparse.ArgumentParser(description='Process Command-line Inputs')
        parser.add_argument('--EPOCHS', 
            type=int, 
            default=5, 
            help='Number of epochs for training. default=5')
        parser.add_argument('--BATCH_SIZE',
            type=int,
            default=32,
            help='Batch Size for training. default=32')
        parser.add_argument('--LR_BERT',
            type=float,
            default=0.00003,
            help='Learning Rate for parameters from the BERT model')
        parser.add_argument('--LR_SOFT_EXTRACT',
            type=float, 
            help='Learning Rate for retention parameters from the new Soft-Extract Layer')
        parser.add_argument('--CHECKPOINT_PATH', 
            type=str, 
            help='Checkpoint path for training')
        parser.add_argument('--DATA_DIR', 
            type=str, 
            help='Dataset Directory containing train.tsv, dev.tsv and test.tsv files')
        parser.add_argument('--LAMBDA', 
            type=float, 
            help='Regularizer parameter Lambda that controls the trade-off between inference time and accuracy. \
                  Higher the Lambda, higher is the word-vector elimination.')
        parser.add_argument('--TASK', 
            type=str,
            help='TASK name from one of the GLUE Benchmark Dataset. Accepted values: cola, rte, qqp, mrpc, sts-b, sst-2, mnli-m, mnli-mm, qnli. \
                  For any other datase define a dataloader in the data_loader.py file')
        parser.add_argument('--VOCAB_PATH',
            type=str,
            help='BERT vocabulary path.')
        parser.add_argument('--CASED',
            action='store_true',
            help='Set if the model is cased.')
        parser.add_argument('--BERT_CONFIG_PATH',
            type=str,
            help='BERT configuration path.')
        parser.add_argument('--OUTPUT_DIR',
            type=str, 
            help='Output directory path for saving checkpoints and logging files.')
        parser.add_argument('--TASK_CONFIG_PATH', 
            type=str, 
            default='./task_config.json', 
            help='Configuration file containing task specific data like sequence length and number of classes.')
        parser.add_argument('--EVAL_ONLY',
            action='store_true',
            help='Evaluation on the dataset using the model in the CHECKPOINT_PATH. \
                  NOTE: if retention config not provided then default vlaues equal to the Sequence length will be used for all the layers')
        parser.add_argument('--PREDICT_ONLY',
            action='store_true',
            help='Prediction on the dataset using the model in the CHECKPOINT_PATH. \
                  NOTE: if retention config not provided then default vlaues equal to the Sequence length will be used for all the layers')
        parser.add_argument('--RETENTION_CONFIG',
            type=str,
            help='String containing comma separated values signifying number of word-vectors to be retained at each layer. \
                  This is used when EVAL_ONLY or PREDICT_ONLY flags are set.')
        parser.add_argument('--MODEL_FORMAT',
            type=str,
            help='Allowed values are (CKPT, HDF5). Checkpoint model should be one of these two formats. CKPT for tensorflow model \
                 and HDF5 for keras model.')
        parser.add_argument('--PRED_GLUE',
            action='store_true',
            help='Set this flag if test predictions are to be saved for GLUE Benchmark datasets to be submitted for online evaluation at \
                  https://gluebenchmark.com/')
        

        ## Parse arguments from Command-Line
        args = parser.parse_args()        

        ## Load TASK specific sequence length and number of classes.
        with open(args.TASK_CONFIG_PATH, 'r') as reader:
                task_config = json.loads(reader.read())
        seq_len = task_config[args.TASK]['seq_len']
        num_classes = task_config[args.TASK]['num_classes']

        ## Create data parser to load Train/Dev/Test data
        dataset_parser = data_parser(VOCAB_PATH=args.VOCAB_PATH, 
                                     TASK=args.TASK, 
                                     SEQ_LEN=seq_len,
                                     DATA_DIR=args.DATA_DIR,
                                     CASED=False if args.CASED is None else True)
        
        ## Setup logging file to write the results
        LOGFILE_PATH = os.path.join(args.OUTPUT_DIR, "log_file.txt")
        with open(LOGFILE_PATH, 'w') as fp:
                fp.write("Command-Line arguments :" + str(args))


        if not args.EVAL_ONLY and not args.PREDICT_ONLY:

            
            train_x, train_y = dataset_parser.get_train_data()
            dev_x, dev_y = dataset_parser.get_dev_data()

            ## Create training object 
            PoWER_BERT = training(BERT_CONFIG_PATH=args.BERT_CONFIG_PATH, 
                                  CHECKPOINT_PATH=args.CHECKPOINT_PATH,
                                  NUM_CLASSES=num_classes,
                                  SEQ_LEN=seq_len,
                                  EPOCHS=args.EPOCHS,
                                  BATCH_SIZE=args.BATCH_SIZE,
                                  TASK=args.TASK,
                                  OUTPUT_DIR=args.OUTPUT_DIR,
                                  train_data=[train_x, train_y],
                                  dev_data=[dev_x, dev_y],
                                  LOGFILE_PATH=LOGFILE_PATH)

            ## Fine-tune the pre-trained BERT model on the downstream task
            fine_tuned_model_path = PoWER_BERT.fine_tuning_step(LR_BERT=args.LR_BERT)
            
            ## Introduce the Soft-Extract layer in the fine-tuned_model and obtain the retention configuration 
            configuration_search_model_path, retention_configuration = PoWER_BERT.configuration_search_step(fine_tuned_model_path, 
                                                                                                            LAMBDA=args.LAMBDA, 
                                                                                                            LR_BERT=args.LR_BERT, 
                                                                                                            LR_SOFT_EXTRACT=args.LR_SOFT_EXTRACT) 

            ## Replace the Soft-Extract layer by Extract layer and use the retention configuration obtained in the previous step to get the final model with word-vectors eliminated
            PoWER_BERT.retraining_step(configuration_search_model_path, 
                                       retention_configuration, 
                                       LR_BERT=args.LR_BERT,) 

        elif args.EVAL_ONLY:

            ## Do the evaluation on the Dev data
            dev_x, dev_y = dataset_parser.get_dev_data()

            ## Obtain number of layers from the config file
            with open(args.BERT_CONFIG_PATH, 'r') as bc:
                bert_config = json.loads(bc.read())
                num_layers = bert_config['num_hidden_layers']
   
            loss, accuracy = eval(args, dev_x, dev_y, num_layers, num_classes, seq_len)

            with open(LOGFILE_PATH, 'a') as fp:
                fp.write("\nloss : "+str(loss))
                fp.write("\naccuracy : "+str(accuracy*100.0))

        elif args.PREDICT_ONLY:

            ## Do the prediction on the test data
            test_x, test_y = dataset_parser.get_test_data()

            ## Obtain number of layers from the config file
            with open(args.BERT_CONFIG_PATH, 'r') as bc:
                bert_config = json.loads(bc.read())
                num_layers = bert_config['num_hidden_layers']

            pred = predict(args, test_x, num_layers, num_classes, seq_len)

            with open(os.path.join(args.OUTPUT_DIR, "prediction.txt"), 'w') as fp:
                for p in pred:
                    fp.write(str(p))
                    fp.write("\n")

            if args.PRED_GLUE:
                save_pred_glue(pred, test_x, test_y, args.TASK, args.OUTPUT_DIR)


