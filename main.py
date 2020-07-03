import os
import keras
import json
import argparse
import numpy as np
from training import training
from data_parser import data_parser
from keras.callbacks import CSVLogger


def test_data_prediction(model, test_data=None, TASK=None, BATCH_SIZE=64, OUTPUT_DIR=None):
        
        if TASK == 'mrpc' or TASK =='cola' or TASK =='sst-2' or TASK =='qqp':
                data_labels = {'0':'0', '1':'1'}
        elif TASK == 'mnli-m' or TASK == 'mnli-mm':
                data_labels = {'0':'contradiction', '1':'entailment', '2':'neutral'}
        elif TASK =='qnli' or TASK =='rte':
                data_labels = {'0':'not_entailment', '1':'entailment'}
        else:
                raise ValueError('No such TASK available.')
        
        ## Generate predictions for the test dataset
        predict = model.predict(test_data[0], batch_size=BATCH_SIZE, verbose=0)

        ## Write the predictions in file
        TEST_PRED_FILE = os.path.join(OUTPUT_DIR, "test_predictions.txt")

        if TASK == "sts-b":
                with open(TEST_PRED_FILE, 'w') as test_fh:
                        test_fh.write("identity\tlabel\n")
                        for idx in range(test_data[1].shape[0]):
                                test_fh.write("%s\t%s\n" % (test_data[1][idx], predict[idx][0]))
        else:
                with open(TEST_PRED_FILE, 'w') as test_fh:
                        test_fh.write("index\tprediction\n")
                        for idx in range(test_data[1].shape[0]):
                                test_fh.write("%s\t%s\n" % (test_data[1][idx], data_labels[str(np.argmax(predict[idx]))]))



if __name__ == "__main__":


        parser = argparse.ArgumentParser(description='Process Command-line Inputs')
        parser.add_argument('--EPOCHS', type=int, default=5, help='Number of epochs for training. default=5')
        parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch Size for training. default=32')
        parser.add_argument('--LR_BERT', type=float, default=0.00003, help='Learning Rate for parameters from the BERT model')
        parser.add_argument('--LR_SOFT_EXTRACT', type=float, help='Learning Rate for retention parameters from the new Soft-Extract Layer')
        parser.add_argument('--CHECKPOINT_PATH', type=str, help='Checkpoint path for training')
        parser.add_argument('--DATA_DIR', type=str, help='Dataset Directory containing train.tsv, dev.tsv and test.tsv files')
        parser.add_argument('--LAMBDA', type=float, help='Regularizer parameter Lambda that controls the trade-off between inference time and accuracy. Higher the Lambda, higher is the word-vector elimination.')
        parser.add_argument('--TASK', type=str, help='TASK name from one of the GLUE Benchmark Dataset. Accepted values: cola, rte, qqp, mrpc, sts-b, sst-2, mnli-m, mnli-mm, qnli. For any other datase define a dataloader in the data_loader.py file')
        parser.add_argument('--VOCAB_PATH', type=str, help='BERT vocabulary path.')
        parser.add_argument('--BERT_CONFIG_PATH', type=str, help='BERT configuration path.')
        parser.add_argument('--OUTPUT_DIR', type=str, help='Output directory path for saving checkpoints and logging files.')
        parser.add_argument('--TASK_CONFIG_PATH', type=str, default='./task_config.json', help='Configuration file containing task specific data like sequence length and number of classes.')


        ## Parse arguments from Command-Line
        args = parser.parse_args()
        

        ## Load TASK specific sequence length and number of classes.
        with open(args.TASK_CONFIG_PATH, 'r') as reader:
                task_config = json.loads(reader.read())
        SEQ_LEN = task_config[args.TASK]['seq_len']
        NUM_CLASSES = task_config[args.TASK]['num_classes']


        ## Load Train, Dev and Test data
        dataset_parser = data_parser(VOCAB_PATH=args.VOCAB_PATH, 
                                     TASK=args.TASK, 
                                     SEQ_LEN=SEQ_LEN,
                                     DATA_DIR=args.DATA_DIR)
        train_x, train_y = dataset_parser.get_train_data()
        #train_x, train_y = dataset_parser.get_dev_data()
        dev_x, dev_y = dataset_parser.get_dev_data()
        test_x, test_y = dataset_parser.get_test_data()

        
        ## Setup logging file to write the results
        LOGFILE_PATH = os.path.join(args.OUTPUT_DIR, "log_file.txt")
        with open(LOGFILE_PATH, 'w') as fp:
                fp.write("Command-Line arguments :" + str(args))


        ## Create training object 
        PoWER_BERT = training(BERT_CONFIG_PATH=args.BERT_CONFIG_PATH, 
                              CHECKPOINT_PATH=args.CHECKPOINT_PATH,
                              NUM_CLASSES=NUM_CLASSES,
                              SEQ_LEN=SEQ_LEN,
                              EPOCHS=args.EPOCHS,
                              BATCH_SIZE=args.BATCH_SIZE,
                              TASK=args.TASK,
                              OUTPUT_DIR=args.OUTPUT_DIR,
                              train_data=[train_x, train_y],
                              dev_data=[dev_x, dev_y],
                              LOGFILE_PATH=LOGFILE_PATH)

        ## Fine-tune the pre-trained BERT model on the downstream task
        fine_tuned_model_path = PoWER_BERT.fine_tuning_step(LR_BERT=args.LR_BERT)

        
        ## Introduce the Soft-Extract layer in the "fine_tuned_model" and obtain the retention configuration 
        configuration_search_model_path, retention_configuration = PoWER_BERT.configuration_search_step(fine_tuned_model_path, 
                                                                                                LAMBDA=args.LAMBDA, 
                                                                                                LR_BERT=args.LR_BERT, 
                                                                                                LR_SOFT_EXTRACT=args.LR_SOFT_EXTRACT,) 

        ## Replace the Soft-Extract layer by Extract layer and use the retention configuration obtained in the previous step to get the final model with word-vectors 
        ## eliminated
        retrained_model = PoWER_BERT.retraining_step(configuration_search_model_path, 
                                                     retention_configuration, 
                                                     LR_BERT=args.LR_BERT,) 


        ## Generate prediction on GLUE Test dataset and save them in test_predictions.txt file in the Output folder for submission to the GLUE evaluation server
        test_data_prediction(retrained_model, 
                             test_data=[test_x, test_y], 
                             TASK=args.TASK, 
                             BATCH_SIZE=args.BATCH_SIZE, 
                             OUTPUT_DIR=args.OUTPUT_DIR)




