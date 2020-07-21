import os
import numpy as np
from utils.mean_squared_error import metric_cor
from utils.retention_parser import retention_config_parser
from checkpoint_loader import build_model_from_config, load_model, load_checkpoint
from Adam_mult import AdamWarmup, calc_train_steps



def save_pred_glue(predict, test_x, test_y, TASK=None, OUTPUT_DIR=None):

    if TASK == 'mrpc' or TASK == 'cola' or TASK == 'sst-2' or TASK == 'qqp':
        data_labels = {'0':'0', '1':'1'}
    elif TASK == 'mnli-m' or TASK == 'mnli-mm':
        data_labels = {'0':'contradiction', '1':'entailment', '2':'neutral'}
    elif TASK == 'qnli' or TASK == 'rte':
        data_labels = {'0':'not_entailment', '1':'entailment'}
    elif TASK == 'sts-b':
        data_labels = {}
    else:
        raise ValueError('No such TASK available.')

    ## Write the predictions in file
    TEST_PRED_FILE = os.path.join(OUTPUT_DIR, "glue_prediction.txt")

    if TASK == "sts-b":
        with open(TEST_PRED_FILE, 'w') as test_fh:
            test_fh.write("identity\tlabel\n")
            for idx in range(test_y.shape[0]):
                test_fh.write("%s\t%s\n" % (test_y[idx], predict[idx][0]))
    else:
        with open(TEST_PRED_FILE, 'w') as test_fh:
            test_fh.write("index\tprediction\n")
            for idx in range(test_y.shape[0]):
                test_fh.write("%s\t%s\n" % (test_y[idx], data_labels[str(np.argmax(predict[idx]))]))


def predict(args, test_x, num_layers, num_classes, seq_len):

    ## Parse the retention configuration if provided else use default values equal to the sequence length
    if args.RETENTION_CONFIG == None:
        retention_config = [seq_len] * num_layers
        warnings.warn("Retention Config not provided. Evaluation will be performed on default config : "+','.join(retention_config))
    else:
        retention_config = retention_config_parser(args.RETENTION_CONFIG, 
                                                   num_layers, 
                                                   seq_len)

    ## Model definition and Evaluation
    model, config = build_model_from_config(
                                args.BERT_CONFIG_PATH,
                                output_dim=num_classes,
                                seq_len=seq_len,
                                retention_configuration=retention_config,
                                FLAG_EXTRACT_LAYER=2,
                                TASK=args.TASK)

    if args.TASK == "sts-b":
        loss='mean_squared_error'
        metrics=[metric_cor, 'mae']
    else:
        loss='sparse_categorical_crossentropy'
        metrics=['accuracy']

    model.compile(
            AdamWarmup(decay_steps=1,
                        warmup_steps=1,
                        lr=0.00002,
                        lr_mult=None),
            loss=loss,
            metrics=metrics,
    )
    if args.MODEL_FORMAT == "CKPT":
        load_checkpoint(model,
                        config,
                        args.CHECKPOINT_PATH
        )
    elif args.MODEL_FORMAT == "HDF5":
        model.load_weights(args.CHECKPOINT_PATH, by_name=True)
    else:
        print ("Model format not supported")
        exit(-1)

    predict = model.predict(test_x, 
                            batch_size=args.BATCH_SIZE, 
                            verbose=0)

    return predict 


