
from utils.retention_parser import retention_config_parser
from utils.mean_squared_error import metric_cor
from checkpoint_loader import build_model_from_config, load_model, load_checkpoint
from Adam_mult import AdamWarmup, calc_train_steps



def eval(args, dev_x, dev_y, num_layers, num_classes, seq_len):

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

    return model.evaluate(dev_x, 
                          dev_y, 
                          verbose=1, 
                          batch_size=args.BATCH_SIZE)





