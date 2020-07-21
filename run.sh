
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export OUTPUT_DIR=/path/to/output/directory

mkdir -p $OUTPUT_DIR

python main.py --TASK "mrpc" \
--BERT_CONFIG_PATH $BERT_BASE_DIR/bert_config.json  \
--CHECKPOINT_PATH $BERT_BASE_DIR/bert_model.ckpt \
--VOCAB_PATH $BERT_BASE_DIR/vocab.txt \
--CASED \
--DATA_DIR $GLUE_DIR/MRPC \
--EPOCH 3 \
--BATCH_SIZE 64 \
--LR_SOFT_EXTRACT 0.003 \
--LR_BERT 0.00006 \
--LAMBDA 0.003 \
--OUTPUT_DIR $OUTPUT_DIR 

