
#source activate /u/saurago/anaconda3/envs/tf
source activate /dccstor/smraje/tf1_15/anaconda3/envs/tf

CHECKPOINT_PATH='../anbang_sentiment_analysis/bert/data/uncased_L-12_H-768_A-12/bert_model.ckpt'
DATA_DIR='../glue_data1/RTE'
VOCAB_PATH='../anbang_sentiment_analysis/bert/data/uncased_L-12_H-768_A-12/vocab.txt'
BERT_CONFIG_PATH='../anbang_sentiment_analysis/bert/data/uncased_L-12_H-768_A-12/bert_config.json'
OUTPUT_DIR='./Output'

mkdir $OUTPUT_DIR

python main.py  --TASK "rte" \
	        --EPOCH 3 \
		--BATCH_SIZE 16 \
		--LR_SOFT_EXTRACT 0.0015 \
		--LR_BERT 0.00002 \
		--LAMBDA 0.007 \
		--CHECKPOINT_PATH $CHECKPOINT_PATH \
		--DATA_DIR $DATA_DIR \
		--VOCAB_PATH $VOCAB_PATH \
		--BERT_CONFIG_PATH $BERT_CONFIG_PATH \
		--OUTPUT_DIR $OUTPUT_DIR


