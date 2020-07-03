import os
#import sys
import keras
import codecs
import csv
#import json
#from keras_bert import get_base_dict, get_model, gen_batch_inputs
#import csv
import numpy as np
from keras_bert import Tokenizer
import unicodedata
import six
#from keras import backend as K



class data_parser:

        def __init__(self, 
                     VOCAB_PATH=None,
                     TASK=None,
                     SEQ_LEN=None,
                     DATA_DIR=None):
                
                self.TASK = TASK
                self.SEQ_LEN = SEQ_LEN
                self.DATA_DIR = DATA_DIR

                self.token_dict = {}
                with codecs.open(VOCAB_PATH, 'r', 'utf8') as reader:
                    for line in reader:
                        token = line.strip()
                        self.token_dict[token] = len(self.token_dict)


                self.tokenizer = Tokenizer(self.token_dict, cased=False)


        def _read_tsv(self, input_file, quotechar=None):

            """Reads a tab separated value file."""
            with open(input_file,"rU")as f:
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                lines = []
                for line in reader:
                   lines.append(line)
                return lines


        def convert_to_unicode(self, text):

          """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
          if six.PY3:
            if isinstance(text, str):
              return text
            elif isinstance(text, bytes):
              return text.decode("utf-8", "ignore")
            else:
              raise ValueError("Unsupported string type: %s" % (type(text)))
          elif six.PY2:
            if isinstance(text, str):
              return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
              return text
            else:
              raise ValueError("Unsupported string type: %s" % (type(text)))
          else:
            raise ValueError("Not running on Python2 or Python 3?")


        def encode(self, first, second=None, max_len=None):

                first_tokens = self.tokenizer._tokenize(first)
                second_tokens = self.tokenizer._tokenize(second) if second is not None else None
                self.tokenizer._truncate(first_tokens, second_tokens, max_len)
                tokens, first_len, second_len = self.tokenizer._pack(first_tokens, second_tokens)
                token_ids = self.tokenizer._convert_tokens_to_ids(tokens)
                segment_ids = [0] * first_len + [1] * second_len
                token_len = first_len + second_len
                pad_len = 0
                if max_len is not None:
                    pad_len = max_len - first_len - second_len
                    token_ids += [self.tokenizer._pad_index] * pad_len
                    segment_ids += [0] * pad_len
                input_mask = [1]*token_len+[0]*pad_len
                return token_ids, segment_ids, input_mask


        def get_train_data(self):

                data_path = os.path.join(self.DATA_DIR, "train.tsv")
                train_x, train_y = self.load_data(data_path, set_type='train')
                return train_x, train_y

        def get_dev_data(self):

                data_path = os.path.join(self.DATA_DIR, "dev.tsv")
                dev_x, dev_y = self.load_data(data_path, set_type='dev')
                return dev_x, dev_y

        def get_test_data(self):

                data_path = os.path.join(self.DATA_DIR, "test.tsv")
                test_x, test_y = self.load_data(data_path, set_type='test')
                return test_x, test_y


        def load_data(self, data_path, set_type=None):
                
                if self.TASK == 'qqp':
                        data_x, data_y = self.load_data_qqp(data_path, set_type=set_type)
                elif self.TASK == 'sst-2':
                        data_x, data_y = self.load_data_sst(data_path, set_type=set_type)
                elif self.TASK == 'qnli':
                        data_x, data_y = self.load_data_qnli(data_path, set_type=set_type)
                elif self.TASK == 'cola':
                        data_x, data_y = self.load_data_cola(data_path, set_type=set_type)
                elif self.TASK == 'rte':
                        data_x, data_y = self.load_data_rte(data_path, set_type=set_type)
                elif self.TASK == 'mrpc':
                        data_x, data_y = self.load_data_mrpc(data_path, set_type=set_type)
                elif self.TASK == 'mnli-m' or self.TASK == 'mnli-mm':
                        data_x, data_y = self.load_data_mnli(data_path, set_type=set_type)
                elif self.TASK == 'sts-b':
                        data_x, data_y = self.load_data_stsb(data_path, set_type=set_type)
                else:
                        raise ValueError('No data loader for the given TASK.')

                return data_x, data_y



        def load_data_qqp(self, path, set_type='train'):

                indices, sentiments, masks, final_segments = [], [], [], []
                lines = self._read_tsv(path)
                for (i, line) in enumerate(lines):
                        if i == 0:
                            continue
                        if (set_type == 'train' or set_type == "dev") and len(line) < 6:
                                continue
                        if set_type == "test":
                            text_a = self.convert_to_unicode(line[1])
                            text_b = self.convert_to_unicode(line[2])
                            label = self.convert_to_unicode(line[0])
                        else:
                            text_a = self.convert_to_unicode(line[3])
                            text_b = self.convert_to_unicode(line[4])
                            label = self.convert_to_unicode(line[5])
                        ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                        indices.append(ids)
                        final_segments.append(segments)
                        sentiments.append(label)
                        masks.append(mask)
                items = list(zip(indices, masks, final_segments, sentiments))
                if set_type != "test":
                        np.random.shuffle(items)
                indices, masks, final_segments, sentiments = zip(*items)
                indices = np.array(indices)
                masks = np.array(masks)
                final_segments = np.array(final_segments)
                sentiments = np.array(sentiments)
                return [indices, final_segments, masks], sentiments



        def load_data_sst(self, path, set_type='train'):

                indices, sentiments, masks = [], [], []
                lines = self._read_tsv(path)
                for (i, line) in enumerate(lines):
                        if i == 0:
                            continue
                        if (set_type == 'train' or set_type == 'dev') and len(line) < 2:
                                continue
                        if set_type == "test":
                            text_a = self.convert_to_unicode(line[1])
                            label = self.convert_to_unicode(line[0])
                        else:
                            text_a = self.convert_to_unicode(line[0])
                            label = self.convert_to_unicode(line[1])
                        ids, segments, mask = self.encode(text_a, max_len=self.SEQ_LEN)
                        indices.append(ids)
                        sentiments.append(label)
                        masks.append(mask)
                items = list(zip(indices, masks, sentiments))
                if set_type != "test":
                        np.random.shuffle(items)
                indices, masks, sentiments = zip(*items)
                indices = np.array(indices)
                masks = np.array(masks)
                return [indices, np.zeros_like(indices), masks], np.array(sentiments)


        def load_data_mrpc(self, path, set_type='train'):

            indices, sentiments, masks, final_segments = [], [], [], []
            lines = self._read_tsv(path)
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if (set_type == 'train' or set_type == 'dev') and len(line) < 5:
                    continue
                if set_type == "test":
                    text_a = self.convert_to_unicode(line[3])
                    text_b = self.convert_to_unicode(line[4])
                    label = self.convert_to_unicode(line[0])
                else:
                    text_a = self.convert_to_unicode(line[3])
                    text_b = self.convert_to_unicode(line[4])
                    label = self.convert_to_unicode(line[0])
                ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                indices.append(ids)
                final_segments.append(segments)
                sentiments.append(label)
                masks.append(mask)
            items = list(zip(indices, masks, final_segments, sentiments))
            if set_type != "test":
                np.random.shuffle(items)
            indices, masks, final_segments, sentiments = zip(*items)
            indices = np.array(indices)
            masks = np.array(masks)
            final_segments = np.array(final_segments)
            return [indices, final_segments, masks], np.array(sentiments)


        def load_data_qnli(self, path, set_type='train'):

            indices, sentiments, masks, final_segments = [], [], [], []
            lines = self._read_tsv(path)
            data_labels = {'entailment':'1', 'not_entailment':'0'}
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if (set_type == 'train' or set_type == 'dev') and len(line) < 4:
                        continue
                if set_type == "test":
                    text_a = self.convert_to_unicode(line[1])
                    text_b = self.convert_to_unicode(line[2])
                    label = self.convert_to_unicode(line[0])
                else:
                    text_a = self.convert_to_unicode(line[1])
                    text_b = self.convert_to_unicode(line[2])
                    label = self.convert_to_unicode(data_labels[line[3]])
                ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                indices.append(ids)
                final_segments.append(segments)
                sentiments.append(label)
                masks.append(mask)
            items = list(zip(indices, masks, final_segments, sentiments))
            if set_type != "test":
                np.random.shuffle(items)
            indices, masks, final_segments, sentiments = zip(*items)
            indices = np.array(indices)
            masks = np.array(masks)
            final_segments = np.array(final_segments)
            return [indices, final_segments, masks], np.array(sentiments)



        def load_data_rte(self, path, set_type='train'):

            indices, sentiments, masks, final_segments = [], [], [], []
            lines = self._read_tsv(path)
            data_labels = {'entailment':'1', 'not_entailment':'0'}
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if (set_type == 'train' or set_type == 'dev') and len(line) < 4:
                        continue
                if set_type == "test":
                    text_a = self.convert_to_unicode(line[1])
                    text_b = self.convert_to_unicode(line[2])
                    label = self.convert_to_unicode(line[0])
                else:
                    text_a = self.convert_to_unicode(line[1])
                    text_b = self.convert_to_unicode(line[2])
                    label = self.convert_to_unicode(data_labels[line[3]])
                ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                indices.append(ids)
                final_segments.append(segments)
                sentiments.append(label)
                masks.append(mask)
            items = list(zip(indices, masks, final_segments, sentiments))
            if set_type != "test":
                np.random.shuffle(items)
            indices, masks, final_segments, sentiments = zip(*items)
            indices = np.array(indices)
            masks = np.array(masks)
            final_segments = np.array(final_segments)
            return [indices, final_segments, masks], np.array(sentiments)


        def load_data_cola(self, path, SEQ_LEN=None, set_type='train'):

                indices, sentiments, masks = [], [], []
                lines = self._read_tsv(path)
                for (i, line) in enumerate(lines):
                        if i == 0 and set_type=='test':
                                continue
                        if (set_type == 'train' or set_type == 'dev') and len(line) < 4:
                                continue
                        if set_type == "test":
                            text_a = self.convert_to_unicode(line[1])
                            label = self.convert_to_unicode(line[0])
                        else:
                            text_a = self.convert_to_unicode(line[3])
                            label = self.convert_to_unicode(line[1])
                        ids, segments, mask = self.encode(text_a, max_len=self.SEQ_LEN)
                        indices.append(ids)
                        sentiments.append(label)
                        masks.append(mask)
                items = list(zip(indices, masks, sentiments))
                if set_type != "test":
                        np.random.shuffle(items)
                indices, masks, sentiments = zip(*items)
                indices = np.array(indices)
                masks = np.array(masks)
                return [indices, np.zeros_like(indices), masks], np.array(sentiments)


        def load_data_mnli(self, path, set_type='train'):

            data_labels = {'contradiction':'0', 'entailment':'1', 'neutral':'2'}
            indices, sentiments, masks, final_segments = [], [], [], []
            lines = self._read_tsv(path)
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if set_type == "test":
                    text_a = self.convert_to_unicode(line[8])
                    text_b = self.convert_to_unicode(line[9])
                    label = self.convert_to_unicode(line[0])
                else:
                    text_a = self.convert_to_unicode(line[8])
                    text_b = self.convert_to_unicode(line[9])
                    label = self.convert_to_unicode(data_labels[line[-1]])
                ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                indices.append(ids)
                final_segments.append(segments)
                sentiments.append(label)
                masks.append(mask)
            items = list(zip(indices, masks, final_segments, sentiments))
            if set_type != "test":
                np.random.shuffle(items)
            indices, masks, final_segments, sentiments = zip(*items)
            indices = np.array(indices)
            masks = np.array(masks)
            final_segments = np.array(final_segments)
            return [indices, final_segments, masks], np.array(sentiments)



        def load_data_stsb(self, path, set_type='train'):

            indices, sentiments, masks, final_segments = [], [], [], []
            lines = self._read_tsv(path)
            for (i, line) in enumerate(lines):
                if i == 0 :
                    continue
                if set_type == "test":
                    text_a = self.convert_to_unicode(line[7])
                    text_b = self.convert_to_unicode(line[8])
                    label = self.convert_to_unicode(line[0])
                else:
                    text_a = self.convert_to_unicode(line[7])
                    text_b = self.convert_to_unicode(line[8])
                    label = float(line[-1])
                ids, segments, mask = self.encode(text_a, text_b, max_len=self.SEQ_LEN)
                indices.append(ids)
                final_segments.append(segments)
                sentiments.append(label)
                masks.append(mask)
            items = list(zip(indices, masks, final_segments, sentiments))
            if set_type != "test":
                np.random.shuffle(items)
            indices, masks, final_segments, sentiments = zip(*items)
            indices = np.array(indices)
            final_segments = np.array(final_segments)
            masks = np.array(masks)
            return [indices, final_segments, masks], np.array(sentiments)


