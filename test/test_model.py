import os
import sys
sys.path.append("..")
sys.path.append(".")
import json
import csv
import numpy as np
import tensorflow as tf
from model.checkpoint_loader import build_model_from_config 
from utils.Adam_mult import AdamWarmup, calc_train_steps
import unittest



class Test_PowerBERTModels(unittest.TestCase):
    def test_finetuned_model(self):
        """Test code for finetuning task."""

        fine_tuned_model, config = build_model_from_config(
                './bert_config.json',
                output_dim=2,
                seq_len=64,
                FLAG_EXTRACT_LAYER=0,
                TASK='cola')

        decay_steps, warmup_steps = calc_train_steps(
                8550,
                batch_size=128,
                epochs=3,
            )
        
        fine_tuned_model.compile(
                AdamWarmup(decay_steps=decay_steps,
                            warmup_steps=warmup_steps,
                            lr=3e-5,
                            lr_mult=None),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
            )

        print("Fine-tuned model summary: ", fine_tuned_model.summary())
        del fine_tuned_model

    def test_search_model(self):
        """Test code for configurtion search."""
        configuration_search_model, config = build_model_from_config(
                './bert_config.json',
                output_dim=2,
                seq_len=64,
                LAMBDA=3e-3,
                FLAG_EXTRACT_LAYER=1,
                TASK='cola')

        decay_steps, warmup_steps = calc_train_steps(
                8550,
                batch_size=128,
                epochs=3,
            )
        configuration_search_model.compile(
                AdamWarmup(decay_steps=decay_steps,
                            warmup_steps=warmup_steps,
                            lr=3e-5,
                            lr_mult=None),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
            )

        print("Configuration search model summary: ", configuration_search_model.summary())
        del configuration_search_model


    def test_retrained_model(self):
        """Test code for retrained model."""
        retrained_model, config = build_model_from_config(
                './bert_config.json',
                output_dim=2,
                seq_len=64,
                retention_configuration=[64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8],
                FLAG_EXTRACT_LAYER=2,
                TASK='cola')

        decay_steps, warmup_steps = calc_train_steps(
                8550,
                batch_size=128,
                epochs=3,
            )
        retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps,
                            warmup_steps=warmup_steps,
                            lr=3e-5,
                            lr_mult=None),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
            )

        print("Retrained model summary: ", retrained_model.summary())
        del retrained_model
        
if __name__ == '__main__':
    unittest.main()


