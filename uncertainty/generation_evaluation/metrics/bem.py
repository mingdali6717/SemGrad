import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
from itertools import chain
import gc
import torch
from loguru import logger
import time
VOCAB_PATH = os.path.abspath("uncertainty/generation_evaluation/metrics/vocab.txt")
CACHED_BEM_PATH = 'data/model/bem/bert-tensorflow2-answer-equivalence-bem-v1'
MAX_LENGTH=512






def pad(a, length=MAX_LENGTH):
    return np.append(a, np.zeros(length - a.shape[-1], np.int32))




#bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
#bem = hub.load("E:/answer_equivalence_bem_1")

# examples = [{
#     'question': 'why is the sky blue',
#     'reference': 'light scattering',
#     'candidate': 'scattering of light'
#     }]

class BemCalculator:
    
    def __init__(self, model_path:str = 'https://tfhub.dev/google/answer_equivalence/bem/1', device_name="gpu0"):

        

        

        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
          try:
            
            # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            
            
          except RuntimeError as e:

            logger.info(e)
        else:
            raise RuntimeError("No GPU devices are available.")

        if "gpu" in device_name:
            device = tf.device(f'/gpu:{device_name.replace("gpu","")}')
        else:
            device = gpus[0]
        logger.info(f"calclating bem score with {device}")
        
        
        with device:
            self.bem = hub.load(model_path)
        #hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
        vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        ),
        num_oov_buckets=1)
        self.cls_id, self.sep_id = tf.cast(vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]'])), dtype=tf.int64)
        self.tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_table,
                               token_out_type=tf.int64,
                               preserve_unused_token=True,
                               lower_case=True)


    def __call__(self, examples, batch_size = 25):

        
        bem_scores = []
        for index in tqdm(range(0, len(examples), batch_size)):
            batch_examples = examples[index:index + batch_size]
            # start_time = time.time()
            batch_inputs = self.bertify_examples(batch_examples)
            # preparation_time = time.time() - start_time
            # logger.info(f"time used for preparing data {preparation_time}")
        
        
            # The outputs are raw logits.
            batch_raw_outputs = self.bem(batch_inputs)
            # logger.info(f"running time: {time.time() - start_time - preparation_time}")
            # They can be transformed into a classification 'probability' like so:
            batch_bem_score = softmax(batch_raw_outputs, axis=1)[:, 1].astype(np.float64).tolist()
            bem_scores.extend(batch_bem_score)
                            
        return bem_scores
    
    def bertify_examples(self, examples):
        input_ids = []
        segment_ids = []
        string_to_tokenize = list(chain(*[[ex['question'], ex['reference'], ex['candidate']] for ex in examples]))
        tokenized_string = self.tokenizer.tokenize(string_to_tokenize).merge_dims(1, 2)
        questions =tf.concat([tokenized_string[i:i+1] for i in range(0, len(string_to_tokenize), 3)], 0)
        references = tf.concat([tokenized_string[i+1:i+2] for i in range(0, len(string_to_tokenize), 3)], 0)
        candidates = tf.concat([tokenized_string[i+2:i+3] for i in range(0, len(string_to_tokenize), 3)], 0)

        
        max_len = MAX_LENGTH-4

        new_candidates = []

        for q, r, c in zip(questions, references, candidates):
        
            ex_len = q.shape[0] + r.shape[0] + c.shape[0]

            if ex_len > max_len:
                
                new_c = c[:-(ex_len-max_len)]
            else:
                new_c = c

            new_candidates.append(new_c)
        candidates = tf.ragged.stack(new_candidates).with_row_splits_dtype(questions.dtype)


        input_ids, segment_ids = text.combine_segments(
            (candidates, references, questions), self.cls_id, self.sep_id)
        
        input_ids = input_ids.numpy()
        segment_ids = segment_ids.numpy()
        
        

        return {'input_ids': np.stack([pad(input_ids[i]) for i in range(input_ids.shape[0])]), 'segment_ids': np.stack([pad(segment_ids[i]) for i in range(segment_ids.shape[0])])}
    
    def release_model(self):
        tf.keras.backend.clear_session()
        del self.bem
        del self.tokenizer
        gc.collect()

# score = bem_score(examples)
# print(score)






