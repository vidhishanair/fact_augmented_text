#!/usr/bin/env python3
# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT-joint baseline for NQ v1.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re
import time

import enum

from bert import modeling
from bert import optimization
from bert import tokenization

import numpy as np
import tensorflow as tf

import spacy

from fat.fat_bert_nq.ppr.shortest_path_lib import ShortestPath

nlp = spacy.load("en_core_web_lg")

# from fat.fat_bert_nq.ppr.apr_lib import ApproximatePageRank

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("eval_data_path", None, "Precomputed eval path for dev set")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")
flags.DEFINE_string("eval_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

# flags.DEFINE_string("apr_files_dir", None,
#                      "Location of KB for Approximate Page Rank")

flags.DEFINE_string(
    "predict_file", None,
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in NQ prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("create_pretrain_data", False, "Whether to create_pretraining_data.")

flags.DEFINE_bool(
    "use_random_fact_generator", False,
    "Whether to retreive random facts "
    "models and False for cased models.")

# flags.DEFINE_bool(
#     "use_question_entities", False,
#     "Whether to use question entities as seeds "
#     "models and False for cased models.")
flags.DEFINE_bool(
    "mask_non_entity_in_text", False,
    "Whether to mask non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_text_only", False,
    "Whether to use text only version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_masked_text_only", False,
    "Whether to use text only version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_text_and_facts", False,
    "Whether to use text and facts version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "augment_facts", False,
    "Whether to do fact extraction and addition ")
flags.DEFINE_bool(
    "anonymize_entities", False,
    "Whether to do add anonymized version")
flags.DEFINE_bool(
    "use_named_entities_to_filter", False,
    "Whether to use_ner_to_filter")

flags.DEFINE_bool(
    "use_shortest_path_facts", False,
    "Whether to do shortest_path expt")
flags.DEFINE_bool(
    "shuffle_shortest_path_facts", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "add_random_question_facts_to_shortest_path", False,
    "Whether to retreive random facts "
    "models and False for cased models.")
flags.DEFINE_bool(
    "add_random_walk_question_facts_to_shortest_path", False,
    "Whether to retreive random walk facts "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_passage_rw_facts_in_shortest_path", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "use_question_rw_facts_in_shortest_path", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "create_fact_annotation_data", False,
    "Whether to do shuffle hortest_path expt")
# flags.DEFINE_bool(
#     "use_only_random_facts_of_question", False,
#     "Whether to use only random_facts")
flags.DEFINE_bool(
    "use_question_to_passage_facts_in_shortest_path", False,
    "Whether to use only question to passage facts")
flags.DEFINE_bool(
    "use_question_level_apr_data", False,
    "Whether to use only question to passage facts")
flags.DEFINE_bool(
    "use_fixed_training_data", False,
    "Whether to use fixed unique ids from list")
tf.flags.DEFINE_string(
    "fixed_training_data_filepath", None,
    "Filepath of fixed training data - unique ids")
tf.flags.DEFINE_string(
    "analyse_incorrect_preds", None,
    "Flag to print incorrect predictions")

flags.DEFINE_integer("num_facts_limit", -1,
                     "Limiting number of facts")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("use_entity_markers", False, "Whether to add explicit entity seperators")
flags.DEFINE_float("alpha", 0.9,
                   "Alpha as restart prob for RWR")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    Relevant_and_Necessary_but_Not_Sufficient = 0
    Relevant_Necessary_and_Sufficient = 1
    Relevant_but_not_Necessary_and_Not_Sufficient = 2
    Irrelevant = 3


class Answer(collections.namedtuple("Answer", ["type", "text"])):
    """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

    def __new__(cls, type_, text=None):
        return super(Answer, cls).__new__(cls, type_, text)


class NqExample(object):
    """A single training/test example."""

    def __init__(self,
                 example_id,
                 question,
                 question_entities,
                 long_answer_text,
                 short_answer_text,
                 short_answer_entities,
                 kb_path,
                 path_relevance_annotation):
        self.example_id = example_id
        self.question = question
        self.question_entities = question_entities
        self.long_answer_text = long_answer_text
        self.short_answer_text = short_answer_text
        self.short_answer_entities = short_answer_entities
        self.kb_path = kb_path
        self.path_relevance_annotation = path_relevance_annotation


def make_nq_answer(answer):
    """Makes an Answer object following NQ conventions.

    Args:
      contexts: string containing the context
      answer: dictionary with `span_start` and `input_text` fields

    Returns:
      an Answer object. If the Answer type is YES or NO or LONG, the text
      of the answer is the long answer. If the answer type is UNKNOWN, the text of
      the answer is empty.
    """

    answer_type = None
    if answer.lower() == 'Relevant Necessary and Sufficient':
        answer_type = AnswerType.Relevant_Necessary_and_Sufficient
    elif answer.lower() == "Relevant but not Necessary and Not Sufficient":
        answer_type = AnswerType.Relevant_but_not_Necessary_and_Not_Sufficient
    elif answer.lower() == "Relevant and Necessary but Not Sufficient":
        answer_type = AnswerType.Relevant_and_Necessary_but_Not_Sufficient
    elif answer.lower() == "Irrelevant":
        answer_type = AnswerType.Irrelevant

    return Answer(answer_type, text=answer)


def create_example_from_line(line):
    """Creates an NQ example from a given line of JSON."""
    items = line.split("\t")
    path_annotation = make_nq_answer(items[7])
    example = NqExample(
        example_id=items[0],
        question=items[1],
        question_entities=items[2],
        long_answer_text=items[3],
        short_answer_text=items[4],
        short_answer_entities=items[5],
        kb_path=items[6],
        path_relevance_annotation=path_annotation)

    return example


def convert_examples_to_features(example, tokenizer, is_training, output_fn, pretrain_file=None):
    """Converts an NqExamples into InputFeatures."""
    # num_spans_to_ids = collections.defaultdict(list)
    example_index = example.example_id
    feature, stats = convert_single_example(example, tokenizer, is_training, pretrain_file)
    feature.example_index = example_index
    feature.unique_id = feature.example_index
    output_fn(feature)


def convert_single_example(example, tokenizer, apr_obj, is_training, pretrain_file=None, fixed_train_list=None):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    tok_to_textmap_index = []
    orig_to_tok_index = []

    question_sub_tokens = []
    question_tokens = example.question.split()[1:]
    for (i, token) in enumerate(question_tokens):
        orig_to_tok_index.append(len(question_sub_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_textmap_index.extend([i] * len(sub_tokens))
        tok_to_orig_index.extend([i] * len(sub_tokens))
        question_sub_tokens.extend(sub_tokens)

    # The -4 accounts for [CLS], [SEP]
    max_tokens_for_path = FLAGS.max_seq_length - len(question_sub_tokens) - 2

    path_sub_tokens = []
    path_tokens = example.kb_path.split()
    for (i, token) in enumerate(path_tokens):
        sub_tokens = tokenize(tokenizer, token)
        path_sub_tokens.extend(sub_tokens)
        if len(path_sub_tokens) >= max_tokens_for_path:
            break

    tokens = []
    segment_ids = []

    # token_to_orig_map = {}
    # token_is_max_context = {}
    tokens.append("[CLS]")
    segment_ids.append(0)

    tokens.extend(question_sub_tokens)
    segment_ids.extend([0] * len(question_sub_tokens))

    tokens.append("[SEP]")
    segment_ids.append(0)

    tokens.extend(path_sub_tokens)
    segment_ids.extend([1] * len(path_sub_tokens))

    if len(tokens) > FLAGS.max_seq_length:
        tokens = tokens[:FLAGS.max_seq_length]
        segment_ids = segment_ids[:FLAGS.max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids) Since we now can have 0/PAD in between due to masking non-entity tokens

    input_mask = [0 if item == 0 else 1 for item in input_ids]

    # Zero-pad up to the sequence length.
    padding = [0] * (FLAGS.max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)
    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    feature = InputFeatures(
        unique_id=-1,
        example_index=-1,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        answer_text=example.path_relevance_annotation.text,
        answer_label=example.path_relevance_annotation.type.value)  # Added facts to is max context and token to orig?
    feature_stats = {}

    return feature, feature_stats


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens


class CreateTFExampleFn(object):
    """Functor for creating NQ tf.Examples."""

    def __init__(self, is_training):
        self.is_training = is_training
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        mode = 'train' if is_training else 'dev'

    def process(self, example, pretrain_file=None, fixed_train_list=None):
        """Coverts an NQ example in a list of serialized tf examples."""
        input_feature, stat_counts = convert_single_example(example, self.tokenizer,
                                                            self.is_training, pretrain_file, fixed_train_list)
        input_feature.example_index = int(example["id"])
        input_feature.unique_id = int(example["id"])

        def create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([input_feature.unique_id])
        features["input_ids"] = create_int_feature(input_feature.input_ids)
        features["input_mask"] = create_int_feature(input_feature.input_mask)
        features["segment_ids"] = create_int_feature(input_feature.segment_ids)
        features["answer_label"] = create_int_feature([input_feature.answer_label])

        # if self.is_training:
        #   features["answer_label"] = create_int_feature([input_feature.answer_label])
        # else:
        #   token_map = [-1] * len(input_feature.input_ids)
        #   for k, v in input_feature.token_to_orig_map.items():
        #     token_map[k] = v
        #   features["token_map"] = create_int_feature(token_map)

        return tf.train.Example(features=tf.train.Features(
            feature=features)).SerializeToString(), stat_counts


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 answer_label,
                 answer_text):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answer_text = answer_text
        self.answer_label = answer_label


def read_nq_examples(input_file, is_training):
    """Read a NQ json file into a list of NqExample."""
    input_paths = tf.gfile.Glob(input_file)
    input_data = []

    def _open(path):
        if path.endswith(".gz"):
            return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))
        else:
            return tf.gfile.Open(path, "r")

    for path in input_paths:
        tf.logging.info("Reading: %s", path)
        with _open(path) as input_file:
            for line in input_file:
                input_data.append(create_example_from_line(line))

    return input_data


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Get the logits for the start and end predictions.
    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    # output_weights = tf.get_variable(
    #     "cls/nq/output_weights", [2, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))
    #
    # output_bias = tf.get_variable(
    #     "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())
    #
    # final_hidden_matrix = tf.reshape(final_hidden,
    #                                  [batch_size * seq_length, hidden_size])
    # logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    #
    # logits = tf.reshape(logits, [batch_size, seq_length, 2])
    # logits = tf.transpose(logits, [2, 0, 1])
    #
    # unstacked_logits = tf.unstack(logits, axis=0)
    #
    # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # Get the logits for the answer type prediction.
    answer_type_output_layer = model.get_pooled_output()
    answer_type_hidden_size = answer_type_output_layer.shape[-1].value

    num_answer_types = 4  # Relevant_Nec_Suff, Rel_Nec_Not_Suff, Rel_Not_Nec_Not_Suf, Irr
    answer_type_output_weights = tf.get_variable(
        "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    answer_type_output_bias = tf.get_variable(
        "answer_type_output_bias", [num_answer_types],
        initializer=tf.zeros_initializer())

    answer_type_logits = tf.matmul(
        answer_type_output_layer, answer_type_output_weights, transpose_b=True)
    answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                        answer_type_output_bias)

    return answer_type_logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        answer_label = features["answer_label"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        answer_type_logits = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            # Computes the loss for positions.
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            # Computes the loss for labels.
            def compute_label_loss(logits, labels):
                one_hot_labels = tf.one_hot(
                    labels, depth=len(AnswerType), dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
                return loss

            # start_positions = features["start_positions"]
            # end_positions = features["end_positions"]
            answer_types = features["answer_label"]

            # start_loss = compute_loss(start_logits, start_positions)
            # end_loss = compute_loss(end_logits, end_positions)
            answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

            total_loss = answer_type_loss

            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            seq_length = modeling.get_shape_list(input_ids)[1]

            # Computes the loss for positions.
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            # Computes the loss for labels.
            def compute_label_loss(logits, labels):
                one_hot_labels = tf.one_hot(
                    labels, depth=len(AnswerType), dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
                return loss

            # start_positions = features["start_positions"]
            # end_positions = features["end_positions"]
            answer_types = features["answer_label"]

            # start_loss = compute_loss(start_logits, start_positions)
            # end_loss = compute_loss(end_logits, end_positions)
            answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

            total_loss = answer_type_loss
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            pred_label = sorted(
                enumerate(answer_type_logits, 1), key=lambda x: x[1], reverse=True)[0]
            predictions = {
                "unique_ids": unique_ids,
                "answer_type_logits": answer_type_logits,
                "input_ids": input_ids,
                "predicted_label": pred_label[0],
                "predicted_label_score": pred_label[1],
                "answer_label": answer_label
                # "loss": total_loss,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "answer_label": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # d = tf.data.TFRecordDataset(input_file)
        d = tf.data.Dataset.list_files(input_file, shuffle=False)
        d = d.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=50, sloppy=is_training))
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        if filename is not None:
            self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1
        tf_example = self.get_processed_feature(feature)
        self._writer.write(tf_example.SerializeToString())

    def get_processed_feature(self, feature):
        """Return feature as an InputFeature."""

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["example_ids"] = create_int_feature([feature.example_index])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["answer_label"] = create_int_feature(feature.answer_label)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example

    def close(self):
        self._writer.close()


# Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
    """Eval data available for a single example."""

    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}


class ScoreSummary(object):

    def __init__(self):
        self.predicted_label = None
        self.answer_type_logits = None


# def read_candidates_from_one_split(input_path):
#   """Read candidates from a single jsonl file."""
#   candidates_dict = {}
#   with gzip.GzipFile(fileobj=tf.gfile.Open(input_path, "rb")) as input_file:
#     tf.logging.info("Reading examples from: %s", input_path)
#     for line in input_file:
#       e = json.loads(line.decode('utf-8'))
#       candidates_dict[e["example_id"]] = e["long_answer_candidates"]
#   return candidates_dict


# def read_candidates(input_pattern):
#   """Read candidates with real multiple processes."""
#   input_paths = tf.gfile.Glob(input_pattern)
#   final_dict = {}
#   for input_path in input_paths:
#     final_dict.update(read_candidates_from_one_split(input_path))
#   return final_dict

# def read_document_tokens_from_one_split(input_path):
#     """Read candidates from a single jsonl file."""
#     candidates_dict = {}
#     with gzip.GzipFile(fileobj=tf.gfile.Open(input_path, "rb")) as input_file:
#         tf.logging.info("Reading doc examples from: %s", input_path)
#         for line in input_file:
#             e = json.loads(line.decode('utf-8'))
#             candidates_dict[e["example_id"]] = [token['token'] for token in e['document_tokens']]
#     return candidates_dict
#
#
# def read_doc_tokens(input_pattern):
#     """Read candidates with real multiple processes."""
#     input_paths = tf.gfile.Glob(input_pattern)
#     final_dict = {}
#     for input_path in input_paths:
#         final_dict.update(read_document_tokens_from_one_split(input_path))
#     return final_dict


# def get_best_indexes(logits, n_best_size):
#   """Get the n-best logits from a list."""
#   index_and_score = sorted(
#       enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
#   best_indexes = []
#   for i in range(len(index_and_score)):
#     if i >= n_best_size:
#       break
#     best_indexes.append(index_and_score[i][0])
#   return best_indexes
#
#
# def compute_predictions(example, tokenizer = None, pred_fp = None, doc_tokens_dict=None):
#   """Converts an example into an NQEval object for evaluation."""
#   predictions = []
#   n_best_size = 10
#   max_answer_length = 30
#   num_hops = None
#   example_id = None
#   for unique_id, result in example.results.items():
#     if unique_id not in example.features:
#       raise ValueError("No feature found with unique_id:", unique_id)
#     token_map = example.features[unique_id]["token_map"].int64_list.value
#     input_ids = example.features[unique_id]["input_ids"].int64_list.value
#     # example_id = example.features[unique_id]["example_ids"].int64_list.value[0]
#     masked_input_ids = []
#     if FLAGS.mask_non_entity_in_text and FLAGS.use_text_only:
#         masked_input_ids = example.features[unique_id]["text_only_input_ids"].int64_list.value
#     if FLAGS.mask_non_entity_in_text and FLAGS.use_masked_text_only:
#         masked_input_ids = example.features[unique_id]["masked_text_tokens_input_ids"].int64_list.value
#     if FLAGS.mask_non_entity_in_text and FLAGS.use_text_and_facts:
#         masked_input_ids = example.features[unique_id]["masked_text_tokens_with_facts_input_ids"].int64_list.value
#     if FLAGS.anonymize_entities:
#         masked_input_ids = example.features[unique_id]["anonymized_text_only_tokens_input_ids"].int64_list.value
#     if FLAGS.use_shortest_path_facts:
#         num_hops = example.features[unique_id]["shortest_path_num_hops"].int64_list.value[0]
#     start_indexes = get_best_indexes(result["start_logits"], n_best_size)
#     end_indexes = get_best_indexes(result["end_logits"], n_best_size)
#     summary = None
#     for start_index in start_indexes:
#       for end_index in end_indexes:
#         if end_index < start_index:
#           continue
#         if token_map[start_index] == -1:
#           continue
#         if token_map[end_index] == -1:
#           continue
#         length = end_index - start_index + 1
#         if length > max_answer_length:
#           continue
#         summary = ScoreSummary()
#         summary.short_span_score = (
#             result["start_logits"][start_index] +
#             result["end_logits"][end_index])
#         summary.cls_token_score = (
#             result["start_logits"][0] + result["end_logits"][0])
#         summary.answer_type_logits = result["answer_type_logits"]
#         start_span = token_map[start_index]
#         end_span = token_map[end_index] + 1
#
#         # Span logits minus the cls logits seems to be close to the best.
#         score = summary.short_span_score - summary.cls_token_score
#         predictions.append((score, summary, start_span, end_span, input_ids, masked_input_ids, unique_id))
#     if summary is None:
#       # Hacking a summary where the first token of the context is the pred span
#       summary = ScoreSummary()
#       start_index = 0
#       # Finding the first token that is actually from the context
#       for idx, tokenmapval in enumerate(token_map):
#         if tokenmapval != -1:
#           start_index = idx
#           break
#       end_index = start_index
#       summary.short_span_score = (
#           result["start_logits"][start_index] + result["end_logits"][end_index])
#       summary.cls_token_score = (
#           result["start_logits"][0] + result["end_logits"][0])
#       summary.answer_type_logits = result["answer_type_logits"]
#       start_span = token_map[start_index]
#       end_span = token_map[end_index] + 1
#       # Span logits minus the cls logits seems to be close to the best.
#       score = summary.short_span_score - summary.cls_token_score
#       predictions.append((score, summary, start_span, end_span, input_ids, masked_input_ids, None))
#
#   # tf.logging.info("Predictions list len : %d", len(predictions))
#   # tf.logging.info("Len of example results: %d", len(example.results.items()))
#   if len(predictions) == 0:
#     summary = ScoreSummary()
#     summary.short_span_score = (0)
#     summary.cls_token_score = (0)
#     #summary.answer_type_logits = result["answer_type_logits"]
#     #start_span = token_map[start_index]
#     #end_span = token_map[end_index] + 1
#     #short_span = Span(-1, -1)
#     #long_span = Span(-1, -1)
#     start_span = -1
#     end_span = -1
#     score = 0
#     predictions.append((score, summary, start_span, end_span, [], [], None))
#   #else:
#   #print(len(example.results.items()))
#   score, summary, start_span, end_span, input_ids, masked_input_ids, unique_id = sorted(
#       predictions, reverse=True, key=lambda tup: tup[0])[0]
#   short_span = Span(start_span, end_span)
#   long_span = Span(-1, -1)
#   for c in example.candidates:
#     start = short_span.start_token_idx
#     end = short_span.end_token_idx
#     if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
#       long_span = Span(c["start_token"], c["end_token"])
#       break
#
#   if FLAGS.analyse_incorrect_preds:
#       # use example id for apr obj
#       sorted_preds = sorted(
#           predictions, reverse=True, key=lambda tup: tup[0])
#       for (n_score, n_summary, n_start_span, n_end_span, n_input_ids, n_masked_input_ids, n_unique_id) in sorted_preds[0:5]:
#           if n_unique_id is None:
#               continue
#           # question_entity_map = example.features[n_unique_id]["question_entity_map"].int64_list.value
#           # document_entity_list = example.features[n_unique_id]["entity_list"].int64_list.value
#           # token_to_textmap_index = example.features[n_unique_id]["token_to_textmap_index"].int64_list.value
#
#           # start_index = token_to_textmap_index[n_start_span]
#           # end_index = token_to_textmap_index[min(n_end_span,
#           #     len(token_to_textmap_index) -
#           #     1)]  # putting this min check need to check all this later
#           #
#           # sub_list = document_entity_list[start_index:end_index + 1]
#           #
#           # question_entities = set()
#           # for start_idx in question_entity_map.keys():
#           #     for sub_span in question_entity_map[start_idx]:
#           #         question_entities.add(sub_span[1])
#           # question_entities = list(question_entities)
#
#           answer = doc_tokens_dict[example.example_id][n_start_span:n_end_span]
#
#           input_ids = list(map(int, input_ids))
#           question = []
#           text = []
#           facts = []
#           current='question'
#           for token in input_ids:
#               try:
#                   word = tokenizer.convert_ids_to_tokens([token])[0]
#                   if current == 'question':
#                       question.append(word)
#                   elif current == 'text':
#                       text.append(word)
#                   elif current == 'facts':
#                       facts.append(word)
#                   else:
#                       print("Some exception in current word")
#                       print(current)
#                   if word == '[SEP]' and current == 'question':
#                       current = 'text'
#                   elif word == '[SEP]' and current == 'text':
#                       current = 'facts'
#                   else:
#                       continue
#               except:
#                   print('didnt tokenize')
#           pred_fp.write(str(example.example_id)+"\t")
#           pred_fp.write(" ".join(question).replace(" ##","")+"\t")
#           pred_fp.write(" ".join(text).replace(" ##", "")+"\t")
#           pred_fp.write(" ".join(facts).replace(" ##","")+"\t")
#           pred_fp.write(" ".join(answer)+"\n")
#
#
#   input_ids = list(map(int, input_ids))
#   if len(input_ids) > 0:
#       input_text = tokenizer.convert_ids_to_tokens(input_ids)
#       input_text = (" ".join(input_text)).replace(" ##","")
#   else:
#       input_text = ""
#
#   masked_input_ids = list(map(int, masked_input_ids))
#   if len(masked_input_ids) > 0:
#       masked_input_text = tokenizer.convert_ids_to_tokens(masked_input_ids)
#       masked_input_text = (" ".join(masked_input_text)).replace(" ##","")
#   else:
#       masked_input_text = ""
#
#   summary.predicted_label = {
#       "example_id": example.example_id,
#       "long_answer": {
#           "start_token": long_span.start_token_idx,
#           "end_token": long_span.end_token_idx,
#           "start_byte": -1,
#           "end_byte": -1
#       },
#       "long_answer_score": score,
#       "short_answers": [{
#           "start_token": short_span.start_token_idx,
#           "end_token": short_span.end_token_idx,
#           "start_byte": -1,
#           "end_byte": -1
#       }],
#       "short_answers_score": score,
#       "yes_no_answer": "NONE",
#       "input_ids": input_ids,
#       "input_text": input_text,
#       "masked_input_ids": masked_input_ids,
#       "masked_input_text": masked_input_text,
#       "shortest_path_num_hops": num_hops,
#   }
#
#   # if FLAGS.write_pred_analysis:
#       # input_ids = map(int, input_ids)
#       # question = []
#       # text = []
#       # facts = []
#       # current='question'
#       # for token in input_ids:
#       #     try:
#       #         word = tokenizer.convert_ids_to_tokens([token])[0]
#       #         if current == 'question':
#       #             question.append(word)
#       #         elif current == 'text':
#       #             text.append(word)
#       #         elif current == 'facts':
#       #             facts.append(word)
#       #         else:
#       #             print("Some exception in current word")
#       #             print(current)
#       #         if word == '[SEP]' and current == 'question':
#       #             current = 'text'
#       #         elif word == '[SEP]' and current == 'text':
#       #             current = 'facts'
#       #         else:
#       #             continue
#       #     except:
#       #         print('didnt tokenize')
#       #
#       # pred_fp.write(" ".join(question).replace(" ##","")+"\t")
#       # pred_fp.write(" ".join(text).replace(" ##", "")+"\t")
#       # pred_fp.write(" ".join(facts).replace(" ##","")+"\n")
#   return summary
#
#
# def compute_pred_dict(raw_results, tokenizer=None, pred_fp=None, doc_tokens_dict=None):
#   """Computes official answer key from raw logits."""
#   sess = tf.Session()
#   # raw_results_by_id = [
#   # (int(res["unique_id"] + 1), res) for res in raw_results]
#   raw_results = [res for res in raw_results]
#   raw_results_ids = tf.to_int32(
#       np.array([int(res["unique_id"] + 1) for res in raw_results
#                ])).eval(session=sess)
#   raw_results_by_id = list(zip(raw_results_ids, raw_results))
#
#   # # Cast example id to int32 for each example, similarly to the raw results.
#   # all_candidates = candidates_dict.items()
#   # example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates
#   #                                    ])).eval(session=sess)
#   # examples_by_id = list(zip(example_ids, all_candidates))
#   #
#   # # Cast unique_id also to int32 for features.
#   # feature_ids = []
#   # features = []
#   # for f in dev_features:
#   #   feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
#   #   features.append(f.features.feature)
#   # feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
#   # features_by_id = list(zip(feature_ids, features))
#
#   # Join examplew with features and raw results.
#   # examples = []
#   # merged = sorted(
#   #     examples_by_id + raw_results_by_id + features_by_id,
#   #     key=lambda tup: tup[0])
#   # for idx, datum in merged:
#   #   if isinstance(datum, tuple):
#   #     examples.append(EvalExample(datum[0], datum[1]))
#   #   elif "token_map" in datum:
#   #     examples[-1].features[idx] = datum
#   #   else:
#   #     examples[-1].results[idx] = datum
#
#   # Construct prediction objects.
#   print("Computing final summaries...")
#   summary_dict = {}
#   nq_pred_dict = {}
#   for e in examples:
#     # if True or FLAGS.mask_non_entity_in_text:
#     #     if len(list(e.features.keys())) == 0 and len(list(e.results)) == 0:
#     #         continue
#     summary = compute_predictions(e, tokenizer,pred_fp, doc_tokens_dict)
#     summary_dict[e.example_id] = summary
#     nq_pred_dict[e.example_id] = summary.predicted_label
#     if len(nq_pred_dict) % 100 == 0:
#       print("Examples processed: %d", len(nq_pred_dict))
#   print("Done computing predictions.")
#
#   return nq_pred_dict

def format_and_write_result(result, tokenizer, output_fp):
    input_ids = result["input_ids"].int64_list.value
    input_ids = map(int, input_ids)
    question = []
    facts = []
    current = 'question'
    for token in input_ids:
        try:
            word = tokenizer.convert_ids_to_tokens([token])[0]
            if current == 'question':
                question.append(word)
            elif current == 'facts':
                facts.append(word)
            else:
                print("Some exception in current word")
                print(current)
            if word == '[SEP]' and current == 'question':
                current = 'facts'
            else:
                continue
        except:
            print('didnt tokenize')
    question = " ".join(question).replace(" ##", "")
    facts = " ".join(facts).replace(" ##", "")
    predicted_label = result["predicted_label"].int64_list.value[0]
    predicted_label_text = AnswerType(predicted_label).name
    answer_label = result["answer_label"].int64_list.value[0]
    answer_label_text = AnswerType(answer_label).name
    output_fp.write(question + "\t" + facts + "\t" +
                    predicted_label_text + "\t" + answer_label_text + "\n")


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `{do_train,do_predict}` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_precomputed_file:
            raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                             "must be specified.")
        if not FLAGS.train_num_precomputed:
            raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                             "must be specified.")

    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # Maintaining tokenizer for future use # pylint: disable=unused-variable
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_features = FLAGS.train_num_precomputed
        num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                              FLAGS.num_train_epochs)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.predict_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training on precomputed features *****")
        tf.logging.info("  Num split examples = %d", num_train_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
    train_filename = FLAGS.train_precomputed_file
    train_input_fn = input_fn_builder(
        input_file=train_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    if FLAGS.do_train:
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        if not FLAGS.output_prediction_file:
            raise ValueError(
                "--output_prediction_file must be defined in predict mode.")
        print("Evaluating 1000 steps now")
        estimator.evaluate(input_fn=train_input_fn, steps=1000)

        eval_filename = FLAGS.eval_precomputed_file

        tf.logging.info("***** Running predictions *****")

        predict_input_fn = input_fn_builder(
            input_file=eval_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of steps.
        all_results = []
        loss = []
        output_fp = tf.gfile.Open(FLAGS.output_prediction_file, "w")
        for result in estimator.predict(predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            format_and_write_result(result, tokenizer, output_fp)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()