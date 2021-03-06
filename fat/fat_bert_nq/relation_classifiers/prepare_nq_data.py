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

r"""Converts an NQ dataset file to tf examples.

Notes:
  - this program needs to be run for every NQ training shard.
  - this program only outputs the first n top level contexts for every example,
    where n is set through --max_contexts.
  - the restriction from --max_contexts is such that the annotated context might
    not be present in the output examples. --max_contexts=8 leads to about
    85% of examples containing the correct context. --max_contexts=48 leads to
    about 97% of examples containing the correct context.

Usage :
    python prepare_nq_data.py --shard_split_id 0 --task_id 0
                              --input_data_dir sharded_nq/train
                              --output_data_dir processed_fat_nq/train
                              --apr_files_dir kb_csr/
    This command would process sharded file 'nq-train-0000.jsonl.gz'.
    For every NQ Example, it would merge and chunk the long answer candidates
    using an overlapping sliding window.
    For each chunk, it would using the linked entities and extract relevant
    facts using PPR and append the facts to the chunk.
    Each chunk is returned as an individual NQ Example and all generated
    examples are stores in tf-record form.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random
from bert import tokenization
import tensorflow as tf
from fat.fat_bert_nq.relation_classifiers import nq_data_utils
from fat.fat_bert_nq.relation_classifiers import run_nq

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

flags.DEFINE_integer(
    "max_dev_tasks", 5,
    "Total no: of tasks for every split")
flags.DEFINE_integer(
    "max_dev_shard_splits", 17,
    "Total no: of splits in sharded data")

flags.DEFINE_integer("shard_split_id", None,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_string(
    "split", "train",
    "Train and dev split to read from and write to. Accepted values: ['train', 'dev', 'test']"
)

flags.DEFINE_string("predict_files", "", "Eval data")

flags.DEFINE_string("input_data_dir", "", "input_data_dir")

flags.DEFINE_string("output_data_dir", " ", "output_data_dir")
flags.DEFINE_bool("merge_eval", "True", "Flag for pre-proc or merge")
flags.DEFINE_string("pretrain_data_dir", " ", "pretrain_data_dir")


def main(_):
    examples_processed = 0
    instances_processed = 0
    num_examples_with_correct_context = 0

    if FLAGS.create_pretrain_data or FLAGS.create_fact_annotation_data:
        pretrain_file = open(nq_data_utils.get_sharded_filename(FLAGS.pretrain_data_dir,
                                                                FLAGS.split, FLAGS.task_id,
                                                                FLAGS.shard_split_id,
                                                                "txt"), 'w')
    else:
        pretrain_file = None

    if FLAGS.is_training:
        fixed_train_list = None
        if FLAGS.use_fixed_training_data:
            fp = open(FLAGS.fixed_training_data_filepath)
            fixed_train_list = []
            for line in fp:
                fixed_train_list.append(int(line))
        creator_fn = run_nq.CreateTFExampleFn(is_training=FLAGS.is_training)
        instances = []
        input_file = nq_data_utils.get_sharded_filename(FLAGS.input_data_dir,
                                                        FLAGS.split, FLAGS.task_id,
                                                        FLAGS.shard_split_id,
                                                        "jsonl.gz")
        annotation_data = nq_data_utils.get_annotated_nq_examples(os.path.join(FLAGS.input_data_dir,"train.tsv"))
        print("Reading file %s", input_file)
        file_stats_counter = {'example_count': 0, 'sp_recall_sum': 0, 'answer_reach_counter':0,
                              'single_answer_reach_counter':0, 'multi_answer_recall':0,
                              'single_answer_counter':0, 'multi_answer_counter':0, 'multi_answer_size_counter':0}
        for example in nq_data_utils.get_nq_examples(input_file):
            stats_count = None
            for instance, stats_count in creator_fn.process(example, pretrain_file, fixed_train_list, annotation_data):
                instances.append(instance)
                instances_processed += 1

            if example["has_correct_context"]:
                num_examples_with_correct_context += 1
            if examples_processed % 100 == 0:
                print("Examples processed: %d", examples_processed)
                print("Instances processed: %d", instances_processed)
            examples_processed += 1
            if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
                break
            #time.sleep(5)
        print("Examples with correct context retained: %d of %d",
              num_examples_with_correct_context, examples_processed)
        random.shuffle(instances)
        print("Total no: of instances in current shard: %d",
              len(instances))
        output_file = nq_data_utils.get_sharded_filename(FLAGS.output_data_dir,
                                                         FLAGS.split, FLAGS.task_id,
                                                         FLAGS.shard_split_id,
                                                         "tf-record")
       # stats_file = nq_data_utils.get_sharded_filename(FLAGS.output_data_dir,
       #                                                 FLAGS.split, FLAGS.task_id,
       #                                                 FLAGS.shard_split_id,
       #                                                 "stats.txt")
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for instance in instances:
                writer.write(instance)
        # with open(stats_file, 'w') as fp:
        #     print("Example count: %d", file_stats_counter['example_count'])
        #     print("Fact Recall sum: %d", file_stats_counter['sp_recall_sum'])
        #     print("Count with answers reached: %d", file_stats_counter['answer_reach_counter'])
        #     print("Single Answer Example count: %d", file_stats_counter['single_answer_counter'])
        #     print("Single Answer Reached count: %d", file_stats_counter['single_answer_reach_counter'])
        #     print("Multi Answer Example count: %d", file_stats_counter['multi_answer_counter'])
        #     print("Multi Answer recall sum: %d", file_stats_counter['multi_answer_recall'])
        #     print("Multi Answer Size counter: %d", file_stats_counter['multi_answer_size_counter'])
        #     fp.write("Example count: "+str(file_stats_counter['example_count'])+"\n")
        #     fp.write("Fact Recall sum: "+str(file_stats_counter['sp_recall_sum'])+"\n")
        #     fp.write("Count with answers reached: "+str(file_stats_counter['answer_reach_counter'])+"\n")
        #     fp.write("Single Answer Example count: "+str(file_stats_counter['single_answer_counter'])+"\n")
        #     fp.write("Single Answer Reached count: "+str(file_stats_counter['single_answer_reach_counter'])+"\n")
        #     fp.write("Multi Answer Example count: "+str(file_stats_counter['multi_answer_counter'])+"\n")
        #     fp.write("Multi Answer recall sum: "+str(file_stats_counter['multi_answer_recall'])+"\n")
        #     fp.write("Multi Answer Size counter: "+str(file_stats_counter['multi_answer_size_counter'])+"\n")

    # For eval - First process every shard in parallel
    elif not FLAGS.is_training and not FLAGS.merge_eval:
        input_file = nq_data_utils.get_sharded_filename(FLAGS.input_data_dir,
                                                        FLAGS.split, FLAGS.task_id,
                                                        FLAGS.shard_split_id,
                                                        "jsonl.gz")
        print("Reading file %s", input_file)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        eval_examples = run_nq.read_nq_examples(
            input_file=input_file, is_training=False)
        output_file = nq_data_utils.get_sharded_filename(FLAGS.output_data_dir,
                                                         FLAGS.split, FLAGS.task_id,
                                                         FLAGS.shard_split_id,
                                                         "tf-record")
        annotation_data = nq_data_utils.get_annotated_nq_examples(os.path.join(FLAGS.input_data_dir,"dev.tsv"))
        eval_writer = run_nq.FeatureWriter(filename=output_file, is_training=False)
        eval_features = []
        examples_processed = 0

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)
            examples_processed = len(eval_features)
            if examples_processed % 10 == 0:
                print("Examples processed: %d", examples_processed)
        print(list(annotation_data.keys())[0])
        _ = run_nq.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            is_training=False,
            output_fn=append_feature,
            pretrain_file=pretrain_file, annotation_data=annotation_data)
        eval_writer.close()

    # For eval - Fianlly merge all shards into 1
    else:
        instances = []
        for task in range(FLAGS.max_dev_tasks):
            for shard_split in range(FLAGS.max_dev_shard_splits):
                input_file = nq_data_utils.get_sharded_filename(FLAGS.input_data_dir,
                                                                FLAGS.split, task,
                                                                shard_split,
                                                                "tf-record")
                print("Reading file %s", input_file)
                instances.extend([
                    tf.train.Example.FromString(r)
                    for r in tf.python_io.tf_record_iterator(input_file)
                ])

        output_file = os.path.join(FLAGS.output_data_dir, "eval.tf-record")
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for instance in instances:
                writer.write(instance.SerializeToString())


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
