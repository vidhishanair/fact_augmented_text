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
import json, gzip
from bert import tokenization
import tensorflow as tf
from fat.fat_bert_nq import nq_data_utils
from fat.fat_bert_nq import run_nq
from fat.fat_bert_nq.ppr.apr_lib import ApproximatePageRank

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
flags.DEFINE_string("qi_apr_path", "", "QI PPR Path")
flags.DEFINE_string("ws_apr_path", "", "WS PPR Path")

flags.DEFINE_string("output_data_dir", " ", "output_data_dir")
flags.DEFINE_bool("merge_eval", "True", "Flag for pre-proc or merge")
flags.DEFINE_string("pretrain_data_dir", " ", "pretrain_data_dir")


def get_related_facts(question_entity_map, answer_entity_map, apr_obj,
                      filter_lower_case_entities=False, drop_facts=False, flip_facts=False):
                      # tokenizer, question_entity_map, answer=None, ner_entity_list=None,
                      # all_doc_tokens=None, fp=None, override_shortest_path=False, use_passage_seeds=True,
                      # use_question_seeds=False, seperate_diff_paths=False):
    """For a given doc span, use seed entities, do APR, return related facts.

    Args:
     doc_span: A document span dictionary holding spart start,
                     end and len details
     token_to_textmap_index: A list mapping tokens to their positions
                                   in full context
     entity_list: A list mapping every token to their
                         WikiData entity if present
     apr_obj: An ApproximatePageRank object
     tokenizer: BERT tokenizer

    Returns:
     nl_fact_tokens: Tokenized NL form of facts
    """

    question_entities = set()
    for start_idx in question_entity_map.keys():
        for sub_span in question_entity_map[start_idx]:
            ent_id = sub_span[1]
            if filter_lower_case_entities:
                if ent_id in apr_obj.data.ent2id:
                    ent_kb_id = apr_obj.data.ent2id[ent_id]
                    ent_name = apr_obj.data.entity_names['e'][str(ent_kb_id)]['name']
                    if ent_name != ent_name.lower():
                        question_entities.add(ent_id)
            else:
                question_entities.add(ent_id)
    question_entities = list(question_entities)

    answer_entities = set()
    for start_idx in answer_entity_map.keys():
        for sub_span in answer_entity_map[start_idx]:
            ent_id = sub_span[1]
            if filter_lower_case_entities:
                if ent_id in apr_obj.data.ent2id:
                    ent_kb_id = apr_obj.data.ent2id[ent_id]
                    ent_name = apr_obj.data.entity_names['e'][str(ent_kb_id)]['name']
                    if ent_name != ent_name.lower():
                        answer_entities.add(ent_id)
            else:
                answer_entities.add(ent_id)
    answer_entities = list(answer_entities)


    question_entity_ids = [int(apr_obj.data.ent2id[x]) for x in question_entities if x in apr_obj.data.ent2id]
    question_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids])

    # answer_entity_ids, answer_entity_names = [], str([])
    answer_entity_ids = [int(apr_obj.data.ent2id[x]) for x in answer_entities if x in apr_obj.data.ent2id]
    answer_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids])
    print(question_entity_names, answer_entity_names)
    num_hops = 0
    sp_only_facts = ""
    sp_facts, num_hops = apr_obj.get_shortest_path_facts(question_entities, answer_entities, passage_entities=[], seed_weighting=True)

    drop_flip_facts = sp_facts.copy()
    modified_facts = []
    if flip_facts:
        for x in sp_facts:
            ((subj, obj), (rel, score)) = x
            print(subj, obj, rel, apr_obj.data.rel_dict[(obj[0], subj[0])])
            if apr_obj.data.rel_dict[(obj[0], subj[0])] != 0:
                rev_rel = apr_obj.data.rel_dict[(obj[0], subj[0])]
                rev_name = apr_obj.data.entity_names['r'][str(rev_rel)]['name']
                print(subj, obj, rel, (rev_rel, rev_name))
                if random.random() > 0.7:
                    modified_facts.append(((obj, subj), ((rev_rel, rev_name), score)))
                else:
                    modified_facts.append(((subj, obj), (rel, score)))
            else:
                modified_facts.append(((subj, obj), (rel, score)))
        drop_flip_facts = modified_facts

    modified_facts = []
    if drop_facts:
        for x in sp_facts:
            if random.random() > 0.9:
                continue
            else:
                modified_facts.append(x)
        drop_flip_facts = modified_facts

    sp_rw_facts = sp_facts.copy()
    sp_dropflip_rw_facts = sp_facts.copy()
    unique_facts = []
    rw_facts = []
    if len(question_entities) > 0:
        unique_facts = apr_obj.get_facts(question_entities, topk=200, alpha=FLAGS.alpha, seed_weighting=True)
        rw_facts = sorted(unique_facts, key=lambda tup: tup[1][1], reverse=True)
    if FLAGS.num_facts_limit > 0:
        rw_facts = rw_facts[0:FLAGS.num_facts_limit]
    if len(sp_rw_facts) > 0:
        sp_rw_facts.extend(rw_facts)
    if len(sp_dropflip_rw_facts) > 0:
        sp_dropflip_rw_facts.extend(rw_facts)

    random.shuffle(sp_facts)
    random.shuffle(drop_flip_facts)
    random.shuffle(sp_dropflip_rw_facts)
    random.shuffle(sp_rw_facts)
    random.shuffle(rw_facts)

    sp_nl_facts = " . ".join([
        str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
        for x in sp_facts[0:FLAGS.num_facts_limit]
    ])
    drop_flip_nl_facts = " . ".join([
      str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
      for x in drop_flip_facts[0:FLAGS.num_facts_limit]
    ])
    sp_dropflip_rw_nl_facts = " . ".join([
      str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
      for x in sp_dropflip_rw_facts[0:FLAGS.num_facts_limit]
    ])
    sp_rw_nl_facts = " . ".join([
      str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
      for x in sp_rw_facts[0:FLAGS.num_facts_limit]
    ])
    rw_nl_facts = " . ".join([
      str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
      for x in rw_facts[0:FLAGS.num_facts_limit]
    ])
    return sp_nl_facts, drop_flip_nl_facts, sp_dropflip_rw_nl_facts,\
           sp_rw_nl_facts, rw_nl_facts, num_hops

def nq_jsonl_to_tsv(in_fname, out_fname):

    def extract_answer(tokens, span):
        """Reconstruct answer from token span and remove extra spaces."""
        start, end = span["start_token"], span["end_token"]
        ans = " ".join(tokens[start:end])
        # Remove incorrect spacing around punctuation.
        ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        ans = ans.replace("( ", "(").replace(" )", ")")
        ans = ans.replace("`` ", "\"").replace(" ''", "\"")
        ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
        return ans

    count = 0
    plain_ppr_apr_obj = ApproximatePageRank(mode=FLAGS.split, task_id=FLAGS.task_id,
                                                           shard_id=FLAGS.shard_split_id)

    with tf.gfile.GFile(in_fname, "rb") as infile, \
            tf.gfile.GFile(out_fname, "w") as outfile:
        for line in gzip.open(infile):
            ex = json.loads(line)
            # Remove any examples with more than one answer.
            if len(ex['annotations'][0]['short_answers']) != 1:
                continue
            # Questions in NQ do not include a question mark.
            question = ex["question_text"] + "?"
            answer_span = ex['annotations'][0]['short_answers'][0]
            answer_entities = answer_span['entity_map']
            question_entities = ex['question_entity_map']

            # Handle the two document formats in NQ (tokens or text).
            tokens = None
            if "document_tokens" in ex:
                tokens = [t["token"] for t in ex["document_tokens"]]
            elif "document_text" in ex:
                tokens = ex["document_text"].split(" ")

            answer = extract_answer(tokens, answer_span)

            ppr_sp_nl_facts, ppr_drop_flip_nl_facts, ppr_drop_flip_rw_nl_facts,\
            ppr_sp_rw_nl_facts, ppr_rw_nl_facts, ppr_num_hops = get_related_facts(question_entities, answer_entities, plain_ppr_apr_obj,
                              filter_lower_case_entities=False, drop_facts=False, flip_facts=False)

            try:
                qi_apr_obj = ApproximatePageRank(question_id=ex['example_id'], apr_path=FLAGS.qi_apr_path)

                qi_sp_nl_facts, qi_drop_flip_nl_facts, qi_drop_flip_rw_nl_facts,\
                qi_sp_rw_nl_facts, qi_rw_nl_facts, qi_num_hops = get_related_facts(question_entities, answer_entities, qi_apr_obj,
                                                                                  filter_lower_case_entities=False, drop_facts=False, flip_facts=False)
            except:
                cqi_sp_nl_facts, qi_drop_flip_nl_facts, qi_drop_flip_rw_nl_facts,\
                                        qi_sp_rw_nl_facts, qi_rw_nl_facts, qi_num_hops = "", "", "", "", "", 0

            try:
                ws_apr_obj = ApproximatePageRank(question_id=ex['example_id'], apr_path=FLAGS.ws_apr_path)

                ws_sp_nl_facts, ws_drop_flip_nl_facts, ws_drop_flip_rw_nl_facts,\
                ws_sp_rw_nl_facts, ws_rw_nl_facts, ws_num_hops = get_related_facts(question_entities, answer_entities, ws_apr_obj,
                                                                               filter_lower_case_entities=True, drop_facts=True, flip_facts=True)
            except:
                ws_sp_nl_facts, ws_drop_flip_nl_facts, ws_sp_rw_nl_facts, ws_drop_flip_rw_nl_facts, ws_rw_nl_facts, ws_num_hops = "","","","","", 0

            # Write this line as <question>\t<answer>
            output_json = {"example_id": ex['example_id'],
                           "question": question,
                           "answer": answer,
                           "plain_ppr_sp_num_hops": ppr_num_hops,
                           "plain_ppr_sp_facts": ppr_sp_nl_facts,
                           "plain_ppr_sp_rw_facts": ppr_sp_rw_nl_facts,
                           "plain_ppr_rw_facts": ppr_rw_nl_facts,
                           "qi_ppr_rw_facts": qi_rw_nl_facts,
                           "filtered_sp_num_hops": ws_num_hops,
                           "filtered_sp_facts": ws_sp_nl_facts,
                           "filtered_dropflip_sp_facts": ws_drop_flip_nl_facts,
                           "filtered_sp_rw_facts": ws_sp_rw_nl_facts,
                           "filtered_dropflip_sp_rw_facts": ws_drop_flip_rw_nl_facts,
                           "filtered_ws_rw_facts": ws_rw_nl_facts
                           }
            print(output_json)
            outfile.write(json.dumps(output_json)+"\n")
            count += 1
            tf.logging.log_every_n(
                tf.logging.INFO,
                "Wrote %d examples to %s." % (count, out_fname),
                200)
        return count


def main(_):
  # input_file = nq_data_utils.get_sharded_filename(FLAGS.input_data_dir,
  #                                               FLAGS.split, FLAGS.task_id,
  #                                               FLAGS.shard_split_id,
  #                                               "jsonl.gz")
  # print("Reading file %s", input_file)
  # output_file = nq_data_utils.get_sharded_filename(FLAGS.output_data_dir,
  #                                                FLAGS.split, FLAGS.task_id,
  #                                                FLAGS.shard_split_id,
  #                                                "jsonl")
  # count = nq_jsonl_to_tsv(input_file, output_file)
  # print(count)
  input_data_dir = '/remote/bones/user/vbalacha/fact_augmented_text/fat/fat_bert_nq/generated_files/nq_t5_data_new/'
  stats = {'train':(50,7), 'dev':(4,16)}
  for split in ['train', 'dev']:
    if split == 'train':
      output_file = os.path.join(input_data_dir, "all_train.jsonl")
    else:
      output_file = os.path.join(input_data_dir, "all_dev.jsonl")
    op = open(output_file, 'w')
    count = 0
    for task in range(stats[split][0]):
      for shard_split in range(stats[split][1]):
        input_file = nq_data_utils.get_sharded_filename(input_data_dir,
                                                      split, task,
                                                      shard_split,
                                                      "jsonl")
        if not os.path.exists(input_file):
            continue
        print("Reading file %s", input_file)
        fp = open(input_file)
        for line in fp:
            count += 1
            op.write(line+"\n")
    print(count)

if __name__ == "__main__":
  #flags.mark_flag_as_required("vocab_file")
  tf.app.run()
