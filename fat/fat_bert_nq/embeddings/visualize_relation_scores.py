#import cPickle as pkl
import pickle as pkl
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
import gzip
from fat.fat_bert_nq import nq_data_utils
from fat.fat_bert_nq.ppr.apr_lib import ApproximatePageRank

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('nq_dir', '/remote/bones/user/vbalacha/datasets/ent_linked_nq_new/', 'Read nq data to extract entities')
#flags.DEFINE_string('apr_files_dir', '', 'Read nq data to extract entities')
flags.DEFINE_string('question_emb_file', '', 'Read nq data to extract entities')
flags.DEFINE_string('relation_emb_file', '', 'Read nq data to extract entities')
flags.DEFINE_string('relations_file', None, 'input relations dict')
flags.DEFINE_string('rel2id_file', None, 'input relations dict')
flags.DEFINE_string('output_file', None, '')

flags.DEFINE_integer("shard_id", None,
                     "Train and dev shard to read from and write to.")
flags.DEFINE_string(
    "mode", "train",
    "Train and dev split to read from and write to. Accepted values: ['train', 'dev', 'test']"
)

input_file = nq_data_utils.get_sharded_filename(FLAGS.nq_dir, FLAGS.mode, FLAGS.task_id, FLAGS.shard_id, 'jsonl.gz')
embeddings_file = "/remote/bones/user/vbalacha/datasets/glove/glove.6B.300d.txt"
with gzip.GzipFile(fileobj=tf.gfile.Open(FLAGS.rel2id_file, 'rb')) as op4:
    rel2id = json.load(op4)
    op4.close()
id2rel = {str(idx): ent for ent, idx in rel2id.items()}
dim = 300

apr_obj = ApproximatePageRank(mode='train', task_id=FLAGS.task_id,
                              shard_id=FLAGS.shard_id)

def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
            a["long_answer"]["end_token"] >= 0)

def get_first_annotation_answer_entities(e):
    """Returns the first short or long answer in the example.

    Args:
      e: (dict) annotated example.

    Returns:
      annotation: (dict) selected annotation
      annotated_idx: (int) index of the first annotated candidate.
      annotated_sa: (tuple) char offset of the start and end token
          of the short answer. The end token is exclusive.
    """
    positive_annotations = sorted(
        [a for a in e["annotations"] if has_long_answer(a)],
        key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]

            # entities from the entire SA span (cross check if this makes sense)
            answer_entities = set()
            for sa in a['short_answers']:
                for start_idx in sa['entity_map'].keys():
                    for sub_span in sa['entity_map'][start_idx]:
                        answer_entities.add(sub_span[1])

            return list(answer_entities)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return []

    return []

question_embeddings = pkl.load(open(FLAGS.question_emb_file, 'rb'))
relation_embeddings = pkl.load(open(FLAGS.relation_emb_file, 'rb'))

with gzip.GzipFile(fileobj=tf.gfile.Open(FLAGS.relations_file, 'rb')) as op4:
    obj = json.load(op4)
    op4.close()
    relations = obj['r']
    # for rel_id, val in relations.items():
    #     rel_name = val['name']
    #     rel = id2rel[rel_id]
wp = open(FLAGS.output_file, 'w')
with gzip.GzipFile(fileobj=tf.gfile.Open(input_file, "rb")) as fp:
    count = 0
    for line in fp:
        count += 1
        if count == 100:
            break
        data = json.loads(line)
        q_id, question_text = data["example_id"], data["question_text"]
        question_entity_map = data["question_entity_map"]

        question_entities = set()
        for start_idx in question_entity_map.keys():
            for sub_span in question_entity_map[start_idx]:
                question_entities.add(sub_span[1])
        question_entities = list(question_entities)
        question_entity_ids = [int(apr_obj.data.ent2id[x]) for x in question_entities if x in apr_obj.data.ent2id]
        question_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids])

        question_emb = question_embeddings[q_id]
        scores = []
        for rel_id, relation_emb in relation_embeddings.items():
            score = np.dot(question_emb, relation_emb) / (
                    np.linalg.norm(question_emb) *
                    np.linalg.norm(relation_emb))
            rel_name = relations[str(rel2id[rel_id])]['name']
            scores.append((rel_name, score))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_score_dict = {rel_name: score for (rel_name, score) in sorted_scores}

        submat = apr_obj.data.adj_mat_t_csr[:, question_entity_ids]
        # Extracting non-zero entity pairs
        row, col = submat.nonzero()
        # print(row)
        # print(col)
        qrels = []
        for ii in range(row.shape[0]):
            obj_id = row[ii]
            subj_id = question_entity_ids[col[ii]]
            rel_id = apr_obj.data.rel_dict[(subj_id, obj_id)]
            rel_name = relations[str(rel_id)]['name']
            qrels.append(rel_name)
        qrels = list(set(qrels))
        filtered_relations = [(rel_name, sorted_score_dict[rel_name]) for rel_name in qrels]
        sorted_filtered_relations = sorted(filtered_relations, key=lambda x: x[1], reverse=True)

        answer_entities = get_first_annotation_answer_entities(data)
        facts, num_hops = apr_obj.get_shortest_path_facts(question_entities, answer_entities, passage_entities=[], seed_weighting=True, fp=fp)
        nl_facts = " . ".join([
            str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
            for x in facts
        ])
        if len(facts) > 0:
            wp.write(str(q_id) + "\t" + question_text + "\t" + question_entity_names
                     + "\t" + str(qrels) + "\t" + str(nl_facts) + "\t"
                     + str(sorted_filtered_relations) + "\t" + str(sorted_scores[0:20]) + "\n")
