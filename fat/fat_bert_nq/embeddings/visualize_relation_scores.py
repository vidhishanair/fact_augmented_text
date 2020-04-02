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
flags.DEFINE_string('apr_files_dir', '', 'Read nq data to extract entities')
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
                              shard_id=FLAGS.shard_split_id)

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

        question_emb = question_embeddings[q_id]
        scores = []
        for rel_id, relation_emb in relation_embeddings.items():
            score = np.dot(question_emb, relation_emb) / (
                    np.linalg.norm(question_emb) *
                    np.linalg.norm(relation_emb))
            rel_name = relations[str(rel2id[rel_id])]['name']
            scores.append((rel_name, score))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[0:20]

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
            rel_name = relations[str(rel2id[rel_id])]['name']
            qrels.append(rel_name)
        qrels = list(set(qrels))
        filtered_relations = [(rel_name, score) for (rel_name,score) in sorted_scores if rel_name in qrels]

        wp.write(str(q_id) + "\t" + question_text + "\t" + str(sorted_scores) + "\t"
                 + str(filtered_relations) + "\n")
