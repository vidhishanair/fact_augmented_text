import os
import tensorflow as tf
import fat.fat_bert_nq.nq_data_utils as nq_utils
tf.compat.v1.enable_eager_execution()
#input_data_dir = "/remote/bones/user/vbalacha/fact_augmented_text/fat/fat_bert_nq/generated_files/shortest_path_fixed_data_threehop_only_question_rw20_downweighted_masking_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02/"
input_data_dir = "/remote/bones/user/vbalacha/fact_augmented_text/fat/fat_bert_nq/generated_files/relation_classifier_data/qrel_eq_neg_wintrelid/"

max_train_tasks = 50
#max_train_tasks = 1
max_shard_splits = 7
#max_shard_splits = 1
mode = "train"

max_dev_tasks = 5
max_dev_splits = 17
mode = "dev"

name_to_features = {
              "unique_ids": tf.FixedLenFeature([], tf.int64),
              "example_ids": tf.FixedLenFeature([], tf.int64),
              "answer_label": tf.FixedLenFeature([], tf.int64),
              "relation_id": tf.FixedLenFeature([], tf.int64),
              "input_ids": tf.FixedLenFeature([512], tf.int64),
              "input_mask": tf.FixedLenFeature([512], tf.int64),
              "segment_ids": tf.FixedLenFeature([512], tf.int64),}

#instances = []
train_count = 0
for task in range(max_train_tasks):
  for shard in range(max_shard_splits):
    input_file = nq_utils.get_sharded_filename(input_data_dir, "train", task, shard, 'tf-record')
    print("Reading file %s", input_file)
    if not os.path.exists(input_file):
        continue
    # for record in tf.python_io.tf_record_iterator(input_file):
    #     example = tf.parse_single_example(record, name_to_features)
    #     #example = tf.train.Example.FromString(record)
    #     #print(example)
    #     relation_id = example["relation_id"]
    #     print(tf.get_static_value(relation_id).decode('utf-8'))
    #     #print(list(map(bytes, relation_id)))
    #instances.extend([
    #    tf.train.Example.FromString(r)
    #    for r in tf.python_io.tf_record_iterator(input_file)
    #])
    train_count += len([tf.train.Example.FromString(r) for r in tf.python_io.tf_record_iterator(input_file)])

#print("Training size: "+str(count))


dev_count = 0
for task in range(max_dev_tasks):
  for shard in range(max_dev_splits):
    input_file = nq_utils.get_sharded_filename(input_data_dir, "dev", task, shard, 'tf-record')
    print("Reading file %s", input_file)
    if not os.path.exists(input_file):
        continue
    for record in tf.python_io.tf_record_iterator(input_file):
        example = tf.parse_single_example(record, name_to_features)
        #example = tf.train.Example.FromString(record)
        #print(example)
        relation_id = example["relation_id"]
        example_id = example["example_ids"]
        print(example_id)
    #instances.extend([
    #    tf.train.Example.FromString(r)
    #    for r in tf.python_io.tf_record_iterator(input_file)
    #])
    dev_count += len([tf.train.Example.FromString(r) for r in tf.python_io.tf_record_iterator(input_file)])

print("Training size: "+str(train_count))
print("Dev size: "+str(dev_count))
