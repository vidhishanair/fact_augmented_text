import os
import tensorflow as tf
import fat.fat_bert_nq.nq_data_utils as nq_utils

input_data_dir = "/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/shortest_path_fixed_data_repreat_relweight_threehop_only_question_rw20_downweighted_masking_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02"

max_train_tasks = 50
max_shard_splits = 7
mode = "train"

max_dev_tasks = 5
max_dev_splits = 17
mode = "dev"

name_to_features = {
              "unique_ids": tf.FixedLenFeature([], tf.int64),
                    "input_ids": tf.FixedLenFeature([512], tf.int64),
                          "input_mask": tf.FixedLenFeature([512], tf.int64),
                                "segment_ids": tf.FixedLenFeature([512], tf.int64),
                                  }

#instances = []
train_count = 0
for task in range(max_train_tasks):
  for shard in range(max_shard_splits):
    input_file = nq_utils.get_sharded_filename(input_data_dir, "train", task, shard, 'tf-record')
    print("Reading file %s", input_file)
    if not os.path.exists(input_file):
        continue
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
    #instances.extend([
    #    tf.train.Example.FromString(r)
    #    for r in tf.python_io.tf_record_iterator(input_file)
    #])
    dev_count += len([tf.train.Example.FromString(r) for r in tf.python_io.tf_record_iterator(input_file)])

print("Training size: "+str(train_count))
print("Dev size: "+str(dev_count))
