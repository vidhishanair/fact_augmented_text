import os
import tensorflow as tf
import fat.fat_bert_nq.nq_data_utils as nq_utils

#input_data_dir = "/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/question_seeded_sharded_kb_data_alpha0.75_mc48_mseq512_unk0.02"
input_data_dir = "/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/shortest_path_relweight_threehop_shuffle_add_question_rw20_downweighted_masking_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02"
#input_data_dir = "/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/tmpdir_unk-1_/"

#input_data_dir = "/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_new_kb_data_mc512_unk0.1_test"
#input_data_dir = "/remote/bones/user/vbalacha/language/language/question_answering/bert_joint/generated_files/baseline_seq512_unk0.02/train"

max_train_tasks = 50
max_shard_splits = 7
mode = "train"

max_dev_tasks = 5
max_dev_splits = 17
mode = "dev"
fp = open("/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/shortest_path_relweight_threehop_shuffle_add_question_rw20_downweighted_masking_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02/unique_id_list_0000.txt", "w")
#fp = open("/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/tmpdir_unk-1_/unique_id_list_0000.txt", "w")
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
#for task in [0]:
#  for shard in [0]:
    input_file = nq_utils.get_sharded_filename(input_data_dir, "train", task, shard, 'tf-record')
    print("Reading file %s", input_file)
    if not os.path.exists(input_file):
        continue
    #instances.extend([
    #    tf.train.Example.FromString(r)
    #    for r in tf.python_io.tf_record_iterator(input_file)
    #])
    #example = tf.parse_single_example(record, name_to_features)
    for record in tf.python_io.tf_record_iterator(input_file):
      #example = tf.parse_single_example(record, name_to_features)
      example = tf.train.Example.FromString(record)
      unique_id = example.features.feature['unique_ids'].int64_list.value[0]
      fp.write(str(unique_id)+"\n")
#print("Training size: "+str(count))
