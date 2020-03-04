import os

#input_data_dir = '/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_mc48_mseq512_unk1.0/pretrain/train/'
input_data_dir = '/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_non_tokenized_mc48_mseq512_unk0.5/pretrain/train/'
max_train_tasks = 50
max_shard_splits = 7
mode = "train"

count=0
for task in range(max_train_tasks):
  for shard in range(max_shard_splits):
    input_file = os.path.join(input_data_dir, 'nq-train-%02d%02d.txt'%(task, shard))
    print("Reading file %s", input_file)
    if not os.path.exists(input_file):
        continue
    data = open(input_file).readlines()
    for line in data:
        if line == '\n':
            count += 1
print("Training size: "+str(count))

