import json
import tensorflow as tf
from bert import tokenization

file_path = '/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/shortest_path_downweighted_masking_anonymized_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02/dev/nq-dev-0000.tf-record'
vocab_path = '/remote/bones/user/vbalacha/bert-joint-baseline/vocab-nq.txt'
data = [ r  for r in tf.python_io.tf_record_iterator(path=file_path)]
fp = open('/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/shortest_path_downweighted_masking_anonymized_sharded_kb_data_mc48_alpha0.75_mseq512_unk0.02/dev_analysis.tsv', 'w')

tokenizer = tokenization.FullTokenizer(vocab_file = vocab_path, do_lower_case=True)
counter = 0
for r in data :
    counter+=1
    ex = tf.train.Example()
    ex.ParseFromString(r)
    text = []
    facts = []
    question = []
    current = 'question'
    input_ids = list(ex.features.feature['input_ids'].int64_list.value)
    input_ids = map(int, input_ids)
    for token in input_ids:
        try:
            word = tokenizer.convert_ids_to_tokens([token])[0]
            if current == 'question':
                question.append(word)
            elif current == 'text':
                text.append(word)
            elif current == 'facts':
                facts.append(word)
            else:
                print(current)
                exit()
            if word == '[SEP]' and current == 'question' : 
                current = 'text'
            elif word == '[SEP]' and current == 'text':
                current = 'facts'
            else:
                continue
        except:
            print('didnt tokenize')
    fp.write(" ".join(question).replace(" ##","")+"\t")
    fp.write(" ".join(text).replace(" ##", "")+"\t")
    fp.write(" ".join(facts).replace(" ##","")+"\t")

    text_only_input_ids = list(ex.features.feature['text_only_input_ids'].int64_list.value)
    text_only_input_ids = map(int, text_only_input_ids)
    text_only_words = tokenizer.convert_ids_to_tokens(text_only_input_ids)
    fp.write(" ".join(text_only_words).replace(" ##","")+"\t")

    masked_text_tokens_input_ids = list(ex.features.feature['masked_text_tokens_input_ids'].int64_list.value)
    masked_text_tokens_input_ids = map(int, masked_text_tokens_input_ids)
    masked_text_text_tokens = tokenizer.convert_ids_to_tokens(masked_text_tokens_input_ids)
    fp.write(" ".join(masked_text_text_tokens).replace(" ##","")+"\t")

    anonymized_text_only_tokens_input_ids = list(ex.features.feature['anonymized_text_only_tokens_input_ids'].int64_list.value)
    anonymized_text_only_tokens_input_ids = map(int, anonymized_text_only_tokens_input_ids)
    anonymized_text_only_tokens = tokenizer.convert_ids_to_tokens(anonymized_text_only_tokens_input_ids)
    fp.write(" ".join(anonymized_text_only_tokens).replace(" ##","")+"\n")
     
