import math

import random
import torch
import numpy as np
from pytorch_transformers import *
# Load pre-trained model (weights)
#model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
#model.eval()
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

#pretrained = 'bert-large-uncased'
pretrained = 'bert-large-uncased-whole-word-masking'

tokenizer = BertTokenizer.from_pretrained(pretrained)
model = BertForMaskedLM.from_pretrained(pretrained)

def score(sentence, fac_inp):
    #tokenize_input = tokenizer.tokenize(sentence)
    #tokenize_input = sentence
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(sentence)])
    #label = torch.tensor([tokenizer.convert_tokens_to_ids(sentence)])
    #label_matrix = label.repeat([len(sentence), 1])
    #print(label_matrix.size())

    ##tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(sentence)])
    ##input_matrix = tensor_input.unsqueeze(0).repeat(len(sentence), -1)
    #tensor_list = []
    #for i in range(0, len(sentence)):
    #    tok_inp = sentence
    #    tok_inp[i] = '[MASK]'
    #    ids = [tokenizer.convert_tokens_to_ids(tok_inp)]
    #    lab = [-1]*len(sentence)
    #    lab[i] = 
    #    mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tok_inp)])
    #    tensor_list.append(mask_input.squeeze())
    #tensor_input = torch.stack(tensor_list, dim=0)
    sentence_loss=0.
    sentence_fact_loss=0.
    #print(tensor_input.size())
    #sentence_loss=model(tensor_input, masked_lm_labels=label_matrix).data.numpy()
    #print('Here')
    list1 = [i for i in range(0, len(sentence))]
    subset = random.sample(list1, math.ceil(0.15*len(sentence)))
    cnt = 0
    #print(" ".join(sentence))
    #if fac_inp is not None: print(" ".join(fac_inp))
    for i in range(0, len(sentence)):
        if i not in subset:
            continue
        cnt += 1
        tok_inp = sentence.copy()
        tok_inp[i]='[MASK]'
        #print(tok_inp)
        ids = [tokenizer.convert_tokens_to_ids(tok_inp)]
        plain_ids = [tokenizer.convert_tokens_to_ids(sentence)]
        lab = [-1]*len(sentence)
        lab[i] = plain_ids[0][i]
        #print(lab)
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tok_inp)])
        labels = torch.tensor([lab])
        word_loss=model(mask_input, masked_lm_labels=labels)[0] #.data.numpy()
        sentence_loss +=word_loss
        
        if fac_inp is not None:
            ids = [tokenizer.convert_tokens_to_ids(tok_inp+fac_inp)]
            lab = [-1]*len(ids[0])
            lab[i] = plain_ids[0][i]
            mask_input = torch.tensor(ids)
            labels = torch.tensor([lab])
            word_loss=model(mask_input, masked_lm_labels=labels)[0]
            sentence_fact_loss+=word_loss
        
	#print("Word: %s : %f"%(word, np.exp(-word_loss)))
    #print('Here')
    #print(cnt)
    return math.exp(sentence_loss/cnt), math.exp(sentence_fact_loss/cnt)


#def score(sentence):
#    #tokenize_input = tokenizer.tokenize(sentence)
#    tokenize_input = sentence
#    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#    loss=model(tensor_input[:-1], masked_lm_labels=tensor_input[1:])[0]
#    return math.exp(loss)

ftname = '/remote/bones/user/vbalacha/google-research/fat/dev_context_text_facts.txt'
fname = '/remote/bones/user/vbalacha/google-research/fat/dev_context_text.txt'
#fp = open(fname, 'r')
#data = [x.split(' ') for x in fp.readlines()]
#print('No: of examples: '+str(len(data)))
##a=['there is a book on the desk',
##   'there is a plane on the desk',
##   'there is a book in the desk']
#ppl_scores = [score(i, None)[0] for i in data[0:500]]
#print('Avg ppl for text: '+str(float(sum(ppl_scores))/len(ppl_scores)))
##
#
#fp = open(ftname, 'r')
#data = [x.split(' ') for x in fp.readlines()]
#print('No: of examples: '+str(len(data)))
#ppl_scores = [score(i, None)[0] for i in data[0:500]]
#print('Avg ppl for text+facts: '+str(float(sum(ppl_scores))/len(ppl_scores)))


fp = open(ftname, 'r')
data = [x.split(' ') for x in fp.readlines()]
fact_data = []
text_data = []
for x in data[0:100]:
    fact_data.append([])
    text_data.append([])
    occ = 0
    for y in x :
        if y == '[SEP]':
            occ+=1
        if occ > 1:
            fact_data[-1].append(y)
        else:
            text_data[-1].append(y)

#for x in text_data[]
#    size = len(x)

#ppl_scores = [score(i, None)[0] for i in fact_data[0:500]]
#print('Avg ppl for facts: '+str(float(sum(ppl_scores))/len(ppl_scores)))

#ppl_scores = [score(i, j) for (i, j) in list(zip(text_data, fact_data))[0:500]]
d = [(i,j) for (i, j) in list(zip(text_data, fact_data))[0:40]]
ppl_scores = [score(i, d[np.random.randint(0, high=len(d), size=1)[0]][1]) for (i, j) in d]

txt_ppl = [x[0] for x in ppl_scores]
txtfct_ppl = [x[1] for x in ppl_scores]

print('Avg ppl for text on text_part: '+str(float(sum(txt_ppl))/len(txt_ppl)))
print('Avg ppl for text+facts on text_part: '+str(float(sum(txtfct_ppl))/len(txtfct_ppl)))

ppl_scores = [score(i, j) for (i, j) in d]

txt_ppl = [x[0] for x in ppl_scores]
txtfct_ppl = [x[1] for x in ppl_scores]

print('Avg ppl for text on text_part: '+str(float(sum(txt_ppl))/len(txt_ppl)))
print('Avg ppl for text+facts on text_part: '+str(float(sum(txtfct_ppl))/len(txtfct_ppl)))

#ppl_scores = [score(i, None) for i in fact_data[0:500]]
#print('Avg ppl for facts: '+str(float(sum(ppl_scores))/len(ppl_scores)))

