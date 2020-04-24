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

"""This file performs Sling based entity linking on NQ.

The file iterates through entire train and dev set of NQ.
For every example it does entity linking on long answer candidates,
  annotated long and short answer and questiopn.
Every paragraph in the dataset is augmented with an entity map from
  every token to it's entity id.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gzip
import json
import os
import time
# import sling
# import sling.flags as flags
# import sling.task.entity as entity
# import sling.task.workflow as workflow
import tensorflow as tf
from google.cloud import language_v1
from google.cloud.language_v1 import enums

# Calling these 'args' to avoid conflicts with sling flags
args = tf.flags
ARGS = args.FLAGS
#args.DEFINE_string("nq_dir", "/home/vbalacha/datasets/v1.0", "NQ data location")
args.DEFINE_string("nq_dir", "gs://fat_storage/sharded_nq/", "NQ data location")
args.DEFINE_string("output_data_dir", "gs://fat_storage/google_ent_linked_nq/", "Location to write augmented data to")
args.DEFINE_string("fb2wiki", "gs://fat_storage/freebase_wikidata_mappings/fb2wiki.json", "Location to write augmented data to")
args.DEFINE_boolean("annotate_candidates", False, "Flag to annotate candidates")
args.DEFINE_boolean("annotate_long_answers", False,
                    "Flag to annotate long answer")
args.DEFINE_boolean("annotate_short_answers", False,
                    "Flag to annotate short answers")
args.DEFINE_boolean("annotate_question", True, "Flag to annotate questions")

client = language_v1.LanguageServiceClient()

def sample_analyze_entities(text_content, fb2wiki):
    """
    Analyzing Entities in a String

    Args:
      text_content The text content to analyze
    """

    #client = language_v1.LanguageServiceClient()

    # text_content = 'California is a state.'

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}
    entities = []
    entity_map = {}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_entities(document, encoding_type=encoding_type)
    #print(text_content)
    # Loop through entitites returned from the API
    for entity in response.entities:
        #print(u"Representative name for the entity: {}".format(entity.name))
        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        # print(u"Entity type: {}".format(enums.Entity.Type(entity.type).name))
        # Get the salience score associated with the entity in the [0, 1.0] range
        # print(u"Salience score: {}".format(entity.salience))
        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        if entity.metadata['mid'] != '':
            mid = entity.metadata['mid'].strip("/").replace('/','.')
            if mid in fb2wiki:
                wikiids = fb2wiki[mid]
                entities.extend(wikiids)
                #print(mid, wikiids)
                # for metadata_name, metadata_value in entity.metadata.items():
                #     print(u"{}: {}".format(metadata_name, metadata_value))

                # Loop over the mentions of this entity in the input document.
                # The API currently supports proper noun mentions.
                for mention in entity.mentions:
                    # print(u"Mention text: {}".format(mention.text.content))
                    # # Get the mention type, e.g. PROPER for proper noun
                    # print(
                    #     u"Mention type: {}".format(enums.EntityMention.Type(mention.type).name)
                    # )
                    #print(mention.text)
                    mention_text = mention.text.content
                    begin = mention.text.begin_offset
                    end = mention.text.begin_offset + len(mention_text)
                    #print(mention_text, begin, end)
                    if begin in entity_map:
                        entity_map[begin].extend([(end, wid) for wid in wikiids])
                    else:
                        entity_map[begin] = [(end, wid) for wid in wikiids]


    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    # print(u"Language of the text: {}".format(response.language))
    return entities, entity_map


def extract_and_link_text(item, tokens, fb2wiki, should_limit=False):
    """Extracts the tokens in passage, tokenizes them using sling tokenizer."""
    start_token = item["start_token"]
    end_token = item["end_token"]
    if start_token >= 0 and end_token >= 0:
        non_html_tokens = [
            x
            for x in tokens[start_token:end_token]
            if not x["html_token"]
        ]
        if should_limit:
            non_html_tokens = non_html_tokens[:20000]
        answer = " ".join([x["token"] for x in non_html_tokens])
        answer_map = [idx for idx, x in enumerate(non_html_tokens)]
        entities, entity_map = sample_analyze_entities(answer, fb2wiki)
        return answer, answer_map, entities, entity_map
    return "", [], [], {}

def entity_link_nq(nq_data):
    """Parse each paragrapgh in NQ (LA candidate, LA, SA, question).

       Prepare a sling corpus to do entity linking.

    Args:
      nq_data: A python dictionary containint NQ data of 1 train/dev shard
      sling_input_corpus: A filename string to write the sling format documents
          into
    """
    fp = tf.gfile.Open(ARGS.fb2wiki, "r")
    fb2wiki = json.load(fp)
    for i in nq_data.keys():
        tokens = nq_data[i]["document_tokens"]
        if ARGS.annotate_candidates:
            for idx, la_cand in enumerate(nq_data[i]["long_answer_candidates"]):
                answer, answer_map, entities, entity_map = extract_and_link_text(la_cand, tokens, fb2wiki)
                if answer:
                    # nq_data[i]["long_answer_candidates"][idx]["text_answer"] = answer
                    # nq_data[i]["long_answer_candidates"][idx]["answer_map"] = answer_map
                    nq_data[i]["long_answer_candidates"][idx]["google_entity_map"] = entity_map
        if ARGS.annotate_short_answers:
            for idx, ann in enumerate(nq_data[i]["annotations"]):
                short_ans = ann["short_answers"]
                if not short_ans:
                    continue
                for sid in range(len(short_ans)):
                    ans = short_ans[sid]
                    answer, answer_map, entities, entity_map = extract_and_link_text(ans, tokens, fb2wiki)
                    if answer:
                        # nq_data[i]["annotations"][idx]["short_answers"][sid][
                        #     "text_answer"] = answer
                        # nq_data[i]["annotations"][idx]["short_answers"][sid][
                        #     "answer_map"] = answer_map
                        nq_data[i]["annotations"][idx]["short_answers"][sid][
                            "google_entity_map"] = entity_map
        if ARGS.annotate_long_answers:
            for idx, ann in enumerate(nq_data[i]["annotations"]):
                long_ans = ann["long_answer"]
                answer, answer_map, entities, entity_map = extract_and_link_text(long_ans, tokens, fb2wiki)
                if answer:
                    # nq_data[i]["annotations"][idx]["long_answer"]["text_answer"] = answer
                    # nq_data[i]["annotations"][idx]["long_answer"][
                    #     "google_answer_map"] = answer_map
                    nq_data[i]["annotations"][idx]["long_answer"][
                        "google_entity_map"] = entity_map
        if ARGS.annotate_question:
            print(i, nq_data[i]["question_text"])
            question_text = str(nq_data[i]["question_text"].encode('utf-8'))
            entities, entity_map = sample_analyze_entities(question_text, fb2wiki)
            nq_data[i]['google_question_entity_map'] = entity_map
        time.sleep(3)
    return nq_data

def extract_nq_data(nq_file):
    """Read nq shard file and return dict of nq_data."""
    fp = gzip.GzipFile(fileobj=tf.gfile.Open(nq_file, "rb"))
    lines = fp.readlines()
    data = {}
    counter = 0
    for line in lines:
        data[str(counter)] = json.loads(line.decode("utf-8"))
        tok = []
        for j in data[str(counter)]["document_tokens"]:
            tok.append(j["token"])
        data[str(counter)]["full_document_long"] = " ".join(tok)
        counter += 1
    return data


def get_shard(mode, task_id, shard_id):
    return "nq-%s-%02d%02d" % (mode, task_id, shard_id)


def get_full_filename(data_dir, mode, task_id, shard_id):
    return os.path.join(
        data_dir, "%s/%s.jsonl.gz" % (mode, get_shard(mode, task_id, shard_id)))


def get_examples(data_dir, mode, task_id, shard_id):
    """Reads NQ data, does sling entity linking and returns augmented data."""
    file_path = get_full_filename(data_dir, mode, task_id, shard_id)
    tf.logging.info("Reading file: %s" % (file_path))
    # if not os.path.exists(file_path):
    #     tf.logging.info("Path doesn't exist")
    #     return None
    nq_data = extract_nq_data(file_path)
    tf.logging.info("NQ data Size: " + str(len(nq_data.keys())))

    tf.logging.info("Performing entity extraction")
    fact_extracted_data = entity_link_nq(nq_data)
    return fact_extracted_data


def main(_):
    # workflow.startup()
    # max_tasks = {"train": 50, "dev": 5}
    # max_shards = {"train": 7, "dev": 17}
    max_tasks = {"train": 1, "dev": 5}
    max_shards = {"train": 1, "dev": 17}
    for mode in ["train"]:
        # Parse all shards in each mode
        # Currently sequentially, can be parallelized later
        for task_id in range(0, max_tasks[mode]):
            for shard_id in range(0, max_shards[mode]):
                nq_augmented_data = get_examples(ARGS.nq_dir, mode, task_id, shard_id)
                if nq_augmented_data is None:
                    continue
                path = get_full_filename(ARGS.output_data_dir, mode, task_id, shard_id)
                with gzip.GzipFile(fileobj=tf.gfile.Open(path, "w")) as output_file:
                    for idx in nq_augmented_data.keys():
                        json_line = nq_augmented_data[idx]
                        output_file.write((json.dumps(json_line) + "\n").encode('utf-8'))
    # workflow.shutdown()


if __name__ == "__main__":
    # This will fail if non-sling CMDLine Args are given.
    # Will modify sling separately to parse known args
    tf.app.run()
