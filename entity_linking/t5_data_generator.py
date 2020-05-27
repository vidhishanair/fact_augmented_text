import tf
import os
import gzip
import json


# Public directory of Natural Questions data on GCS.
NQ_JSONL_DIR = "gs://natural_questions/v1.0-simplified/"
NQ_SPLIT_FNAMES = {
    "train": "simplified-nq-train.jsonl.gz",
    "validation": "nq-dev-all.jsonl.gz"
}
nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "nq-train.tsv"),
    "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
}

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
    with tf.io.gfile.GFile(in_fname, "rb") as infile, \
            tf.io.gfile.GFile(out_fname, "w") as outfile:
        for line in gzip.open(infile):
            ex = json.loads(line)
            # Remove any examples with more than one answer.
            if len(ex['annotations'][0]['short_answers']) != 1:
                continue
            # Questions in NQ do not include a question mark.
            question = ex["question_text"] + "?"
            answer_span = ex['annotations'][0]['short_answers'][0]

            # Handle the two document formats in NQ (tokens or text).
            if "document_tokens" in ex:
                tokens = [t["token"] for t in ex["document_tokens"]]
            elif "document_text" in ex:
                tokens = ex["document_text"].split(" ")

            answer = extract_answer(tokens, answer_span)
            # Write this line as <question>\t<answer>
            outfile.write("%s\t%s\n" % (question, answer))
            count += 1
            tf.logging.log_every_n(
                tf.logging.INFO,
                "Wrote %d examples to %s." % (count, out_fname),
                1000)
        return count

if tf.io.gfile.exists(nq_counts_path):
    # Used cached data and counts.
    tf.logging.info("Loading NQ from cache.")
    num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
else:
    # Create TSVs and get counts.
    tf.logging.info("Generating NQ TSVs.")
    num_nq_examples = {}
    for split, fname in NQ_SPLIT_FNAMES.items():
        num_nq_examples[split] = nq_jsonl_to_tsv(
            os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
    json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))
