import json
from nltk.tokenize import sent_tokenize


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {} {}...".format(len(obj), message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def tokenize_context(context):
    """
    tokenize context and assign index [(start_index, end_index), sentence]
    :param context: squad context (passage)
    :return: pair of (start, end) index and sentence
    """
    # context_sents = sent_detector.tokenize(context)
    context_sents = sent_tokenize(context)
    start_idx = 0
    result = []
    for sent in context_sents:
        end_idx = start_idx + len(sent)
        result.append([(start_idx, end_idx), sent.strip()])
        start_idx = end_idx + 1
    return result


def has_answer(indices, sent, answers):
    for ans in answers:
        text = ans['text']
        answer_start = ans['answer_start']
        if indices[0] <= answer_start <= indices[1]:
            if sent.find(text) == -1:
                print(text, ",", sent)
            return True


def main():

    with open("dev-v2.0.json", "r") as f:
        source = json.load(f)

    data = source["data"]

    parsed = []
    n_sent_sum = 0
    for article in data:
        paragraphs = article['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context_sents = tokenize_context(context)
            n_sent_sum += len(context_sents)
            qas = paragraph['qas']
            for qa in qas:
                _id = qa['id']
                question = qa['question'].strip()
                answers = qa['answers']
                if qa['is_impossible'] is False:
                    for i, (indices, sent) in enumerate(context_sents):
                        __id = _id + "_" + str(i)
                        label = 'entailment' if has_answer(indices, sent, answers) else 'not_entailment'
                        parsed.append("{}\t{}\t{}\t{}\n".format(__id, question, sent, label))

    # stat
    print("total: ", len(parsed))
    print("avg: ", len(parsed)/n_sent_sum)

    with open("dev.tsv", "w", encoding='utf-8') as fw:
        fw.write("\t".join(["index", "question", "sentence", "label"]) + "\n")
        for d in parsed:
            fw.write(d)


if __name__ == "__main__":
    main()
