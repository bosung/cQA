import xml.etree.ElementTree as ET
import json

# xml_file = "../data/semeval/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml"
# xml_file = "../data/semeval/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml"
# xml_file = "../data/semeval/SemEval2016-task3-English-test-subtaskA.xml"
xml_file = "../data/semeval/SemEval2016-Task3-CQA-QL-test-subtaskA.xml"

tree = ET.parse(xml_file)
root = tree.getroot()

data = []

"""
for org in root:
    org_q_subject, org_q_body, thread = org
    org_q = org_q_body.text

    rel = thread[0]
    rel_q = rel[1].text  # rel[1].tag = 'RelQBody'
    if rel_q is None:
        rel_q = rel[0].text  # data bug. RelQBody was empty. replace with RelQSubject

    data.append({"guid": rel.attrib['RELQ_ID'], "question": org_q, "sentence": rel_q,
                 "label": rel.attrib['RELQ_RELEVANCE2ORGQ']})

    for comment in thread[1:]:
        _id = comment.attrib['RELC_ID']
        comm2org_label = comment.attrib['RELC_RELEVANCE2ORGQ']
        comm2rel_label = comment.attrib['RELC_RELEVANCE2RELQ']
        rel_c = comment[0].text
        data.append({"guid": _id+"_ORG", "question": org_q, "sentence": rel_c, "label": comm2org_label})
        data.append({"guid": _id+"_REL", "question": rel_q, "sentence": rel_c, "label": comm2rel_label})
"""
good = 0
bad = 0

for threads in root:
    rel = threads[0]
    rel_q = rel[1].text  # rel[1].tag = 'RelQBody'
    if rel_q is None:
        rel_q = rel[0].text  # data bug. RelQBody was empty. replace with RelQSubject

    for comment in threads[1:]:
        _id = comment.attrib['RELC_ID']
        comm2rel_label = comment.attrib['RELC_RELEVANCE2RELQ']
        rel_c = comment[0].text
        data.append("{}\t{}\t{}\t{}\n".format(_id, rel_q, rel_c, comm2rel_label))
        if comm2rel_label == "Good":
            good += 1
        else:
            bad += 1

print(len(data))
print(good, bad, good+bad)
# json.dump(data, open("semeval-2016-task3-subtaskA-train2.json", "w"))

with open("../data/semeval/test.tsv", "w", encoding='utf-8') as fw:
    fw.write("\t".join(["index", "question", "sentence", "label"]) + "\n")
    for d in data:
        fw.write(d)



