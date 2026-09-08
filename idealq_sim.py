""" Compute similarity / overlap between ideal queries
Keep 3 params: 1: dataset 2: method-1 (LS-variants) and 3: dfo"""
import os

from io_utils import parse_queries
import numpy as np

from iqg_learn import IdealQueryGeneration
import sys
# dataset = "msmarco_passage"

# dataset = "msmarco_passage_v2"
dataset = sys.argv[1]

# name1 = "leastsq_1"
# name2 ="dfo"
name1 = sys.argv[2]
name2 = sys.argv[3]

ideal_query_dir = f"./ideal_queries/{dataset}"

def get_angle(x, y):
    cos = cosine(x, y)
    return np.arccos(cos) * 180 / np.pi

def cosine(x, y):
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx == 0 or normy == 0:
        return 0
    return max(min(np.dot(x, y) / normx / normy, 1), -1)

intersections = []
cosine_intersections = []
for qid in os.listdir(f"{ideal_query_dir}/{name1}"):
    qvec1 = parse_queries(f"{ideal_query_dir}/{name1}/{qid}")[qid]
    qvec2 = parse_queries(f"{ideal_query_dir}/{name2}/{qid}")[qid]
    # term_list = list(qvec1.keys() | qvec2.keys())
    # qarr1 = IdealQueryGeneration.qvec_to_array(qvec1, term_list)
    # qarr2 = IdealQueryGeneration.qvec_to_array(qvec2, term_list)
    # jaccard = len(qvec1.keys() & qvec2.keys()) / len(term_list)
    # print(f"{qid}\t{cosine(qarr1, qarr2)}\t{get_angle(qarr1, qarr2)}\t{jaccard}\t{len(qvec1.keys() & qvec2.keys())}")

    qvec1.sort_by_weight()
    qvec1.trim(200)
    term_list = list(qvec1.keys() | qvec2.keys())
    qarr1 = IdealQueryGeneration.qvec_to_array(qvec1, term_list)
    qarr2 = IdealQueryGeneration.qvec_to_array(qvec2, term_list)
    jaccard = len(qvec1.keys() & qvec2.keys()) / len(term_list)
    intersection = len(qvec1.keys() & qvec2.keys())
    print(f"{qid}\t{cosine(qarr1, qarr2)}\t{get_angle(qarr1, qarr2)}\t{jaccard}\t{intersection}")
    intersections.append(intersection)
    cosine_intersections.append(cosine(qarr1, qarr2))

print(f"{np.mean(intersections):.4f}")
print(f"{np.mean(cosine_intersections):.4f}, #of elements in {len(cosine_intersections)}")
