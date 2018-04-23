import time
from sqlalchemy.sql.expression import func
from sqlalchemy import and_
import numpy as np
import sys
import json
import os
import cPickle as pickle
import jellyfish as jf
import argparse

import hiski_models as hiski
import person_matching_model as pmm
from historical_name_normalizer import name_normalizer
name_normalizer.DEBUG = False


def get_person(hiski_id):
    hiski_id = int(hiski_id)
    return hiski.Birth.query.get(hiski_id)


def get_candidate_matches(
        first_name=None, dad_first_name=None, dad_last_name=None,
        mom_first_name=None, mom_last_name=None, parish_id=None, min_year=None,
        max_year=None, max_matches=1000, do_print=True):
    t0 = time.time()
    q = hiski.session.query(hiski.Birth)
    if first_name is not None:
        q = q.filter(hiski.Birth.child_first_nameN == first_name)
    if dad_first_name is not None:
        q = q.filter(hiski.Birth.dad_first_nameN == dad_first_name)
    if dad_last_name is not None:
        q = q.filter(hiski.Birth.dad_last_nameN == dad_last_name)
    if mom_first_name is not None:
        q = q.filter(hiski.Birth.mom_first_nameN == mom_first_name)
    if mom_last_name is not None:
        q = q.filter(hiski.Birth.mom_last_nameN == mom_last_name)
    if min_year is not None:
        q = q.filter(hiski.Birth.birth_year >= min_year)
    if max_year is not None:
        q = q.filter(hiski.Birth.birth_year <= max_year)
    if parish_id is not None:
        q = q.filter(hiski.Birth.parish_id == parish_id)
    if max_matches is not None:
        q = q.limit(max_matches)
    candidates = q.all()
    if do_print:# and dad_first_name is not None:
        #print u"{}, {}, {}, {}, {}, {}".format(first_name, dad_first_name, dad_last_name, min_year, max_year, parish_id)
        print "Retrieving {} candidates took {:.4f} seconds.".format(
            len(candidates), time.time()-t0)
    return candidates


def get_scored_candidate_matches(
        child_id, is_dad, M, first_name_normalizer, ignored_features=None,
        do_print=False, method='classifier'):
    '''
    Find the candidate mothers or fathers of the given child and compute their
    probabilities. Yield (child_id, candidate_id, probability, is_dad, rank)
    tuples where rank is the probability rank among the candidates.
    '''
    ret = []
    p = get_person(child_id)
    birth_year = p.get_year_int()
    if birth_year is None:
        #print "Skipping", p.event_id
        return []

    if is_dad:
        fname = p.dad_first_nameN
        lname = p.dad_last_nameN
        dad_first_name = p.dad_patronymN
    else:
        fname = p.mom_first_nameN
        lname = p.mom_last_nameN
        dad_first_name = p.mom_patronymN
    
    if len(fname) == 0:
        return []
    
    parish_id = None
    if len(lname) == 0:
        # People without a last name.
        lname = ''
        if dad_first_name.endswith('s'):
            dad_first_name = dad_first_name[:-1]
        if dad_first_name.endswith('son'):
            dad_first_name = dad_first_name[:-3]
        dad_first_name = first_name_normalizer.normalize(
                dad_first_name, find_nearest=True, only_first_token=True)
        if len(dad_first_name) == 0:
            return []
        # Limit to the child's birth parish to reduce the number of candidates.
        parish_id = p.parish_id
    else:
        # Don't use patronym if last name is known (to improve recall).
        dad_first_name = None
        
    cands = get_candidate_matches(
            first_name=fname, dad_first_name=dad_first_name,
            dad_last_name=lname, parish_id=parish_id, min_year=birth_year-65,
            max_year=birth_year-10, do_print=do_print)
    # Score candidates.
    probs, no_match_prob = M.get_matching_probabilities(
            p, cands, are_dads=is_dad, ignored_features=ignored_features,
            method=method, debug_print=do_print)
    probs_and_cands = sorted(zip(probs, cands), reverse=True)
    is_dad_str = 'D' if is_dad else 'M'
    for rank, (prob, c) in enumerate(probs_and_cands):
        ret.append((child_id, c.event_id, prob, is_dad_str, rank+1))
    return ret


def edge2str(cand_tuple, use_json=True):
    if not use_json:
        return '{}\n'.format('\t'.join(map(str, cand_tuple)))
    else:
        c = cand_tuple
        d = {"dad": c[3] == 'D', 
             "parent": int(c[1]),
             "child": int(c[0]), 
             "prob": float(c[2]),
             "rank": int(c[4]),
             "selected": int(c[4]) == 1,
            }
        return '{}\n'.format(json.dumps(d))


def process_people(child_ids, output_path):
    n_links = 0
    M = pmm.MatchingModel()
    first_name_normalizer = name_normalizer.NameNormalizer('first')
    with open(output_path, 'w') as fout:
        for i, child_id in enumerate(child_ids):
            do_print = False
            if i % 10 == 0:
                print "\nProcessing child", i
                #do_print = True
            try:
                child_id = int(child_id)
            except:
                print "Bad child id:", child_id
                continue
            for cand_tuple in get_scored_candidate_matches(
                    child_id, is_dad=True, M=M,
                    first_name_normalizer=first_name_normalizer,
                    do_print=do_print):
                fout.write(edge2str(cand_tuple))
                n_links += 1
            for cand_tuple in get_scored_candidate_matches(
                    child_id, is_dad=False, M=M,
                    first_name_normalizer=first_name_normalizer,
                    do_print=do_print):
                fout.write(edge2str(cand_tuple))
                n_links += 1
    print "Wrote {} child-parent links to: {}".format(n_links, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default="birth_event_ids.txt",
                        help=("Path to the event_id file."))
    parser.add_argument("--output", default="reconstructions/all_parent_edges",
                        help=("Prefix of the output file (batch info will be "
                              "appended)."))
    parser.add_argument("--num_batches", type=int, help=(
            "The total number of batches to which the event_ids are divided."))
    parser.add_argument("--batch_index", type=int,
                        help=("Index of the current batch (starting from 1)."))
    args = parser.parse_args()

    assert args.batch_index >= 1, "Batch index has to be a positive integer."
    assert args.batch_index <= args.num_batches, "Batch index cannot exceed the total number of batches."

    # Retrieve the batch of event_ids.
    f = open(args.indices)
    all_event_ids = f.readlines()
    f.close()
    n_ids = len(all_event_ids)
    batch_size = int(np.ceil(n_ids / float(args.num_batches)))
    event_ids = all_event_ids[((args.batch_index - 1) * batch_size):((args.batch_index) * batch_size)]
    
    output_fname = "{}_{}of{}_rows.json".format(args.output, args.batch_index, args.num_batches)
    print "Writing to:", output_fname

    process_people(event_ids, output_fname)
