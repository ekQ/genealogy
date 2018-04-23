import hiski_models as hiski
import time
import numpy as np
import sys
import json
import os
import cPickle as pickle
import jellyfish as jf
import argparse
import datetime as dt
from dateutil.relativedelta import relativedelta

import compute_matches


def date_int(d):
    try:
        d = int(d)
    except:
        d = 0
    return d

def parse_year_and_date(year1, year2, month1, month2,
                        day1, day2, debug=0):
    year1 = date_int(year1)
    year2 = date_int(year2)
    month1 = date_int(month1)
    month2 = date_int(month2)
    day1 = date_int(day1)
    day2 = date_int(day2)
    if year1 > 0:
        year = year1
    elif year2 > 0:
        year = year2
    else:
        return None, None
    if month1 > 0:
        month = month1
        if day1 > 0:
            day = day1
        else:
            day = 1
    elif month2 > 0:
        month = month2
        if day2 > 0:
            day = day2
        else:
            day = 1
    else:
        return year, None
    try:
        t = dt.datetime(year, month, day)
        return year, t
    except:
        if debug > 0:
            print "Bad date: {}-{}-{}".format(year, month, day)
        return year, None

    
def subtract_time_delta(t, year_delta, month_delta, week_delta, day_delta):
    year_delta = date_int(year_delta)
    month_delta = date_int(month_delta)
    week_delta = date_int(week_delta)
    day_delta = date_int(day_delta)
    if (year_delta == 0 and month_delta == 0 and
        week_delta == 0 and day_delta == 0):
        return None, None
    elif ((month_delta == 0 and week_delta == 0 and day_delta == 0) or
        t[1] is None):
        return t[0] - year_delta, None
    t_delta = relativedelta(years=year_delta, months=month_delta,
                                weeks=week_delta, days=day_delta)
    t2 = t[1] - t_delta
    return t2.year, t2


def compute_time_error(t1, t2):
    if t1[0] is None or t2[0] is None:
        year_err = None
    else:
        year_err = t1[0] - t2[0]
    if t1[1] is None or t2[1] is None:
        dt_err = None
    else:
        dt_err = (t1[1] - t2[1]).days
    return year_err, dt_err

filter_counts = [0] * 8

def get_matches(event_id, parish_dist_map, max_year_error=2,
                max_day_error=40, max_parish_dist=60,
                use_patronym_filtering=False, do_print=True):
    global filter_counts
    t0 = time.time()
    Bu = hiski.Burial.query.get(int(event_id))
    t_bu = parse_year_and_date(
            Bu.death_year, Bu.burial_year, Bu.death_month,
            Bu.burial_month, Bu.death_day, Bu.burial_day)
    if t_bu[0] is None:
        if do_print:
            print "Invalid death/burial year for:", event_id
        return []
    t_bi_est = subtract_time_delta(t_bu, Bu.age_years, Bu.age_months,
                                   Bu.age_weeks, Bu.age_days)
    if t_bi_est[0] is None:
        if do_print:
            print "Invalid death age for:", event_id
        return []
    candidates = compute_matches.get_candidate_matches(
        first_name=Bu.first_nameN, dad_last_name=Bu.last_nameN,
        min_year=t_bi_est[0]-max_year_error, max_year=t_bi_est[0]+max_year_error,
        do_print=False)
    # Filter out some of the candidates based on dates, location,
    # and patronym.
    filtered_births = []
    for Bi in candidates:
        t_bi = parse_year_and_date(
                Bi.birth_year, Bi.baptism_year, Bi.birth_month,
                Bi.baptism_month, Bi.birth_day, Bi.baptism_day)
        t_err = compute_time_error(t_bi, t_bi_est)
        # Check the time constraints.
        if (len(candidates) > 1 and t_err[0] is not None and t_err[1] is None and
                np.abs(t_err[0]) > max_year_error):
            filter_counts[0] += 1
            continue
        #if (Bu.age_days > 0 and Bu.age_years < 2 and t_err[1] is not None and
        #    np.abs(t_err[1]) > 7):
        #    filter_counts[1] += 1
        #    continue
        if (len(candidates) > 1 and t_err[1] is not None and
            np.abs(t_err[1]) > max_day_error):
            filter_counts[5] += 1
            continue
        if t_err[0] is None:
            filter_counts[2] += 1
            continue
        # Check the location constraint.
        parish_distance = parish_dist_map.get((Bi.parish_id, Bu.parish_id), None)
        if len(candidates) > 1 and (parish_distance is None or parish_distance > max_parish_dist):
            filter_counts[3] += 1
            continue
        if (len(candidates) > 1 and use_patronym_filtering and len(Bu.patronymN) > 1 and
                len(Bi.dad_first_nameN) > 1 and
                jf.jaro_winkler(Bu.patronymN, Bi.dad_first_nameN) < 0.8):
            filter_counts[4] += 1
            continue
        #if (len(Bu.first_name) > 1 and len(Bi.child_first_name) > 1 and
        #    jf.jaro_winkler(clean_name(Bu.first_name), clean_name(Bi.child_first_name)) < 0.95):
        #    filter_counts[5] += 1
        #    continue
        filtered_births.append(Bi.event_id)
    if do_print:
        print "Retrieved {} and kept {} candidates in {:.4f} seconds.".format(
            len(candidates), len(filtered_births), time.time()-t0)
    return filtered_births


def process_people(burial_ids, output_path):
    import parish_distances
    parish_dist = parish_distances.get_distance_map()
    match_counts = []
    n_links = 0
    with open(output_path, 'w') as fout:
        for i, burial_id in enumerate(burial_ids):
            do_print = False
            if i % 1000 == 0:
                print "\nProcessing burial", i
                do_print = True
            else:
                do_print = False
            try:
                burial_id = int(burial_id)
            except:
                print "Bad burial id:", burial_id
                continue
                
            matches = get_matches(
                    burial_id, parish_dist, max_year_error=1, max_day_error=61,
                    max_parish_dist=60, use_patronym_filtering=True, do_print=do_print)
            if len(matches) == 1:
                fout.write('{} {}\n'.format(burial_id, matches[0]))                
                n_links += 1
            match_counts.append(len(matches))
    match_counts = np.asarray(match_counts)
    print "Zero matches:", np.mean(match_counts == 0)
    print "One match:", np.mean(match_counts == 1)
    print "Multiple matches:", np.mean(match_counts > 1)
    print "Wrote {} burial-birth links to: {}".format(n_links, output_path)


def write_links_to_db(link_fname):
    burial2birth = {}
    birth2burial = {}
    f = open(link_fname)
    for line in f:
        burial_id, birth_id = line.strip().split()
        birth_id = int(birth_id)
        burial_id = int(burial_id)
        if burial_id in burial2birth:
            print "Burial {} assigned to {} and {}".format(
                    burial_id, birth_id, burial2birth[burial_id])
        burial2birth[burial_id] = birth_id
        #if birth_id in birth2burial:
        #    print "Birth {} assigned to {} and {}".format(
        #            birth_id, burial_id, birth2burial[birth_id])
        birth2burial[birth_id] = burial_id
    print "{} original links.".format(len(burial2birth))
    t0 = time.time()
    burials = hiski.Burial.query.filter(
            hiski.Burial.event_id.in_(burial2birth.keys())).all()
    print "Burial querying took {:.2f} seconds".format(time.time() - t0)
    inserts = []
    n_ignored = 0
    i = 0
    for Bu in burials:
        year = Bu.death_year
        if year == 0:
            year = Bu.burial_year
        if 1600 < year < 1918:
            birth_id = burial2birth[Bu.event_id]
            inserts.append({'birth_id': birth_id, 'burial_id': Bu.event_id,
                            'death_year': year})
            i += 1
        else:
            n_ignored += 1
    print "Inserting {} links ({} ignored)".format(len(inserts), n_ignored)
    t0 = time.time()
    hiski.engine.execute(hiski.BirthBurialLink.__table__.insert(), inserts)
    print "Inserting took {:.2f} seconds".format(time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default="burial_event_ids.txt",
                        help=("Path to the event_id file."))
    parser.add_argument("--output", default="burial_birth_links",
                        help=("Prefix of the output file (batch info will be "
                              "appended)."))
    parser.add_argument("--num_batches", type=int, help=(
            "The total number of batches to which the event_ids are divided."))
    parser.add_argument("--batch_index", type=int,
                        help=("Index of the current batch (starting from 1)."))
    parser.add_argument('--writedb', action="store_true", default=False, help=(
            "If this flag is set, don't compute the links but store them into the DB."))
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
    
    output_fname = "{}_{}of{}.txt".format(args.output, args.batch_index, args.num_batches)
    print "Writing to:", output_fname

    if not args.writedb:
        process_people(event_ids, output_fname)
    else:
        write_links_to_db(output_fname)

