'''
Assortative mating analysis (Figures 4 and 5 in the paper).

Usage:
    python assortative_mating_analysis.py
'''
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from sqlalchemy import or_
import codecs
import cPickle as pickle
import os
from scipy.stats import pearsonr
import scikits.bootstrap as bootstrap
import sys

import normalize_occupations


#TODO: Uncomment and adjust the following two lines if you want to re-compute
# the status vectors.
#sys.path.append('<ADD THE FULL PATH TO THE PARENT src DIRECTORY HERE>')
#import hiski_models as hiski

seed = 42
np.random.seed(seed)

def precompute_status_vectors(fname_out, min_prob=0.9):
    fnames = glob.glob('../reconstructions/all_parent_edges7_rank1_rows.json')
    child_counts = {True: {}, False: {}}
    moms = {}
    dads = {}
    moms[min_prob] = {}
    dads[min_prob] = {}

    t0 = time.time()
    for fname in fnames:
        for line in open(fname):
            d = json.loads(line)
            child_counts[d['dad']][d['parent']] = child_counts[d['dad']].get(d['parent'], 0) + d['prob']
            if d['prob'] >= min_prob:
                if d['dad']:
                    dads[min_prob][d['child']] = d['parent']
                else:
                    moms[min_prob][d['child']] = d['parent']
    print "Counting took {:.2f} seconds.".format(time.time() - t0)
    print "Found {} certain dad links and {} certain mom links (p>={:.2f}).".format(
            len(dads[min_prob]), len(moms[min_prob]), min_prob)

    OccNorm = normalize_occupations.OccupationNormalizer()
    kids_with_both_parents = list(set(moms[min_prob].keys()) & set(dads[min_prob].keys()))
    spouses = list(set([(moms[min_prob][kid], dads[min_prob][kid]) for kid in kids_with_both_parents]))
    np.random.shuffle(spouses)
    print "{} spouses found.".format(len(spouses))

    spouses_filtered = []
    years = []
    occupations = []
    statuses = []
    parishes = []

    n_status = 0
    for (wife, husb) in spouses:
        if i % 900 == 0:
            print i, time.time() - t0
        wife_object = hiski.Birth.query.get(wife)
        husb_object = hiski.Birth.query.get(husb)
        occupation_wife, status_wife = OccNorm.normalize_and_get_status(
            wife_object.dad_profession)
        occupation_husb, status_husb = OccNorm.normalize_and_get_status(
            husb_object.dad_profession)
        if len(occupation_wife) < 1 or len(occupation_husb) < 1:
            continue
        years.append((wife_object.get_year_int(), husb_object.get_year_int()))
        occupations.append((occupation_wife, occupation_husb))
        parishes.append((wife_object.parish_id, husb_object.parish_id))
        statuses.append((status_wife, status_husb))
        spouses_filtered.append((wife, husb))
        if status_wife is not None and status_husb is not None:
            n_status += 1

    print "{} spouses with known dad occupations.".format(len(occupations))
    print "{} spouses with known dad statuses.".format(n_status)

    status_vectors = (spouses_filtered, years, occupations, statuses, parishes)
    pickle.dump(status_vectors, open(fname_out, 'wb'))
    print "Wrote:", fname_out
    return status_vectors


def get_status_vectors(min_prob=0.9):
    fname = 'data/status_data_p{:.2f}.pckl'.format(min_prob)
    if not os.path.exists(fname):
        status_vectors = precompute_status_vectors(fname, min_prob)
    else:
        status_vectors = pickle.load(open(fname, 'rb'))
    spouses, years, occupations, statuses, parishes = status_vectors
    return spouses, years, occupations, statuses, parishes


def find_similar_person_occupation(
        parish2gender2personInfos, parish, is_male, year, ignored_person,
        max_year_delta=10, n_persons=1):
    if parish not in parish2gender2personInfos:
        return None
    best_delta = max_year_delta + 1
    best_occupation = None
    ok_occupations = []
    for candidate_year, occupation, pid, status in parish2gender2personInfos[parish][is_male]:
        delta = np.abs(candidate_year - year)
        if delta < best_delta and pid != ignored_person:
            best_delta = delta
            best_occupation = occupation
        if delta < max_year_delta and pid != ignored_person:
            ok_occupations.append((occupation, status))
    if len(ok_occupations) > 0:
        idxs = np.random.choice(range(len(ok_occupations)), n_persons)
        ok_occupations = np.asarray(ok_occupations)
        return ok_occupations[idxs]
    else:
        return None


def get_comparisons(spouses, years, occupations, statuses, parishes, n_alt_spouses=1):
    parish2gender2personInfos = {}
    for i in range(len(occupations)):
        for gender in range(2):
            if parishes[i][gender] not in parish2gender2personInfos:
                parish2gender2personInfos[parishes[i][gender]] = {0: [], 1: []}
            parish2gender2personInfos[parishes[i][gender]][gender].append(
                (years[i][gender], occupations[i][gender], spouses[i][gender], statuses[i][gender]))

    is_same = []
    is_same_randomized = []
    status4_same = []
    status4_same_randomized = []
    hiscam_diffs = []
    hiscam_diffs_randomized = []
    years_wife = []
    years_wife_randomized = []
    ok_idxs = []
    for idx in range(len(spouses)):
        # Randomly pick the spouse to replace.
        spouse_to_replace = np.random.randint(0,2)
        occ_new_spouses = find_similar_person_occupation(
                parish2gender2personInfos=parish2gender2personInfos,
                parish=parishes[idx][spouse_to_replace],
                is_male=spouse_to_replace,
                year=years[idx][spouse_to_replace],
                ignored_person=spouses[idx][spouse_to_replace],
                n_persons=n_alt_spouses)
        if occ_new_spouses is None:
            spouse_to_replace = 1 - spouse_to_replace
            occ_new_spouses = find_similar_person_occupation(
                    parish2gender2personInfos=parish2gender2personInfos,
                    parish=parishes[idx][spouse_to_replace],
                    is_male=spouse_to_replace,
                    year=years[idx][spouse_to_replace],
                    ignored_person=spouses[idx][spouse_to_replace],
                    n_persons=n_alt_spouses)
            if occ_new_spouses is None:
                # Suitable new spouse was not found.
                continue
        # Compute comparisons for the actual spouses.
        is_same.append(occupations[idx][0] == occupations[idx][1])
        if statuses[idx][0] is not None and statuses[idx][1] is not None:
            status4_same.append(statuses[idx][0]['socialClass4'] == statuses[idx][1]['socialClass4'])
            hiscam_diffs.append(np.abs(statuses[idx][0]['hiscam'] - statuses[idx][1]['hiscam']))
        else:
            status4_same.append(None)
            hiscam_diffs.append(None)
        years_wife.append(years[idx][0])
        # Compute comparisons for the null model spouses.
        new_occupations = list(occupations[idx])
        new_statuses = list(statuses[idx])
        for sp_i in range(n_alt_spouses):
            new_occupations[spouse_to_replace] = occ_new_spouses[sp_i][0]
            new_statuses[spouse_to_replace] = occ_new_spouses[sp_i][1]
            is_same_randomized.append(new_occupations[0] == new_occupations[1])
            if new_statuses[0] is not None and new_statuses[1] is not None:
                status4_same_randomized.append(new_statuses[0]['socialClass4'] == new_statuses[1]['socialClass4'])
                hiscam_diffs_randomized.append(np.abs(new_statuses[0]['hiscam'] - new_statuses[1]['hiscam']))
            else:
                status4_same_randomized.append(None)
                hiscam_diffs_randomized.append(None)
            # Wife year is set based on the original wife even if it's swapped.
            years_wife_randomized.append(years[idx][0])
        ok_idxs.append(idx)
    comparisons = {
            'occupation': (is_same, is_same_randomized),
            'status4': (status4_same, status4_same_randomized),
            'hiscam': (hiscam_diffs, hiscam_diffs_randomized),
            'years': (years_wife, years_wife_randomized),
            }
    return comparisons


def ratio_statfunction(data, weights=None):
    n = data.shape[0]
    if weights is None:
        weights = np.ones(n)
    p, p_null = np.average(data, weights=weights, axis=0)
    return p / p_null


def ratio_statfunction_hiscam(data, weights=None):
    n = data.shape[0]
    if weights is None:
        weights = np.ones(n)
    p, p_null = np.average(data, weights=weights, axis=0)
    return p_null / p


def compute_curves_bootstrap(is_same, is_same_null, years, min_year, max_year, delta=10,
                             comparison='occupation', use_bootstrap=True):
    assert len(is_same) == len(years)
    assert len(is_same) == len(is_same_null)
    yearly_vals = [([], []) for i in range(max(max(years), max_year + delta) + 1)]
    for i, year in enumerate(years):
        match = is_same[i]
        match_null = is_same_null[i]
        if match is not None and match_null is not None:
            yearly_vals[year][0].append(match)
            yearly_vals[year][1].append(match_null)
    curve = []
    lbs = []
    ubs = []
    curve_null = []
    lbs_null = []
    ubs_null = []
    ratio = []
    rlbs = []
    rubs = []
    for y in range(min_year, max_year+1):
        y_vals = []
        y_vals_null = []
        for y2 in range(max(min_year, y-delta), min(max_year, y+delta)):
            y_vals += yearly_vals[y2][0]
            y_vals_null += yearly_vals[y2][1]
        n = len(y_vals)
        assert n == len(y_vals_null)
        data = np.hstack((np.reshape(y_vals, (n, 1)), np.reshape(y_vals_null, (n, 1))))
        p, p_null = np.average(data, axis=0)
        curve.append(p)
        curve_null.append(p_null)
        # Confidence intervals
        lb, ub = bootstrap.ci(data=y_vals, statfunction=np.average, method='abc')
        lbs.append(lb)
        ubs.append(ub)
        lb_null, ub_null = bootstrap.ci(data=y_vals_null, statfunction=np.average, method='abc')
        lbs_null.append(lb_null)
        ubs_null.append(ub_null)
        # Ratio.
        if comparison == 'hiscam':
            r = ratio_statfunction_hiscam(data)
            rlb, rub = bootstrap.ci(data=data, statfunction=ratio_statfunction_hiscam, method='abc')
        else:
            r = ratio_statfunction(data)
            rlb, rub = bootstrap.ci(data=data, statfunction=ratio_statfunction, method='abc')
        ratio.append(r)
        rlbs.append(rlb)
        rubs.append(rub)
        if y % 10 == 0:
            print y
    curve = np.asarray(curve)
    lbs = np.asarray(lbs)
    ubs = np.asarray(ubs)
    curve_null = np.asarray(curve_null)
    lbs_null = np.asarray(lbs_null)
    ubs_null = np.asarray(ubs_null)
    ratio = np.asarray(ratio)
    rlbs = np.asarray(rlbs)
    rubs = np.asarray(rubs)
    return curve, lbs, ubs, curve_null, lbs_null, ubs_null, ratio, rlbs, rubs


def analyze(min_prob, comparison='status4', min_year=1760, max_year=1880,
            year_delta=5, n_alt_spouses=1, axes=None, ylim1=None, ylim2=None, xlim=None,
            show_legends=True, fixed_ylabel1=None, fixed_xlabel=None, show_box=False,
            use_sa_settings=False):
    t0 = time.time()
    spouses, years, occupations, statuses, parishes = get_status_vectors(min_prob)
    C = get_comparisons(spouses, years, occupations, statuses, parishes, n_alt_spouses)
    n_hiscams = len([val for val in C['hiscam'][0] if val is not None])
    print "{} spouses retrieved, {} with replacable spouse, {} with hiscam for both. (min_prob: {}, delta: {}, time: {:.2f})".format(
            len(spouses), len(C['occupation'][0]), n_hiscams, min_prob, year_delta, time.time() - t0)

    strat_curve, lb, ub, null_curve, null_lb, null_ub, ratios, ratio_lb, ratio_ub = \
        compute_curves_bootstrap(
                C[comparison][0], C[comparison][1], C['years'][0],
                min_year-year_delta, max_year+year_delta, delta=year_delta,
                comparison=comparison)

    if axes is None:
        plt.figure(figsize=(10,3))
        plt.rcParams["font.family"] = "times new roman"
        plt.rc('text', usetex=True)
        ax = plt.subplot(1,2,1)
    else:
        ax = axes[0]
    if comparison != 'hiscam':
        sc = 100
    else:
        sc = 0.01
    year_ticks = range(min_year, max_year+1)
    legend_loc1 = 'best'
    legend_loc2 = 'best'
    if comparison == 'occupation':
        leg1 = 'Actual spouses ($p$)'
        leg2 = 'Null model ($p_n$)'
        if use_sa_settings:
            leg1 = '$p$'
            leg2 = '$p_n$'
        ylab1 = '\% of matching: Occupations'
        ylab2 = '$p / p_n$'
        legend_loc1 = 'lower right'
    elif comparison == 'status4':
        leg1 = 'Actual spouses ($q$)'
        leg2 = 'Null model ($q_n$)'
        ylab1 = '\% of matching: Class4'
        ylab2 = '$q / q_n$'
        #legend_loc1 = 'lower right'
    elif comparison == 'hiscam':
        leg1 = 'Actual spouses ($\delta$)'
        leg2 = 'Null model ($\delta_n$)'
        ylab1 = 'Absolute difference: HISCAM'
        ylab2 = '$\delta_n / \delta$'
    if fixed_xlabel is not None:
        xlab = fixed_xlabel
    else:
        xlab = 'Year'
    if fixed_ylabel1 is not None:
        ylab1 = fixed_ylabel1
    yd = year_delta
    ax.plot(year_ticks, sc * strat_curve[yd:-yd], '-', label=leg1)
    ax.fill_between(year_ticks, sc * lb[yd:-yd], sc * ub[yd:-yd], alpha=0.2)
    ax.plot(year_ticks, sc * null_curve[yd:-yd], '-', label=leg2)
    ax.fill_between(year_ticks, sc * null_lb[yd:-yd], sc * null_ub[yd:-yd], alpha=0.2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab1)
    if show_legends or use_sa_settings:
        ax.legend(loc=legend_loc1, framealpha=0.3)
    if not show_box:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if comparison == 'hiscam':
        ax.set_ylim((0, 7))
    if ylim1 is not None:
        ax.set_ylim(ylim1)
    if xlim is not None:
        ax.set_xlim(xlim)

    if axes is None:
        ax = plt.subplot(1,2,2)
    else:
        ax = axes[1]
    #ax.plot(year_ticks, np.ones(len(year_ticks)), 'k--', label='(No stratification)')
    #ax.plot(year_ticks, ratios[yd:-yd], '-', label='Social stratification')
    ax.plot(year_ticks, np.ones(len(year_ticks)), 'k--', label='Baseline')
    ax.plot(year_ticks, ratios[yd:-yd], '-', label=ylab2)
    ax.fill_between(year_ticks, ratio_lb[yd:-yd], ratio_ub[yd:-yd], alpha=0.2)
    #ax.set_ylabel(ylab2)
    ax.set_ylabel('Assortative mating')
    ax.set_xlabel(xlab)
    if show_legends:
        handles, labels = ax.get_legend_handles_labels()
        # Reverse the legend order.
        ax.legend(handles[::-1], labels[::-1], loc=legend_loc2, framealpha=0.3)
        #ax.legend()
    if not show_box:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    #if comparison == 'hiscam':
    #    ax.set_ylim((0, 4))
    ax.set_ylim((0.6, 3.1))
    if ylim2 is not None:
        ax.set_ylim(ylim2)
    if xlim is not None:
        ax.set_xlim(xlim)
    #plt.ylim((0.6, 3.2))
    #plt.subplot(2,2,3)
    #plt.plot(year_ticks, ns, '-')
    #plt.ylabel('marriage count')
    #plt.xlabel('year')
    if axes is None:
        plt.tight_layout()
        plt.savefig('plots/social_stratification.pdf')
        #plt.show()
    return ratios[yd:-yd]


def analyze_paper(label=''):
    t0 = time.time()
    min_year = 1735
    max_year = 1885
    year_delta = 10
    n_alt_spouses = 1
    
    plt.figure(figsize=(8,7.5))
    plt.rcParams["font.family"] = "times new roman"
    plt.rc('text', usetex=True)
    plot_ids = range(1,7)
    axes = [plt.subplot(3,2,i) for i in plot_ids]

    min_prob = 0.9
    c1 = analyze(min_prob, comparison='occupation', min_year=min_year, max_year=max_year,
            year_delta=year_delta, n_alt_spouses=n_alt_spouses, axes=axes[:2])
    c2 = analyze(min_prob, comparison='status4', min_year=min_year, max_year=max_year,
            year_delta=year_delta, n_alt_spouses=n_alt_spouses, axes=axes[2:4])
    c3 = analyze(min_prob, comparison='hiscam', min_year=min_year, max_year=max_year,
            year_delta=year_delta, n_alt_spouses=n_alt_spouses, axes=axes[4:6])
    print "Pearson c1-c2:", pearsonr(c1, c2)
    print "Pearson c1-c3:", pearsonr(c1, c3)
    print "Pearson c2-c3:", pearsonr(c2, c3)
    plt.tight_layout()
    fname = 'plots/paper_assortative_mating_{}.pdf'.format(label)
    fname = 'plots/paper_assortative_mating_{}_minProb{:.2f}.pdf'.format(label, min_prob)
    plt.savefig(fname)
    print "Wrote: {} ({:.2f} seconds)".format(fname, time.time() - t0)


def sensitivity_analysis(label=''):
    t0 = time.time()
    min_year=1735
    max_year=1885
    year_delta=10
    n_alt_spouses=1
    comparison = 'occupation'
    
    plt.figure(figsize=(8,8))
    plt.rcParams["font.family"] = "times new roman"
    plt.rc('text', usetex=True)

    min_probs = [0.8, 0.9, 0.95]
    year_deltas = [10, 3]
    nr = 2 * len(year_deltas)
    nc = len(min_probs)
    plot_ids = range(1, nr * nc + 1)
    axes = [plt.subplot(nr, nc, i) for i in plot_ids]
    for i, year_delta in enumerate(year_deltas):
        for j, min_prob in enumerate(min_probs):
            print "\nYear delta: {}, min_prob: {:.2f}".format(year_delta, min_prob)
            ax1 = axes[nc * i * 2 + j]
            ax2 = axes[nc * i * 2 + nc + j]
            analyze(min_prob, comparison=comparison, min_year=min_year, max_year=max_year,
                    year_delta=year_delta, n_alt_spouses=n_alt_spouses, axes=[ax1, ax2],
                    ylim2=(0.7, 3), xlim=(1735, 1885), show_box=True,
                    fixed_xlabel='', fixed_ylabel1='\% match.: Occ.',
                    show_legends=False, use_sa_settings=True)
            ax1.set_title('$p_{th} = %.2f, \Delta t = %d$' % (min_prob, year_delta), fontsize=14)
    plt.tight_layout()
    fname = 'plots/sensitivity_analysis_{}.pdf'.format(label)
    plt.savefig(fname)
    print "Wrote: {} ({:.2f} seconds)".format(fname, time.time() - t0)
    plt.show()


if __name__ == '__main__':
    seed = 534895718
    np.random.seed(seed)
    print "Used seed:", seed
    analyze_paper(label='SEED-{}'.format(seed))
    sensitivity_analysis(label='SEED-{}'.format(seed))
