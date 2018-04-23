import numpy as np
import jellyfish as jf
import cPickle as pickle
from sqlalchemy import func

from historical_name_normalizer.name_normalizer import clean_name
import parish_distances
import hiski_models as hiski


class MatchingModel:

    def __init__(self,
                 dad_ages_path='estimated_models/dad_age_likelihoods.pckl',
                 mom_ages_path='estimated_models/mom_age_likelihoods.pckl',
                 known_ages_path='estimated_models/known_age_likelihoods.pckl',
                 birth_counts_path='estimated_models/birth_counts_per_year.pckl',
                 fname_path='estimated_models/first_name_lefts_and_ratios.pckl',
                 lname_path='estimated_models/last_name_lefts_and_ratios.pckl',
                 patronym_path='estimated_models/patronym_lefts_and_ratios.pckl',
                 location_path='estimated_models/location_ratios.pckl',
                 same_birth_parish_lh=0.921052631579,
                 same_birth_parish_lh_nomatch=0.108722815973,
                 blocking_min_age=10,
                 blocking_max_age=65,
                 classifier_path='estimated_models/xgb_model_20features_0.6194precision_2017-08-09.pckl',
                ):
        self.mom_min_year, self.mom_age_hist = _load_file(mom_ages_path)
        self.dad_min_year, self.dad_age_hist = _load_file(dad_ages_path)
        self.max_age_error, self.known_age_hist = _load_file(known_ages_path)
        # Normalize the distributions just in case.
        self.mom_age_hist /= np.sum(self.mom_age_hist)
        self.dad_age_hist /= np.sum(self.dad_age_hist)
        self.birth_counts = _load_file(birth_counts_path)
        # Compute cumulative birth counts for quick normalization.
        self.cum_birth_counts = np.cumsum(self.birth_counts)
        self.blocking_min_age = blocking_min_age
        self.blocking_max_age = blocking_max_age
        # Name data.
        self.fname_bins, self.fname_lh_ratios = _load_file(fname_path)
        self.lname_bins, self.lname_lh_ratios = _load_file(lname_path)
        self.patronym_bins, self.patronym_lh_ratios = _load_file(patronym_path)
        self.fname_bin_w = _get_fixed_bin_width(self.fname_bins)
        self.lname_bin_w = _get_fixed_bin_width(self.lname_bins)
        self.patronym_bin_w = _get_fixed_bin_width(self.patronym_bins)
        # Location data.
        self.location_estimates = _load_file(location_path)
        self.parish_dists = parish_distances.load_or_compute_distance_map(
            'estimated_models/distance_map.pckl')
        self.parish_coords = parish_distances.get_parish_coordinates()
        self.same_birth_parish_lh = same_birth_parish_lh  # DEPRECATED.
        self.same_birth_parish_lh_nomatch = same_birth_parish_lh_nomatch  # DEPRECATED.
        # Match or not classifier.
        self.classifier = pickle.load(open(classifier_path, 'rb'))

    def birth_year_lh_ratio(self, year_child, year_parent, is_dad,
                            known_parent_age=None):
        '''Likelihood ratio for birth year.'''
        age = year_child - year_parent
        if known_parent_age is not None:
            age = age - known_parent_age
            min_age = -self.max_age_error
            age_hist = self.known_age_hist
            #print "Known age:", known_parent_age, age, age - min_age
        else:
            if is_dad:
                age_hist = self.dad_age_hist
                min_age = self.dad_min_year
            else:
                age_hist = self.mom_age_hist
                min_age = self.mom_min_year
        if age < min_age or age - min_age >= len(age_hist):
            return 0
        lh_match = age_hist[age - min_age]
        
        year_range = (year_parent + self.blocking_min_age, year_parent + self.blocking_max_age)
        birth_count_sum = self.cum_birth_counts[min(year_range[1], len(self.cum_birth_counts))] \
                - self.cum_birth_counts[max(year_range[0] - 1, 0)]
        lh_no_match = self.birth_counts[year_child] / max(float(birth_count_sum), 0.0000001)

        return lh_match / float(lh_no_match)

    def parish_lh_ratio(self, parish1, parish2):
        if parish1 == parish2:
            return self.same_birth_parish_lh / self.same_birth_parish_lh_nomatch
        else:
            return (1 - self.same_birth_parish_lh) / (1 - self.same_birth_parish_lh_nomatch)

    def location_lh_ratio(self, parish_id1, village1, parish_id2, village2, is_dad):
        ratio = 1.0
        same_parish, parish_distance, same_village = _location_sims(
            parish_id1, village1, parish_id2, village2, self.parish_dists)
        if same_parish:
            idx = 0
            if same_village is not None:
                if same_village and is_dad:
                    ratio *= self.location_estimates['dad_village_match_ratio']
                elif same_village and not is_dad:
                    ratio *= self.location_estimates['mom_village_match_ratio']
                elif not same_village and is_dad:
                    ratio *= self.location_estimates['dad_village_nonmatch_ratio']
                elif not same_village and not is_dad:
                    ratio *= self.location_estimates['mom_village_nonmatch_ratio']
        else:
            if parish_distance is not None:
                idx = _get_bin_index(self.location_estimates['bin_lefts'], parish_distance)
            else:
                # If parishes are different and distance is None, return (neutral) 1 ratio.
                return 1.0
        if is_dad:
            ratio *= self.location_estimates['dad_distance_ratios'][idx]
        else:
            ratio *= self.location_estimates['mom_distance_ratios'][idx]
        return ratio

    def first_name_lh_ratio(self, fname1, fname2):
        return _name_lh_ratio(fname1, fname2, self.fname_bins,
                              self.fname_lh_ratios, self.fname_bin_w)

    def last_name_lh_ratio(self, lname1, lname2):
        return _name_lh_ratio(lname1, lname2, self.lname_bins,
                              self.lname_lh_ratios, self.lname_bin_w)

    def patronym_lh_ratio(self, patronym1, patronym2):
        return _name_lh_ratio(patronym1, patronym2, self.patronym_bins,
                              self.patronym_lh_ratios, self.patronym_bin_w)

    def overall_lh_ratio(self, child, parent, is_dad, debug_print=0,
                         ignored_features=None, precomputed_death_year=None):
        if ignored_features is None:
            ignored_features = set()

        ratios = []

        # Time.
        n_nones = 0
        year_child = _parse_year(child.birth_year)
        year_parent = _parse_year(parent.birth_year)
        # Get mother age if known.
        known_parent_age = child.mom_age
        try:
            known_parent_age = int(known_parent_age)
            if known_parent_age < 15 or known_parent_age > 60:
                known_parent_age = None
        except:
            known_parent_age = None
        if year_child is not None and year_parent is not None:
            if "birth_year" not in ignored_features:
                ratios.append(self.birth_year_lh_ratio(year_child, year_parent,
                                                       is_dad, known_parent_age))
            if debug_print > 0:
                print "Birth year ratio: {:.4f} ({} vs {}, is dad {})".format(
                        ratios[-1], year_child, year_parent, is_dad)
        else:
            n_nones += 1
        
        # Burial date if known.
        if "death_year" not in ignored_features:
            if precomputed_death_year is None:
                parent_death_year = (hiski.session.query(func.max(hiski.BirthBurialLink.death_year))
                        .filter_by(birth_id=parent.event_id).first()[0])
            else:
                parent_death_year = precomputed_death_year
            if parent_death_year is not None:
                #print "Death year found for {}: {}. Child year: {}".format(parent.event_id, parent_death_year, year_child)
                if year_child > parent_death_year:
                    ratios.append(0.000000001)

        # Location.
        #if "location_old" not in ignored_features:
        #    ratios.append(self.parish_lh_ratio(child.parish_id, parent.parish_id))
        #if debug_print > 0:
        #    print "Parish ratio: {:.4f} ({} vs {})".format(
        #            ratios[-1], child.parish_id, parent.parish_id)
        if "location" not in ignored_features and "village_location" not in ignored_features:
            ratios.append(self.location_lh_ratio(
                    child.parish_id, child.village, parent.parish_id,
                    parent.village, is_dad))
        elif "location" not in ignored_features and "village_location" in ignored_features:
            ratios.append(self.location_lh_ratio(
                    child.parish_id, '', parent.parish_id, '', is_dad))
        if debug_print > 0:
            print u"Location ratio: {:.4f} (p{} vs p{}; '{}' vs '{}')".format(
                    ratios[-1], child.parish_id, parent.parish_id,
                    child.village, parent.village)

        # Name.
        clean_strong = False
        fname1 = parent.get_clean_first_name(clean_strong=clean_strong)
        lname1 = parent.get_clean_last_name(clean_strong=clean_strong)
        patronym1 = parent.get_clean_dad_first_name(clean_strong=clean_strong)
        if is_dad:
            fname2 = child.get_clean_dad_first_name(clean_strong=clean_strong)
            lname2 = child.get_clean_dad_last_name(clean_strong=clean_strong)
            patronym2 = child.get_clean_dad_patronym(clean_strong=clean_strong)
        else:
            fname2 = child.get_clean_mom_first_name(clean_strong=clean_strong)
            lname2 = child.get_clean_mom_last_name(clean_strong=clean_strong)
            patronym2 = child.get_clean_mom_patronym(clean_strong=clean_strong)
        if len(fname1) > 0 and len(fname2) > 0:
            if "first_name" not in ignored_features:
                ratios.append(self.first_name_lh_ratio(fname1, fname2))
                if debug_print > 0:
                    print u"First name ratio: {:.4f} ({} vs {})".format(
                            ratios[-1], fname1, fname2)
        if len(lname1) > 0 and len(lname2) > 0:
            if "last_name" not in ignored_features:
                ratios.append(self.last_name_lh_ratio(lname1, lname2))
                if debug_print > 0:
                    print u"Last name ratio: {:.4f} ({} vs {})".format(
                            ratios[-1], lname1, lname2)
        if len(patronym1) > 0 and len(patronym2) > 0:
            if "patronym" not in ignored_features:
                ratios.append(self.patronym_lh_ratio(patronym1, patronym2))
                if debug_print > 0:
                    print u"Patronym ratio: {:.4f} ({} vs {})".format(
                            ratios[-1], patronym1, patronym2)

        # Multiply ratios.
        overall = 1
        for r in ratios:
            overall *= r
        if debug_print > 0:
            print "Overall ratio: {:.4f}\n".format(overall)
        return overall

    def get_matching_probabilities(
            self, child, parents, are_dads, no_match_prior=0.105487480021,
            ignored_features=None, method='classifier', debug_print=0):
        '''
        Get matching probabilities for the given (candidate) parents to be the
        true parent of the child. are_dads is a boolean indicating whether we
        are looking for the dad or the mother of the child.

        Output:
            Tuple with the array of probabilities and the probability that none
            of the candidate parents is a match.
        '''
        if len(parents) == 0:
            # No match is certain.
            return np.zeros(0), 1
        # Parent candidates get a uniform prior distributed over the remaining
        # probability mass.
        match_prior = (1 - no_match_prior) / float(len(parents))
        if method == 'NaiveBayes':
            ratios = [self.overall_lh_ratio(child, p, is_dad=are_dads, 
                                            ignored_features=ignored_features,
                                            debug_print=debug_print)
                      for p in parents]
        elif method == 'classifier':
            feat_vectors = [self.extract_features(
                    child, p, is_dad=are_dads, ignored_features=ignored_features,
                    debug_print=debug_print) for p in parents]
            X = np.vstack(feat_vectors)
            pred = self.classifier.predict_proba(X)
            pred = np.asarray([val[1] for val in pred])
            ratios = pred / np.fmax((1 - pred), 1e-8)
        else:
            raise Exception('Unknown probability calculation method: {}'.format(method))
        ratios = np.asarray(ratios)
        denominator = match_prior * np.sum(ratios) + no_match_prior
        probabilities = match_prior * ratios / denominator
        no_match_probability = no_match_prior / denominator
        prob_sum = np.sum(probabilities) + no_match_probability
        assert np.abs(prob_sum - 1) < 0.0001, \
                "Probabilities don't sum to 1 but to {}.\nNo match: {}\nOthers: {}".format(
                        prob_sum, no_match_probability, probabilities)
        return probabilities, no_match_probability

    def extract_features(self, child, parent, is_dad, debug_print=0,
                         ignored_features=None):
        if ignored_features is None:
            ignored_features = set()

        ratios = []  # For the Naive Bayes feature.

        # Time.
        n_nones = 0
        year_child = child.get_year_int()
        year_parent = parent.get_year_int()
        
        # Get mother age if known.
        known_parent_age = child.mom_age
        try:
            known_parent_age = int(known_parent_age)
            if known_parent_age < 15 or known_parent_age > 60:
                known_parent_age = None
        except:
            known_parent_age = None
        if known_parent_age is not None:
            age = year_child - year_parent
            age_error = np.abs(age - known_parent_age)
        else:
            age_error = 0
        
        birth_lh_ratio = self.birth_year_lh_ratio(
                year_child, year_parent, is_dad, known_parent_age)
        ratios.append(birth_lh_ratio)
        
        # Burial date if known.
        if "death_year" not in ignored_features:
            parent_death_year = (hiski.session.query(func.max(hiski.BirthBurialLink.death_year))
                    .filter_by(birth_id=parent.event_id).first()[0])
            if parent_death_year is not None:
                child_born_after_death = year_child > parent_death_year
                child_born_after_death_years = max(0, year_child - parent_death_year)
                if year_child > parent_death_year:
                    ratios.append(0.000000001)
            else:
                child_born_after_death = 0
                child_born_after_death_years = 0

        # Location.
        if "location" not in ignored_features:
            same_parish, parish_distance, same_village = _location_sims(
                    child.parish_id, child.village, parent.parish_id,
                    parent.village, self.parish_dists)
            if same_village is None:
                same_village = False
            if parish_distance is None:
                parish_distance = 100  # This is arbitrary
            if child.parish_id in self.parish_coords:
                coords = self.parish_coords[child.parish_id]
            else:
                coords = self.parish_coords['avg_coord']
            ratios.append(self.location_lh_ratio(
                    child.parish_id, child.village, parent.parish_id,
                    parent.village, is_dad))

        # Name.
        clean_strong = False
        fname1 = parent.get_clean_first_name(clean_strong=clean_strong)
        lname1 = parent.get_clean_last_name(clean_strong=clean_strong)
        patronym1 = parent.get_clean_dad_first_name(clean_strong=clean_strong)
        if is_dad:
            fname2 = child.get_clean_dad_first_name(clean_strong=clean_strong)
            lname2 = child.get_clean_dad_last_name(clean_strong=clean_strong)
            patronym2 = child.get_clean_dad_patronym(clean_strong=clean_strong)
        else:
            fname2 = child.get_clean_mom_first_name(clean_strong=clean_strong)
            lname2 = child.get_clean_mom_last_name(clean_strong=clean_strong)
            patronym2 = child.get_clean_mom_patronym(clean_strong=clean_strong)
        same_fname = (fname1 == fname2)
        same_lname = (lname1 == lname2)
        fname_sim = jf.jaro_winkler(fname1, fname2)
        lname_sim = jf.jaro_winkler(lname1, lname2)
        patronym_sim = jf.jaro_winkler(patronym1, patronym2)
        fnames_nz = (len(fname1) > 0 and len(fname2) > 0)
        lnames_nz = (len(lname1) > 0 and len(lname2) > 0)
        patronyms_nz = (len(patronym1) > 0 and len(patronym2) > 0)
        if not patronyms_nz:
            patronym_sim = 1
        fparts1 = fname1.split()
        fparts2 = fname2.split()
        if len(fparts1) > 1 and len(fparts2) > 1:
            second_name_sim = jf.jaro_winkler(fparts1[1], fparts2[1])
        else:
            second_name_sim = 1
        if len(fname1) > 0 and len(fname2) > 0:
            ratios.append(self.first_name_lh_ratio(fname1, fname2))
        if len(lname1) > 0 and len(lname2) > 0:
            ratios.append(self.last_name_lh_ratio(lname1, lname2))
        if len(patronym1) > 0 and len(patronym2) > 0:
            ratios.append(self.patronym_lh_ratio(patronym1, patronym2))

        # Overall likelihood ratio.
        overall = 1
        for r in ratios:
            overall *= r
        
        features = []
        features.append(self.birth_year_lh_ratio(year_child, year_parent, is_dad))
        features.append(year_child - year_parent)
        features.append(year_child - 1815)  # Deduct the avg(birth_year) to normalize.
        features.append(age_error)
        if "death_year" not in ignored_features:
            features.append(child_born_after_death)
            features.append(child_born_after_death_years)
        if "location" not in ignored_features:
            features.append(same_parish)
            features.append(parish_distance)
            features.append(same_village)
            features.append(coords[0])
            features.append(coords[1])
        features.append(same_fname)
        features.append(same_lname)
        features.append(fname_sim)
        features.append(lname_sim)
        features.append(patronym_sim)
        #features.append(fnames_nz)
        #features.append(lnames_nz)
        features.append(patronyms_nz)
        features.append(second_name_sim)
        features.append(overall)
        features.append(is_dad)
        for i, feat in enumerate(features):
            assert feat is not None, "{} is None for child {}-parent{}".format(
                i, child.event_id, parent.event_id)

        return features


### Helper functions. ###

def _load_file(path):
    return pickle.load(open(path, 'rb'))


def _parse_year(year):
    try:
        year = int(year)
        return year
    except:
        return None


def _get_bin_index(lefts, x, fixed_bin_w=None):
    '''
    Get index of the bin to which x falls. If x < lefts[0], return -1.
    
    Input:
        lefts -- Left borders of the bins.
        x -- Value whose bin index is searched for.
        fixed_bin_w -- If not None, all bins have this width.
    
    Output:
        Index [0, len(lefts)-1].
    '''
    if x < lefts[0]:
        return -1
    
    if fixed_bin_w is None:
        if x >= lefts[-1]:
            idx = len(lefts) - 1
        else:
            lefts = np.asarray(lefts)
            #idx = np.nonzero(lefts <= x)[0][-1]
            idx = np.argmax(lefts > x) - 1
    else:
        # Count number of steps and add small delta to avoid rounding errors.
        idx = int((x - lefts[0]) / fixed_bin_w + 1e-6)
        idx = min(idx, len(lefts) - 1)
    return idx


def _get_fixed_bin_width(lefts, tol=1e-6):
    if len(lefts) < 2:
        return None
    lefts = np.asarray(lefts)
    bin_w = lefts[1] - lefts[0]
    if np.all(np.abs(lefts[1:] - lefts[:-1] - bin_w) < tol):
        #print "Detected fixed bin widht:", bin_w
        return bin_w
    else:
        return None


def _name_lh_ratio(name1, name2, bins, lh_ratios, bin_w):
    sim = jf.jaro_winkler(name1, name2)
    if sim < bins[0]:
        idx = 0
    else:
        idx = _get_bin_index(bins, sim, bin_w)
    return lh_ratios[idx]


def _location_sims(parish_id1, village1, parish_id2, village2,
                   parish_dists, location_match_threshold=0.8):
    same_parish = (parish_id1 == parish_id2)
    if same_parish:
        parish_distance = 0
    else:
        parish_distance = parish_dists.get((parish_id1, parish_id2), None)
    if same_parish and len(village1) > 1 and len(village2) > 1:
        village_name_sim = jf.jaro_winkler(clean_name(village1),
                                           clean_name(village2))
        same_village = (village_name_sim >= location_match_threshold)
    else:
        same_village = None
    return (same_parish, parish_distance, same_village)

#########################