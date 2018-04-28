# -*- coding: utf-8 -*-
import codecs
import re
import json
import os


class OccupationNormalizer:
    def __init__(self):
        self.normalized_occ2status, self.occ2normalized = get_status_and_abbreviation_map()

    def normalize_and_get_status(self, occupation):
        '''
        Return the given occupation normalized and a status dict. If status is
        not found, a None status is returned.
        '''
        occupationN = clean(occupation)
        occ_matches = self.occ2normalized.get(occupationN, [])
        if len(occ_matches) == 1:
            occupationN = occ_matches[0][0]
        if occupationN in self.normalized_occ2status:
            status = self.normalized_occ2status[occupationN]
        else:
            status = None
        return occupationN, status


def clean(n):
    if n is None:
        return u''
    n = n.lower()
    n = n.split('\\k')[0]
    n = re.sub(u'[^a-zöäå ]', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def get_abbreviation_map(add_non_abbreviated=True, debug=0):
    am = {}
    fname = os.path.join(os.path.dirname(__file__), 'historismi_lyhenteet_filtered.tsv')
    f = codecs.open(fname, 'r', 'utf8')
    for line in f:
        parts = line.rstrip('\n').lower().split('\t')
        abbs, term, fi = parts
        abb_list = abbs.split(', ')
        abb_list += term.split(', ')
        abb_list += fi.split(', ')
        for abb in abb_list:
            abb = clean(abb)
            if abb not in am:
                am[abb] = []
            existing_terms = [t for t, f in am[abb]]
            if term not in existing_terms:
                am[abb].append((term, fi))
    len0 = len(am)
    # Add non-abbreviated occupations to the map.
    if add_non_abbreviated:
        f = codecs.open(os.path.join(os.path.dirname(__file__), 'historismi_ammatit.tsv'), 'r', 'utf8')
        for line in f:
            parts = line.rstrip('\n').lower().split('\t')
            se, fi, en = parts
            abb_list = se.split(', ')
            abb_list += fi.split(', ')
            term_fis = []
            for abb in abb_list:
                abb = clean(abb)
                if abb in am:
                    if am[abb][0] not in term_fis:
                        term_fis.append(am[abb][0])
            if len(term_fis) == 0:
                term_fis = [(se.split(', ')[0], fi.split(', ')[0])]
            elif debug > 0 and len(term_fis) > 1:
                print "Ambiguous:", se, fi, term_fis
            for abb in abb_list:
                abb = clean(abb)
                if abb not in am:
                    am[abb] = []
                existing_terms = [t for t, f in am[abb]]
                if term_fis[0][0] not in existing_terms:
                    am[abb].append(term_fis[0])
    if debug > 0:
        print "{} abbreviations ({} strict abbreviations).".format(len(am), len0)
    return am

def get_status_and_abbreviation_map():
    am = get_abbreviation_map()
    occ2status_raw = json.load(open(os.path.join(os.path.dirname(__file__), 'occupation2status.json')))
    occ2status = {}
    for occ, status in occ2status_raw.iteritems():
        occ = clean(occ)
        occ_matches = am.get(occ, [])
        if len(occ_matches) == 1:
            occ = occ_matches[0][0]
        occ2status[occ] = status
    return occ2status, am

if __name__ == '__main__':
    ON = OccupationNormalizer()
    occupation = 'bd.'
    occupationN, status = ON.normalize_and_get_status(occupation)
    print u'The normalized form of {} is {}\nStatus dict: {}'.format(
            occupation, occupationN, status)
