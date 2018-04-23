from sqlalchemy.sql.expression import bindparam
import time
import datetime as dt

from historical_name_normalizer import name_normalizer
name_normalizer.DEBUG = False
# TODO Update the following line if you want to normalize names for other than
# HisKi DB.
from hiski_models import *


BATCH_SIZE = 100000

first_normalizer = name_normalizer.NameNormalizer('first')
last_normalizer = name_normalizer.NameNormalizer('last_extended')
patronym_normalizer = name_normalizer.NameNormalizer('patronym')
cod_normalizer = name_normalizer.NameNormalizer('cause_of_death_extended')


def populate_normalizations(table, fields, field_name_types):
    assert len(fields) == len(field_name_types)
    normalized_fields = [field + 'N' for field in fields]
    stmt = table.__table__.update().\
        where(table.event_id == bindparam('_event_id')).\
        values({normalized_fields[i]: bindparam(normalized_fields[i])
                for i in range(len(fields))})

    updates = []
    n_exact = 0
    for i, row in enumerate(table.query.yield_per(1000)):
        if i % 10000 == 0:
            print "{}    {} rows.".format(dt.datetime.now().isoformat()[:-7], i)
        up = {'_event_id': row.event_id}
        for j in range(len(fields)):
            fnt = field_name_types[j]
            if fnt == 'first':
                normalizer = first_normalizer
            elif fnt == 'last':
                normalizer = last_normalizer
            elif fnt == 'patronym':
                normalizer = patronym_normalizer
            elif fnt == 'cod':
                normalizer = cod_normalizer
            else:
                raise Exception('Bad field name type.')
            name = row.__dict__[fields[j]]
            if fnt == 'first':
                only_first_token = True
            else:
                only_first_token = False
            info = {}
            norm_name = normalizer.normalize(name, find_nearest=True, info=info,
                                             only_first_token=only_first_token)
            up[normalized_fields[j]] = norm_name
            if info['sim'] == 1:
                n_exact += 1
        updates.append(up)
        #if i > 110000:
        #    break
    print "  {} out of {} normalizations were exact.".format(n_exact, i*len(fields))
    print "  Starting to execute..."
    batch_idx = 0
    while batch_idx * BATCH_SIZE < len(updates):
        t0 = time.time()
        engine.execute(stmt, updates[(batch_idx * BATCH_SIZE):((batch_idx+1) * BATCH_SIZE)])
        print "  {}: Executing took {:.4f} seconds.".format(batch_idx, time.time()-t0)
        batch_idx += 1

def populate_birth():
    fields = [
            #"dad_profession",
            "dad_first_name",
            "dad_patronym",
            "dad_last_name",
            #"mom_profession",
            "mom_first_name",
            "mom_patronym",
            "mom_last_name",
            "child_first_name"
            ]
    name_types = ['first', 'patronym', 'last', 'first', 'patronym', 'last', 'first']
    
    #fields = ["dad_first_name", "mom_first_name", "child_first_name"]
    #name_types = ['first', 'first', 'first']
    populate_normalizations(Birth, fields, name_types)

def populate_marriage():
    fields = [
            #"husb_profession",
            "husb_first_name",
            "husb_patronym",
            "husb_last_name",
            #"wife_profession",
            "wife_first_name",
            "wife_patronym",
            "wife_last_name",
            ]
    name_types = ['first', 'patronym', 'last', 'first', 'patronym', 'last']
    
    #fields = ["husb_first_name", "wife_first_name"]
    #name_types = ['first', 'first']
    populate_normalizations(Marriage, fields, name_types)

def populate_burial():
    fields = [
            "first_name",
            "patronym",
            "last_name",
            "death_cause",
            ]
    name_types = ['first', 'patronym', 'last', 'cod']
    
    #fields = ["first_name"]
    #name_types = ['first']
    populate_normalizations(Burial, fields, name_types)

def populate_emigrated():
    fields = [
            #"destination",
            "first_name",
            "patronym",
            "last_name",
            ]
    name_types = ['first', 'patronym', 'last']
    
    #fields = ["first_name"]
    #name_types = ['first']
    populate_normalizations(Emigrated, fields, name_types)

def populate_immigrated():
    fields = [
            #"source",
            "first_name",
            "patronym",
            "last_name",
            ]
    name_types = ['first', 'patronym', 'last']
    
    #fields = ["first_name"]
    #name_types = ['first']
    populate_normalizations(Immigrated, fields, name_types)


if __name__ == "__main__":
    populate_birth()
    populate_burial()
    populate_marriage()
    populate_emigrated()
    populate_immigrated()
