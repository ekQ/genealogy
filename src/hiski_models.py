# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:50:54 2014

@author: emalmi
"""
import os
import sqlalchemy
from sqlalchemy import func
from sqlalchemy import MetaData, Table, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    relationship,
    backref,
)

from historical_name_normalizer.name_normalizer import clean_name


db_fname = 'sqlite:///{}'.format(os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                  '../data/all_hiski_records2018.sqlite3')))
engine = sqlalchemy.create_engine(db_fname, echo=False)
metadata = MetaData(bind=engine)
session = sqlalchemy.orm.scoped_session(sqlalchemy.orm.sessionmaker(autoflush=False, bind=engine))

Base = declarative_base()
Base.query = session.query_property()


class Birth(Base):
    __table__ = Table('birth', metadata, autoload=True)
    ''' This table should contain at least the following fields:

    birth_year
    baptism_year
    mom_age (optional)
    parish_id
    village
    child_first_name
    child_first_nameN (normalized child first name, will be populated automatically)
    dad_first_name
    dad_first_nameN
    dad_last_name
    dad_last_nameN
    dad_patronym
    mom_first_name
    mom_first_nameN
    mom_last_name
    mom_last_nameN
    mom_patronym
    '''

    def _join_names(self, first_name=None, last_name=None, patronymic=None,
                    clean_strong=False):
        if clean_strong:
            fname_type = 'first'
            lname_type = 'last'
            patronym_type = 'patronym'
        else:
            fname_type = 'dont_substitute'
            lname_type = 'dont_substitute'
            patronym_type = 'dont_substitute'
        names = []
        if first_name is not None:
            names.append(clean_name(first_name, fname_type))
        if last_name is not None:
            names.append(clean_name(last_name, lname_type))
        if patronymic is not None:
            names.append(clean_name(patronymic, patronym_type))
        return ' '.join(names).strip()
        
    def get_clean_name(self, clean_strong=False):
        return self._join_names(self.child_first_name, self.dad_last_name,
                                clean_strong=clean_strong)

    def get_clean_first_name(self, clean_strong=False):
        name_type = 'first' if clean_strong else 'dont_substitute'
        return clean_name(self.child_first_name, name_type)

    def get_clean_last_name(self, clean_strong=False):
        name_type = 'last' if clean_strong else 'dont_substitute'
        return clean_name(self.dad_last_name, name_type)

    def get_clean_dad_name(self, clean_strong=False):
        return self._join_names(self.dad_first_name, self.dad_last_name,
                                clean_strong=clean_strong)
        
    def get_clean_dad_first_name(self, clean_strong=False):
        name_type = 'first' if clean_strong else 'dont_substitute'
        return clean_name(self.dad_first_name, name_type)

    def get_clean_dad_last_name(self, clean_strong=False):
        name_type = 'first' if clean_strong else 'dont_substitute'
        return clean_name(self.dad_last_name, name_type)

    def get_clean_dad_patronym(self, clean_strong=False):
        name_type = 'patronym' if clean_strong else 'dont_substitute'
        return clean_name(self.dad_patronym, name_type)

    def get_clean_mom_name(self, clean_strong=False):
        return self._join_names(self.mom_first_name, self.mom_last_name,
                                clean_strong=clean_strong)
        
    def get_clean_mom_first_name(self, clean_strong=False):
        name_type = 'first' if clean_strong else 'dont_substitute'
        return clean_name(self.mom_first_name, name_type)

    def get_clean_mom_last_name(self, clean_strong=False):
        name_type = 'last' if clean_strong else 'dont_substitute'
        return clean_name(self.mom_last_name, name_type)

    def get_clean_mom_patronym(self, clean_strong=False):
        name_type = 'patronym' if clean_strong else 'dont_substitute'
        return clean_name(self.mom_patronym, name_type)

    def get_label(self, clean_strong=False):
        return self.get_clean_name(clean_strong=clean_strong) + ' ' + str(self.year)
        

    def get_year_int(self):
        try:
            year = int(self.birth_year)
            if year == 0:
                year = int(self.baptism_year)
            return year
        except:
            return None

    def info_dict(self, birth2burial=None):
        I = {}
        I['hiski_id'] = self.event_id
        I['first_name'] = self.get_clean_first_name()
        I['last_name'] = self.get_clean_last_name()
        I['normalized_first_name'] = self.child_first_nameN
        I['year'] = self.birth_year
        I['month'] = self.birth_month
        I['day'] = self.birth_day
        I['parish_id'] = int(self.parish_id)
        if self.village_id is None:
            I['village_id'] = None
        else:
            I['village_id'] = int(self.village_id)
        I['dad_first_name'] = self.get_clean_dad_first_name()
        I['dad_last_name'] = self.get_clean_dad_last_name()
        I['dad_patronym'] = self.get_clean_dad_patronym()
        I['normalized_dad_first_name'] = self.dad_first_nameN
        I['normalized_dad_last_name'] = self.dad_last_nameN
        I['mom_first_name'] = self.get_clean_mom_first_name()
        I['mom_last_name'] = self.get_clean_mom_last_name()
        I['mom_patronym'] = self.get_clean_mom_patronym()
        I['normalized_mom_first_name'] = self.mom_first_nameN
        I['normalized_mom_last_name'] = self.mom_last_nameN
        I['mom_age'] = self.mom_age
        if birth2burial is not None and self.event_id in birth2burial:
            I['burial_id'] = birth2burial[self.event_id][0]
            I['death_year'] = birth2burial[self.event_id][1]
        else:
            I['burial_id'] = None
            I['death_year'] = None
        return I


class Burial(Base):
    __table__ = Table('burial', metadata, autoload=True)


class Marriage(Base):
    __table__ = Table('marriage', metadata, autoload=True)


class Emigrated(Base):
    __table__ = Table('emigrated', metadata, autoload=True)


class Immigrated(Base):
    __table__ = Table('immigrated', metadata, autoload=True)


class Village(Base):
    __table__ = Table('village', metadata, autoload=True)


class Parish(Base):
    __table__ = Table('parish', metadata, autoload=True)

class BirthBurialLink(Base):
    __table__ = Table('birth_burial_link', metadata, autoload=True)

