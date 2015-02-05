'''
Created on 22 Nov 2014

@author: Antonio
'''

import nltk
import re
from metamap import Metamap
import cPickle as pickle
import bs4
from ierct.take_abstract import take_title
from ierct.preprocessing_functions import remove_paratag

import os
os.chdir(r"C:\Users\Antonio\git\IERCT\ierct\baselines")





def find_tag_postion(abstract):
    
    out = {}

    abstract = re.sub(r'\<\/(p|a1|a2|oc|r1|r2)\>','',abstract)
    
    s = re.finditer('<p>',abstract.replace('<a1>','').replace('<a2>','').replace('<oc>','').replace('<r1>','').replace('<r2>',''))
    out['patient'] = [s_.start() for s_ in s if s_]
    
    s = re.finditer('<a1>',abstract.replace('<p>','').replace('<a2>','').replace('<oc>','').replace('<r1>','').replace('<r2>',''))
    out['intervention'] = [s_.start() for s_ in s if s_]
    
    s = re.finditer('<a2>',abstract.replace('<p>','').replace('<a1>','').replace('<oc>','').replace('<r1>','').replace('<r2>',''))
    out['comparison'] = [s_.start() for s_ in s if s_]
            
    s = re.finditer('<oc>',abstract.replace('<p>','').replace('<a1>','').replace('<a2>','').replace('<r1>','').replace('<r2>',''))
    out['outcome_measure'] = [s_.start() for s_ in s if s_]
    
    s = re.finditer('<r1>',abstract.replace('<p>','').replace('<a1>','').replace('<a2>','').replace('<oc>','').replace('<r2>',''))
    out['result1'] = [s_.start() for s_ in s if s_]
    
    s = re.finditer('<r2>',abstract.replace('<p>','').replace('<a1>','').replace('<a2>','').replace('<oc>','').replace('<r1>',''))
    out['result2'] = [s_.start() for s_ in s if s_]
    
    return out



class RuleBasedExtractor:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def extract(self, abstract):
        out = {}
        out['patient'] = self.extract_patient(abstract)
        out['intervention-comparison'] = self.extract_intervention_comparison(abstract)
        out['outcome-measure'] = self.extract_outcome_measure(abstract)
        out['results'] = self.extract_results(abstract)
        
        return out
    




class RuleBasedExtractorAT(RuleBasedExtractor):
    
    def __init__(self):
        RuleBasedExtractor.__init__(self)
    
    def extract_patient(self, abstract):
        for i in range(len(abstract)-1):
            if abstract[i]['semantic-class'] == 'PATIENTS' \
                and abstract[i+1]['word'].lower() == 'with':
                return abstract[i]
            
    
    
    def extract_intervention_comparison(self, abstract):
        pass
    
    
    def extract_outcome_measure(self, abstract):
        pass
    
    
    def extract_results(self, abstracts):
        pass
    



class RuleBasedExtractorDF(RuleBasedExtractor):
    
    def __init__(self):
        RuleBasedExtractor.__init__(self)
        self.mm = Metamap()
        self._processed_cache = {}#pickle.load(open('./processed.pk','rb'))
        self._semclasses = self._load_semclasses()
    
    
    
    def _load_semclasses(self):
        sn = pickle.load(open('./semanticnetwork.pk','rb'))
        
        scs = []
        for s in sn:
            if s[1] in ('treats', 'carries out') and s[2] == 'Disease or Syndrome':
                scs.append(s[0])
        
        return scs
    
    
    def format_abstract(self, pmid, title, abstract):
        abstract = abstract.replace(pmid,'')
        abstract = re.sub(r'^\W*', '', abstract)
        out = '\n'.join(['UI  - %s' %pmid,
                         'TI  - %s' %title,
                         'AB  - %s' %abstract])
        
        return out
        
    
    
    def preprocess(self, abstract, keeptags = False, nparagraphs = 4):
        
        pmid = re.findall(r'\<pmid\sversion\=\"1\"\>([0-9]+)\<\/pmid\>', abstract)[0]
        
        if True:#'label=' in abstract:
            abstract = '\n'.join(remove_paratag(abstract))
            
        if keeptags:
            abstract = abstract.replace(r'<pmid version="1">', '')
        else:
            abstract = self.remove_tags(abstract)
        
#         pmid = re.findall(r'[0-9]+\W',abstract)[0].strip()
        title = take_title(pmid)
        
        abstract = self.format_abstract(pmid,title,abstract)
        
        abpars = abstract.split('\n')
        cutoff = 1
        while len(''.join(abpars[:cutoff])) < len(abstract)/2.0:
            cutoff += 1
        abstract = '\n'.join(abpars[:cutoff])
        
        return [abstract, pmid]
    
    
    def extract(self, abstract_file):
        '''
        
        :param abstract_file: str; abstract file path
        '''
        
        abstract = open(abstract_file, 'rb').read()
        abstract, pmid = self.preprocess(abstract)
        
        with open('./abstract.tmp', 'wb') as foow:
            foow.write(abstract)
        
        if self._processed_cache.has_key(pmid):
            result = self._processed_cache[pmid]
        else:
            result = self.mm.process_file('./abstract.tmp')
            self._processed_cache[pmid] = result
            pickle.dump(self._processed_cache, open('./processed.pk','wb'))
        
        soup = bs4.BeautifulSoup(result, 'html5lib')
        
        out = {}
        out['patient'] = self.extract_patient(soup)
        out['intervention-comparison'] = self.extract_intervention_comparison(soup)
#         out['outcome-measure'] = self.extract_outcome_measure(abstract)
#         out['results'] = self.extract_results(abstract)
        return out
    

    def extract_patient(self, soup):
        for utterance in soup.findAll('utterance')[1:]:
            for phrase in utterance.findAll('phrase'):
                if int(phrase.mappings.attrs['count'])>0:
                    patient = False
                    disease = False
                    number = False
                    if re.search('[0-9]*\s[Pp]atients?\swith', phrase.phrasetext.text):
                        number = True
                        patient = True
                    for cnd in phrase.find('mappingcandidates').findAll('candidate'):
                        if cnd and cnd.semtypes:
                            if not disease and self.mm.semmap[cnd.semtypes.semtype.text] == 'Disease or Syndrome':
                                disease = True
                            if not patient and self.mm.semmap[cnd.semtypes.semtype.text] == 'Patient or Disabled Group':
                                patient = True
                            if not number and self.mm.semmap[cnd.semtypes.semtype.text] == 'Quantitative Concept':
                                number = True
                    if patient and disease or patient and number:
                        start = int(phrase.phrasestartpos.text)
                        end = start + int(phrase.phraselength.text)
                        return [(start, end), phrase.phrasetext.text]
        return
    
    
    def extract_intervention_comparison(self, soup):
        
        out = []
        for utterance in soup.findAll('utterance')[1:]:
            for phrase in utterance.findAll('phrase'):
                if int(phrase.mappings.attrs['count'])>0:
                    tp = False
                    for cnd in phrase.find('mappingcandidates').findAll('candidate'):
                        if cnd and cnd.semtypes:
                            if self.mm.semmap[cnd.semtypes.semtype.text] in self._semclasses:
                                tp = True
                    if tp:
                        start = int(phrase.phrasestartpos.text)
                        end = start + int(phrase.phraselength.text)
                        out.append([(start,end),phrase.phrasetext.text])
                        if len(out) == 2:
                            return out
        return out
    
    
    def extract_outcome_measure(self, abstract):
        pass
    
    
    def extract_results(self, abstracts):
        pass
    
    
    def remove_tags(self, abstract):
        pattern = re.compile(r'\<\/?\w+\>')
        abstract = abstract.replace(r'<pmid version="1">', '')
        abstract = re.sub(pattern, '', abstract)
        
        return abstract
    
    

    
    

        
        
        
        
        
        
        


    