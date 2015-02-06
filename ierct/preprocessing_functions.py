# -*- coding: utf-8 -*-
'''
Created on 15 Feb 2014

@author: Antonio
'''

from __future__ import division
import nltk, re, bs4, json, urllib, pickle, os, time, shelve
from GeniaTagger import GeniaTagger
import numpy as np
from take_abstract import take_title

import os
os.chdir('c:/users/antonio/git/ierct/ierct')


# imports
#-  stopword list
from nltk.corpus import stopwords
sw = stopwords.words('english')+['non']

#-  abbreviations dictionary
foo = open(r".\data\abbreviations2.dat",'rb')
abbrev_dict = pickle.load(foo)
foo.close()

#- common words
common_words = []
f = open(r'./data/common_words.txt')
for line in f:
    common_words.append(line.strip())

f.close()


# lemmatiser instance
lem = nltk.WordNetLemmatizer()

# genia tagger instance
genia_tagger_instance = GeniaTagger()




# Chunk categorisation procedure

api_key = open("freebase.api_key").read()

admissible_classes = {'DISEASE-OR-MEDICAL-CONDITION':
                       [ 'Disease or medical condition',
                         'Disease',
                         'Symptom',
                         'Biological process',
                         'Disease cause',
                         'BV: Medical Condition',
                         ],                        
                      'MEDICAL-TREATMENT':                      
                       ['Prescription drug',
                        'Solution Drug Formulation',
                        'Drug',
                        'Drug brand',
                        'Drug ingredient',
                        'Chemical Compound',                        
                        'Medical Treatment',
                        'Drug class',
                        'Manufactured drug form',
                        'Injury treatment'],
                      'DIAGNOSTIC-TEST':
                       ['Diagnostic Test'],
                      'CLINICAL-TRIAL':
                       ['Medical trial',
                        'Treatment Medical Trial',
                        'Interventional Medical Trial',
                        'Medical trial design']
                      }
admissible_classes_list = [c for classes in admissible_classes.values()
                             for c in classes]


def get_superclass(cl):
    '''
    maps the class returned by freebase to the relative superclass
    '''
    
    supercl = [c[0] for c in admissible_classes.items()
                    if cl in c[1]]
    return supercl[0]


def freebase_query(query):
    '''
    query is a string to be searched in freebase
    '''
    
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'filter': '(all name:"'+query+'" domain:/Medicine/medicine/)',
            'output': '(type)',
            'key': api_key
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    classes = []    
    for result in response['result']:
        classes += [i['name'] for i in result['output']['type']['/type/object/type']]
        try:    
            classes.append(result['notable']['name'])
        except KeyError: None                      
    classes.append('none')
    return classes


def get_semantic_class(chunk,debug=False): 
    '''   
    returns the semantic class of chunk, where chunk is in string format
    '''
    
    # rule-based categorisation
    patterns = r'''(?x) \_[A-Z]+\_*
                      | \(.*\)
                      | [\(\[\<\>\]\)]
                      | \+\/\-
                      | standard\sdeviation'''    
    group = re.compile(r'''(?x) \_G(ONE|TWO)\_
                              | ([Tt]he)?.*\s([Pp]atients|[Ss]ubjects
                                         |[Ee]yes|[Gg]roup|[Aa]rm)$''')
    poft = re.compile(r'''(?x) ^\_POFT\_$
                             | \_(NUM|RANGE|CONFINT)\_\s*([Hh]ours
                                 |[Dd]ays|[Ww]eeks|[Mm]onths
                                 |[Yy]ears)''')
    patients = re.compile(r'''(?x)[Pp]atients|[Ee]yes|[Ss]ubjects
                                  |[Pp]articipants|[Aa]dults
                                  |[Mm]en|[Ww]omen''')
    clinical_trial = re.compile(r'''(?x) [Cc]linical\s
                                          ([Ss]tudy|[Tt]rial)$
                                         | [Mm]asked.*
                                            ([Ss]tudy|[Tt]rial)$''')
    treatment = re.compile(r'''(?x)([Mm]edications?|[Tt]reatments?
                                |[Vv]ehicle|[Pp]lacebo
                                |[Tt]herap(y|ies)|[Ii]mplant(ation)?s?
                                |[Vv]alve|[Pp]rocedures?)$''')    
    outcome = re.compile(r'''(?x)([Rr]ate|[Cc]hanges?
                                |[Ss]cores?|[Rr]eductions?
                                |[Ii]ncreases?|[Dd]ecreases?
                                |[Ll]evels?|[Vv]aules?)$''')
    frequency = re.compile(r'''(?x)([Oo]nce|[Tt]wice)?.*[Dd]aily''')
    diagtest = re.compile(r'''(?x)([Ee]xam|[Tt]est)$''')
    if patients.search(chunk):
        return 'PATIENTS'    
    if group.search(chunk):
        return 'ARM'
    if poft.search(chunk):
        return 'PERIOD-OF-TIME'
    if clinical_trial.search(chunk):
        return 'CLINICAL-TRIAL'
    if treatment.search(chunk):
        return 'MEDICAL-TREATMENT'
    if outcome.search(chunk):
        return 'OUTCOME-MEASURE'
    if frequency.search(chunk):
        return 'FREQUENCY'
    if diagtest.search(chunk):
        return 'DIAGNOSTIC-TEST'
    
    chunk = re.sub(patterns,'',chunk)
    chunk = re.sub(r'[\-\+\*\.\/]',' ',chunk)    
    if len(chunk.strip())<3:
        return 'none'
    
    # search routine in freebase
    chunk_split = [lem.lemmatize(w.lower())
                    for w in chunk.split() if w not in sw]
    if debug: print chunk_split
    chunk = ' '.join(chunk_split)
    if np.all([(w in common_words) for w in chunk_split]):
        return 'none'
    
    #- searches the entire chunk
    if debug: print chunk+' (0)'
    classes = freebase_query(chunk)
    for cl in classes[:3]:
        if cl in admissible_classes_list:
            return get_superclass(cl)
    
    #- searches chunk without starting words
    for i in range(1,len(chunk_split)):
        chunk = (' '.join(chunk_split[i:]) if len(chunk_split)>1 
                                  else chunk_split)
        if chunk in common_words: continue
        if debug: print chunk+' (1)'
        classes = freebase_query(chunk)
        for cl in classes[:2]:
            if cl in admissible_classes_list:
                return get_superclass(cl)
    
    #- searches chunk without ending words
    for i in range(1,len(chunk_split)):
        chunk = (' '.join(chunk_split[:-i]) if len(chunk_split)>1 
                                  else chunk_split)
        if chunk in common_words: continue        
        if debug: print chunk+' (2)'        
        classes = freebase_query(chunk)
        for cl in classes[:2]:
            if cl in admissible_classes_list:
                return get_superclass(cl)
    
    #- searches single words in chunk
    if len(chunk_split)>2:
        chunk_split = [w for w in chunk_split if w not in common_words]    
        if debug: print chunk_split
        for w in reversed(chunk_split):
            classes = freebase_query(w)
            if classes[0] in admissible_classes_list:
                return get_superclass(classes[0])
    
    return 'none'



 
# Abbreviation expansion

def abbreviations(text):
    '''
    finds all the abbreviations in text and returns them as a dictionary
    with the abbreviations as keys and expansions as values
    '''
    
    tokens = nltk.word_tokenize(text)
    sent_out = str(text)
    pippo = {}
    obrack_pt = re.compile(r'[\(\[]')   
    cbrack_pt = re.compile(r's?[\)\]]') 
    
    for i,t in enumerate(tokens[:-2]):
        tests = (obrack_pt.search(t) and tokens[i+1].isupper() 
                and cbrack_pt.search(tokens[i+2]) 
                and not '=' in tokens[i+1] )       
        if tests:
            pippo[tokens[i+1]] = [w.title() for w in tokens[:i]]   
            sent_out = re.sub(r'[\(\[]'+tokens[i+1]+'[\)\]]','',sent_out)
    
    for a in pippo.iterkeys():
        candidates = []
        for i,w in enumerate(reversed(pippo[a])):
            if i>len(a)+1: break     
            condition = (i>(len(a)-3) if len([1 for l in a if l==a[0]])>1 else True)
            if condition and w.lower().startswith(a[0].lower()) and not w in sw:
                candidates.append(pippo[a][-(i+1):])            
        candidates.sort(key=lambda x: len(x))
        pippo[a] = (candidates[0] if candidates else [])
    
    return [pippo, sent_out]



def expand_abbreviations(sent):
    '''
    returns a copy of 'sent' with abbreviations expanded
    '''
    
    [abbrev_new, sent_new] = abbreviations(sent)
    abbrev_dict.update({k:v for (k,v) in abbrev_new.iteritems()
                        if v})
    keys = sorted(abbrev_dict.keys(), key=lambda x:-len(x))
    
    for k in keys:
        neww = (' '.join(abbrev_dict[k]) if type(abbrev_dict[k]) is list
                else abbrev_dict[k])
        sent_new = (re.sub(k,neww,sent_new) 
                    if neww else sent_new)
    
    return sent_new 



 
# Normalisation

#- list of patterns
patterns = {
    # population
    '_POP_': r'''(?x) (?<=\W)[Nn]\s*\=\s*\d+
                    | (?<=_POP_\,\s)_NUM_
                    | (?<=_POP_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\sand\s)_NUM_
                    | (?<=_POP_\,\s_NUM_\,\sand\s)_NUM_''',
    # confidence intervals
    '_CONFINT_': r'''(?x)\-?\d+(?:\.\d+)?\%?\s*
                        (?:\+\/\-|\Â\±)\s*
                        \d+(?:\.\d+)?\%?
                      | CI\s*(?:[\>\<]|(?:\&lt\;|\&gt\;))\s*\d+(?:\.\d+)?
                      | \(?\[?([Cc][Ii]|[Cc]onfidence\s[Ii]nterval)
                          \,?\)?\]?\s*\=?\s*_RANGE_
                      | (_NUM_|_PERC_)\s*\(?(\+\/|\Â\±|\±)\s*
                          (_NUM_|_PERC_)\)?
                      | _CONFINT_\s\(_NUM_\)
                      | _PERC_\s*_CONFINT_''',
    # confidence intervals with a measure indicator
    '_CONFINTM_': r'''(?x) _CONFINT_\s*mmHg
                         | _NUM_\s?[\(\[]_NUM_[\)\]]\s?mmHg
                         | _MEAS_\s*\(?(\+\/|\Â\±|\±)\s*_NUM_\)?
                         | _NUM_\s*\(?(\+\/|\Â\±|\±)\s*_MEAS_\)?''',
    # ranges
    '_RANGE_': r'''(?x)[\+\-]?\d+\.?\d*\s*(?:\-|to|\/)\s*[\+\-]?
                         \d+\.?\d*(?:\s*\%)?''',
    # ranges with a measure indicator
    '_RANGEM_': r'''(?x) _RANGE_\s*mmHg''',
    # p-values
    '_PVAL_': r'''(?x)[Pp]\s*
                      (?:[\>\<\=]|(?:\&lt\;|\&gt\;)){,2}\s*[01]?(\.\d+|\d+\%)
                    | [Pp]\s*
                      (?:\<|\>|\&gt\;|\&lt\;)\s*(?:or)\s*\=\s*(?:to)?\s*
                      [01]?(\.\d+|\d+\%)''',
    # percentages
    '_PERC_': r'''(?x)(?:[\-\+]\s*)?\d+\.?\d*\s*\%
                     | _NUM_\s[Pp]erc(\.|ent)? 
                     | _NUM_\s_PERC_''',
    # time indications
    '_TIME_': r'''(?x)\d+\W?\d*\s*([AaPp]\.?\s?[Mm]\.?)
                | \d+\:\d{2}
                | hrs|[Hh]ours|hh''', 
    # measurements
    '_MEAS_': r'''(?x)  (\>|\<|\&lt\;|\&gt\;|\=|\≤)(\s*or\s*\=)?
                        \s*\-?\s*\d+\.?\d*\%?
                      | (_NUM_|_MEAS_)\s*
                         \/?(mm\s?[Hh][Gg]|mm|mg\/m[Ll]|mg|m[Ll]
                         |dB(\/(y|year|month))?|DB(\/(y|year|month))?)
                      | _NUM_(?=\sand\s_MEAS_)
                      | _NUM_(?=(\,\s_NUM_)*\,?\sand\s_MEAS_)''',
    # years
    '_YEAR_': r'(?<=[^\d])([12][019]\d{2})(?=[^\d])',
    # numbers (real, integers and in words)
    '_NUM_': r'''(?x) (?:[\-\+]\s*)?\d+(?:\.\d+)?(?=[^\>])
                    | _NUM_\d
                    | (?<=\b)([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive
                         |[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine
                         |[Tt]en|[Ee]leven|[Tt]welve|[A-Za-z]+teen
                         |[Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]ighty|[Nn]inety)(?=\W)
                    | _NUM_\s?([Hh]undred|[Tt]housand)(\s(and\s)?_NUM_)?
                    | ([Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty
                         |[Ss]ixty|[Ss]eventy|[Ee]igthty|[Nn]inety)
                      [\s\-]?_NUM_
                    | _NUM_\-_NUM_''', 
    # urls
    '_URL_': r'(?:http\:\/\/)?www\..+\.(?:com|org|gov|net|edu)',
    # dates
    '_DATE_': r'''(?x)
                    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?
                        |Apr(?:il)?|May|June?|July?
                        |Aug(?:ust)?|Sep(?:tember)?
                        |Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
                    (\s*[0123]?\d[\,\s+])?
                    (?:\s*(?:19|20)\d{2})? 
                    (?=\W+)                     #ie: January 01, 2011
                  | [0123]?\d\s*
                    (?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?
                        |[Aa]pr(?:il)?|[Mm]ay|[Jj]une?|[Jj]uly?
                        |[Aa]ug(?:ust)?|[Ss]ep(?:tember)?
                        |[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)
                    (?:[\,\s+]\s*(?:19|20)\d{2})?       #ie: 12 Jan 2011
                  | [0123]?\d[\-\/]
                      [01]?\d[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/01/2001
                  | [0123]?\d[\-\/]
                      (?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul
                        |[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[\-\/]
                      (?:19|20)?\d{2}           #ie: 12/jan/2001''',
    # periods of time
    '_POFT_': r'''(?x) \d{1,3}\-(?:[Mm]inutes?|[Hh]ours?|[Dd]ays?|[Ww]eeks?
                                 |[Mm]onths?|[Yy]ears?)(?=[\s\W])
                      | (?<=\W)_NUM_\-?(?=\,?\s(_NUM_\-?\,?\s)?(to|or)\s_POFT_)''',
    # arm one references
    '_GONE_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*([1IiAa]|[Oo]ne)(?=[\s\W])
                    | (?<=\W)(?:1st|[Ff]irst|[Ii]ntervention|[Oo]ne|[Ss]tudy)\s+
                        (?:[Aa]rm|[Gg]roup)(?=[\s\W])''',
    # arm two references
    '_GTWO_': r'''(?x) (?:[Aa]rm|[Gg]roup)\s*(?:[2Bb]|II|ii|[Tt]wo)(?=[\s\W])
                    | (?:2nd|[Ss]econd|[Cc]ontrol|[Pp]lacebo)\s+
						(?:[Aa]rm|[Gg]roup)(?=[\s\W])
                    | (?<=\_GONE\_\sand\s)([2Bb]|II|ii)(?=[\s\W])
                    | (?<=\_GONE\_\,\s)([2Bb]|II|ii)(?=[\s\W])''',
    # ratios
    '_RATIO_': r'''(?x) (\_NUM\_|\_RATIO\_)[\:\/]\_NUM\_
                   | _NUM_\sof\s_NUM_''',
    # other
    'mmHg':    r'mm[\s\/]*[Hh][Gg]',
    ' ': r'(\-|\s+)'
}                       

pat_ordered = ['mmHg','_CONFINT_','_CONFINTM_','_GONE_','_GTWO_',
               '_DATE_','_POFT_','_URL_','_POP_','_RATIO_','_RANGE_',
               '_RANGEM_','_PVAL_','_TIME_','_MEAS_','_PERC_',
               '_YEAR_','_NUM_',' ']


def normalise_sentence(sent):
    '''
    given string 'sent' expands abbreviations and substitutes patterns
    with normalisation tags
    '''
    
    sent = expand_abbreviations(sent)
    for key in pat_ordered*3:
        sent = re.sub(patterns[key],key,sent)
    return sent



# POS tagging and creating training set

def remove_paratag(paragraph):
    '''
    removes paragraph tags from paragraph
    '''
    pattern = r'\<abstracttext.*\"\>(.*)\<\/abstracttext\>'
    return re.findall(pattern,paragraph)

def take_paragraph_label(paragraph):
    '''
    returns the paragraph label
    '''
    pattern = r'\<abstracttext.*label\=\"(.*)\"\snlmcategory.*\>'
    return re.findall(pattern,paragraph)[0]

def take_paragraph_category(paragraph):
    '''
    returns the paragraph category
    '''
    pattern = r'\<abstracttext.*nlmcategory\=\"(.*)\"\>'
    return re.findall(pattern,paragraph)[0]

def sentence_preprocessing(sent):
    '''
    normalises 'sent' and transforms the tagging "<tag>...</tag>" in
    a list of labels
    '''
    tags = r'(\<\/?(?:p|oc|a|r)[12]?\>)'    
    sent_splitted = re.split(tags,sent)
    sent_plain = nltk.clean_html(sent)
    sent_plain = normalise_sentence(sent_plain)
    
    norm_sent = list()
    tagging = list()
    tag = 'O'
    for chunk in sent_splitted:
        if re.search(tags,chunk):
            tag = re.findall(r'\<(.+)\>',chunk)[0]
            if re.search(r'\/.',tag):
                tag = 'O'
        else:
            norm_chunk = normalise_sentence(chunk)            
            for word in nltk.word_tokenize(norm_chunk):
                tagging.append(tag.upper())
                norm_sent.append(word)
    
    i = 0 
    for tag in tagging:
        if tag == 'O':
            tagging[i] = 'O'
        elif i==(len(tagging)-1) or tagging[i+1] == 'O':
            tagging[i] = tag
        else:
            tagging[i] = 'O'
        i += 1
    return [norm_sent,tagging,sent_plain]



def sent_to_pos(sent):
    '''
    substitutes some normalisation tags in 'sent' with appropriate strings
    for the part of speech tagging
    '''
    out = list()            
    for tok in sent:
        tok = re.sub(r'_URL_','www.abc.com',tok)
        tok = re.sub(r'_G(ONE|TWO)_','group',tok)    
        tok = re.sub(r'_POFT_','1-year',tok)
        tok = re.sub(r'\_[A-Z]+\_','999',tok)
        out.append(tok)
    return out




def sentence_preprocessing_chunks(sent):
    '''
    chunking, pos tagging and chunk categorisation of string 'sent'
    '''
    foo, tokens_labels, sent_to_genia = sentence_preprocessing(sent)
    #- chunking and pos-tagging
    sent_chunked_iob = []
    for i in genia_tagger_instance.process(sent_to_genia):
        sent_chunked_iob.append(((i['word'],i['POStag'],i['chunktag'])))
    sent_chunked = nltk.chunk.util.conlltags2tree(sent_chunked_iob)
    
    #- chunk categorization
    chunks = []
    for subt in sent_chunked:
        try:
            n = subt.node
        except AttributeError:
            chunks.append(('O',[tuple(subt)]))
        else:
            chunks.append((n,list(subt)))   
    tokens_sem_class = []
    chunks_out = []
    for c in chunks:
        cstring = ' '.join([w[0] for w in c[1]])
        sem_class = (get_semantic_class(cstring) if c[0] in ['NP','ADVP']
                     else 'none')
        tokens_sem_class += [sem_class]*len(c[1])
        chunks_out += [c]*len(c[1])
   
   #- removes punctuation and prepares output
    tokens_out = []
    tokens_labels_out = []
    tokens_sem_class_out = []
    for c,l,s in zip(sent_chunked_iob,tokens_labels,tokens_sem_class):
        if not re.search(r'^[\,\.\;]',c[0]):
            tokens_out.append(c)
            tokens_labels_out.append(l)
            tokens_sem_class_out.append((s if not c[0] in sw else 'none'))
        
    return [tokens_out,tokens_labels_out,tokens_sem_class_out,chunks_out]



def preprocess_file(filename,verbose=True):
    '''
    preprocesses the .xml file in path 'filename', saved in the format
    returned by takeAbstract.take
    returns a list of dictionaries of token features, one for each token
    in the text
    '''
    
    # opens, processes the file, tokenises in sentences and retrieves
    # paragraph labels and categories
    f = open(filename).read()
    soup = bs4.BeautifulSoup(f,"html5lib")
    title = take_title(soup.pmid.text.encode('utf-8'))
    title = [w.lower() for w in nltk.word_tokenize(normalise_sentence(title))]
    if len(soup.abstracttext.attrs)==0:
        soup.abstracttext['label'] = 'None'
        soup.abstracttext['nlmcategory'] = 'None'        
    
    sentences = [(take_paragraph_label(child.encode('utf-8')),
                  take_paragraph_category(child.encode('utf-8')),
               nltk.sent_tokenize(remove_paratag(child.encode('utf-8'))[0]))
               for child in soup.find('abstract').children 
               if not child=='\n' 
                   and 'abstracttext' in child.encode('utf-8')]
    
    # builds the dictionaries list
    train = []  
    j = len(sentences)
    for (plabel,pcategory,para) in sentences:
        if verbose: print j,        
        
        k = 0        
        for sent in para:
            [tokens, labels, 
             sem_classes,chunks] = sentence_preprocessing_chunks(sent)         
            
            i = 0
            for t,l,s,c in zip(tokens, labels, sem_classes, chunks):
                change_to_cd = ['_CONFINT_','_DATE_','_POP_','_RATIO_',
                                '_RANGE_','_PVAL_','_TIME_','_MEAS_',
                                '_PERC_','_YEAR_','_NUM_']
                pos_tag =  (t[1] if t[0] not in change_to_cd else 'CD')
                
                train.append({'word': t[0],
                              'pos-tag': pos_tag,
                              'paragraph_l': plabel,
                              'paragraph_c': pcategory,
                              'position': (i,len(tokens)-1),
                              'sentence-position': (k,len(para)-1),
                              'chunk': c,
                              'sentence': tokens,
                              'semantic-class': s,
                              'sem-classes': sem_classes,
                              'label': l,
                              'file': filename,
                              'title': title})
                i += 1
            k += 1
        j -= 1
    
    return train      
        


def preprocess_data(preprocess_from = [], load_dat = []):
    '''
    # creates a list of preprocessed files from path 'preprocess_from', or
    # reads an already preprocessed list from file 'load_dat'
    '''
    data_raw = []
    if load_dat:
        preprocessed_dat = shelve.open(load_dat)
        for abst in preprocessed_dat.itervalues():
            data_raw.append(abst)
        preprocessed_dat.close()
    elif preprocess_from:    
        rootdir = preprocess_from
        files = list()
        for f in os.walk(rootdir):
            files.append(f)
        
        suffix = int(time.time())
        preprocessed_dat = shelve.open(r'./data/preprocessed'+
                                        str(suffix)+
                                        re.findall(r'([Aa]nnotation .+)\/',
                                                    preprocess_from)[0]+
                                        '.dat')
        i = 0
        for filename in files[0][2]:
            print '#-- %d/%d - %s ...' % (i+1, len(files[0][2]), filename),
            file_path = files[0][0]+'/'+filename
            preprocessed_abstract = preprocess_file(file_path)
            data_raw.append(preprocessed_abstract)
            preprocessed_dat[filename] = preprocessed_abstract
            print 'preprocessed --#'
            i += 1
        preprocessed_dat.close()   
    else:
        raise Exception('ehm.. what?')
    return data_raw




#### test
#
#rootdir = r'./data/annotated/'
#files = list()
#for f in os.walk(rootdir):
#    files.append(f)
#
#for f in files[0][2]:
#    a = preprocess_file(files[0][0]+f,verbose=False)
#    print '\n#-- %s --#' %f    
#    for i in a:
#        if i['position'][0] == 0: print
#        print '%s(%s,%s,%s,%s,%s,(%d,%d)|%s)' %(i['word'],
#                                                 i['pos-tag'],
#                                                 i['chunk'][0],
#                                                 i['semantic-class'],
#                                                 i['paragraph_l'],
#                                                 i['paragraph_c'],
#                                                 i['position'][0],
#                                                 i['position'][1],
#                                                 i['label']),
#
#
#for f in files[0][2][files[0][2].index('11158795.xml'):]:
#    a = preprocess_file(files[0][0]+f,verbose=False)    
#    print '\n\n#-- %s --#' %f    
#    for i in a:
#        if i['position'][0] == 0: print
#        print '%s%s' %(i['word'],('('+i['label']+')' if i['label']!='O'
#                                            else '')),     
#    print   
#
#
#
#a = []
#for filename in files[0][2]:
#    print filename,    
#    f = open(files[0][0]+filename).read()
#    soup = bs4.BeautifulSoup(f,"html5lib")
#    if len(soup.abstracttext.attrs)==0:
#        soup.abstracttext['label'] = 'None'
#        soup.abstracttext['nlmcategory'] = 'None'        
#    sentences = [(take_paragraph_label(child.encode('utf-8')),
#                  take_paragraph_category(child.encode('utf-8')),
#               nltk.sent_tokenize(remove_paratag(child.encode('utf-8'))[0]))
#               for child in soup.find('abstract').children 
#               if not child=='\n' 
#                   and 'abstracttext' in child.encode('utf-8')]
#    a.append((filename,[(i[0],i[1]) for i in sentences]))
#
### test end
