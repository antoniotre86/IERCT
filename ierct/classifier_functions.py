# -*- coding: utf-8 -*-
'''
Created on 15 Feb 2014

@author: Antonio
'''


from __future__ import division
import nltk, re, os, shelve, time, datetime, pickle
from numpy import *
import preprocessing_functions as pre
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from gurobipy import *
from preprocessing_functions import preprocess_data



# lemmatiser instance
lem = nltk.WordNetLemmatizer()

# stopword list
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))

# global variables
OPEN_BRACKET = False



# Imports
#- Vocabularies
V = open(r".\data\vocabulary.txt").read()
V = nltk.word_tokenize(V)
V.append('UNK')
V = set(V)
K = len(V)

norm_tags = [t.lower() for t in pre.patterns.keys()]
norm_tags_types = {t:i for i,t in enumerate(norm_tags)}
norm_tags_types['_range_'] = norm_tags_types['_rangem_']
norm_tags_types['_confint_'] = norm_tags_types['_confintm_']
norm_tags_types['_num_'] = norm_tags_types['_meas_']



#- Paragraphs
paragraph_labels = {'Objective': ['background',
                                  'objective',
                                  'purpose',
                                  'aim'],
                    'Methods': ['method',
                                'design',
                                'setting',
                                'material',
                                'registration'],
                    'Patients': ['patient',
                                 'participant',
                                 'subject'],
                    'Intervention': ['intervention'],
                    'Main Outcome': ['outcome'],
                    'Results': ['result',
                                'statistics'],
                    'Conclusions': ['conclusion'
                                    'financial']
              }

paragraph_categories = ['METHODS', 
                        'CONCLUSIONS', 
                        'RESULTS', 
                        'BACKGROUND', 
                        'OBJECTIVE']

Pc0 = ['None', 'METHODS', 'BACKGROUND', 'OBJECTIVE']
Pc1 = ['None', 'RESULTS']
                        
#- Pos-tags to take out
posto = ["''", '(', ')', ':','CC','DT','EX','FW','IN','LS','PRP$',
         'WDT','WP','RP','TO','PRP','WRB','PDT','WP$','MD','JJR','JJS']

#- Labels
L = ['O','P','A1','A2','OC','R1','R2']
l = len(L)



# Features
#- chunk bag of words
def f_cbow(tok):
    lemmas = [lem.lemmatize(w[0].lower()) for w in tok['chunk'][1]
                if not w[0] == tok['word']]
    f = {'"%s"-in-chunk' %l:True for l in lemmas if l in V-sw}    
    return [f,lemmas]    

#- sentence bag of words
def f_sbow(tok):
    lemmas = [lem.lemmatize(w[0].lower()) for w in tok['sentence']
                if not w[0] == tok['word']]
    f = {'"%s"-in-sent' %l:True for l in lemmas 
							if l in (V-sw)-(set(f_cbow(tok)[1])-sw)}    
    return f       

#- semantic classes in sentence
def f_scinsent(tok):
    sem_classes = ( set(tok['sem-classes'])  
                    - set([tok['semantic-class']])
                    - set(['none']) )
    f = {'%s-in-sent' %sc:True for sc in sem_classes}
    return f

# paragraph label
def f_paral(tok):
    for p_class in paragraph_labels.iteritems():
        for pp in p_class[1]:
            if re.search(pp,tok['paragraph_l'].lower()):
                return p_class[0]
    return 'None'

# position in sentence
def f_pis(tok):
    f = (int(10*tok['position'][0]/tok['position'][1])
                if tok['position'][1]>0
                else 0)
    return f

# sentence position in paragraph
def f_spp(tok):
    f = (int(10*tok['sentence-position'][0]/tok['sentence-position'][1])
            if tok['sentence-position'][1]>0
            else 0)
    return f
    
# position in chunk
def f_pic(tok):
    if tok['chunk'][0] == 'NP' and tok['word'] == tok['chunk'][1][-1][0]:
        return True

# inside brackets
def f_inbrackets(tok):
    global OPEN_BRACKET
    f = bool(OPEN_BRACKET)
    if tok['word'] == '(': 
        OPEN_BRACKET=True
    if tok['word'] == ')': 
        OPEN_BRACKET=False
        f = bool(OPEN_BRACKET)
    return f

def check_position(tok,position):
    f = ( tok['position'][0] + position == 0,
          tok['position'][0] + position == tok['position'][1])
    return f


# feature extractor
def feature_extractor(abstract, i):
    features = {}
    tok = abstract[i]
    
    (isstart,isend) = check_position(tok,0)
    isstart_prev = check_position(tok,-1)
    isend_next = check_position(tok,+1)
    if not isstart: 
        tok_prev = abstract[i-1]
        if not isstart_prev: tok_prev_prev = abstract[i-2]
    if not isend: 
        tok_next = abstract[i+1]
        if not isend_next: tok_next_next = abstract[i+2]
    
    # word features
    features['word'] = tok['word'].lower()
    if not isstart: 
        features['word(-1)'] = tok_prev['word'].lower()
        features['2gram_back'] = (tok_prev['word'].lower()+'+'+ 
                                  tok['word'].lower())
        if not isstart_prev:
            features['word(-2)'] = tok_prev_prev['word'].lower()    
    if not isend: 
        features['word(+1)'] = tok_next['word'].lower()
        features['2gram_ahead'] = (tok['word'].lower()+'+'+ 
                                    tok_next['word'].lower())
        if not isend_next:
            features['word(+2)'] = tok_next_next['word'].lower() 
    
    # chunk bow    
    features.update(f_cbow(tok)[0])
    
    # sent bow
    features.update(f_sbow(tok))
    
    # semantic class
    features['sem-class(0)'] = tok['semantic-class']
    if not isstart: features['sem-class(-1)'] = tok_prev['semantic-class']
    if not isend: features['sem-class(+1)'] = tok_next['semantic-class']
    
    # sem class in sentence
    features.update(f_scinsent(tok))    
    
    # chunk type
    features['chunk-type(0)'] = tok['chunk'][0]
    if not isstart: 
        features['chunk-type(-1)'] = tok_prev['chunk'][0]
    if not isend:
        features['chunk-type(+1)'] = tok_next['chunk'][0]
    
    # POS tags
    features['pos-tag'] = tok['pos-tag']
    if not isstart: features['pos-tag(-1)'] = tok_prev['pos-tag']
    if not isend: features['pos-tag(+1)'] = tok_next['pos-tag']
    
     # paragraph category
#      features['paragraph-cat'] = tok['paragraph_c']    
#      if features['paragraph-cat'] == 'None': 
#          features['paragraph-cat'] = paragraph_categories[int(i*
#  													4/len(abstract))]    
    features['paragraph-cat'] = '' 
     
     # paragraph label
#      features['paragraph-lab'] = f_paral(tok)    
#      if features['paragraph-lab'] == 'None':
#          features['paragraph-lab'] = features['paragraph-cat']    
    features['paragraph-lab'] = ''
        
    # position in sentence
    features['position'] = f_pis(tok)
    
    # sentence position in paragraph
    features['sentence-position'] = f_spp(tok)
    
    # inside brackets
    features['inside-brackets'] = f_inbrackets(tok)
    
    # word in title
    features['word-in-title'] = tok['word'].lower() in tok['title']
    
    # Conjunctions
    features['paragraph&sem-class'] = (features['paragraph-cat']+'+'+
                                 features['sem-class(0)'])
                                 
    features['in-title&sem-class'] = (features['word-in-title']*
                                    features['sem-class(0)'])
                                     
    return features
    


def apply_features(data,feature_extractor,exclude=True):
    '''  
    # applies features in 'feature_extractor' to 'data' and returns a list
    # of training tokens and the indices of tokens that have been excluded
    '''
    tagged_abstracts = []
    untagged_abstracts = []
    tags = []
    excluded = []
    j = 0
    for abstract in data:
        tagged_abstract = []
        untagged_abstract = []
        abstract_tags = []
        excluded_tokens = []
        for i,tok in enumerate(abstract):
            excluding_cond = exclude and (tok['pos-tag'] in posto 
                                or tok['paragraph_c']=='CONCLUSIONS')
            if not excluding_cond:
                tag = tok['label'] if (tok['label'] in L) else 'O'
                feats = (feature_extractor(abstract,i))
                tagged_abstract.append((feats,tag))
                untagged_abstract.append(feats)
                abstract_tags.append(tag)
            else:
                excluded_tokens.append(i)
        if len(data)>1 and (j+1)%int(len(data)/10)==0: print '.',
        tagged_abstracts.append(tagged_abstract)
        untagged_abstracts.append(untagged_abstract)
        tags.append(abstract_tags)
        excluded.append(excluded_tokens)
        j += 1
    return [tagged_abstracts,untagged_abstracts,tags,excluded]   



class Classifier_zero:
    '''
    # zero classifier: classifies tokens without any constraint in the ILP
    # stage
    '''
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def train(self,train_data,verbose = True,**kwargs):
        start = time.time()
        if verbose: print '#---- Training...',
        samples, labels = zip(*[i for a in train_data for i in a])
        v = DictVectorizer()        
        self.vectorize = v.fit(samples)
        X = self.vectorize.fit_transform(samples)
        
        # maximum entropy model
        maxent = linear_model.LogisticRegression(**kwargs)
        self.classifier = maxent.fit(X,labels)
        self.labels = self.classifier.classes_
        if verbose: print '... trained, time elapsed: %s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]
    
    def batch_tagger(self,test_data,excluded, dw, oc_con=True, verbose = True):
        '''
        # returns the predicted tagging of abstracts in 'test_data'
        '''
        if verbose: print '#------ Tagging...'
        tagged_out = []
        j = 0
        for abst,ex in zip(test_data,excluded):
            start = time.time()
            if verbose: print '\t- %d/%d,' % (j+1,len(test_data)), 
            untagged_abst = [tok[0] for tok in abst]  
            tags = self.optim_from_probs(untagged_abst)
            for i in ex:
                tags.insert(i,'O')
            tagged_out.append(tags)
            if verbose: print '%s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]          
            j += 1
        return tagged_out
    
    def batch_probabilities(self,test_data):
        out = []
        for a in test_data:
            out.append(self.probabilities(a))
        return out
    
    def probabilities(self,abstract):
        X = self.vectorize.transform(abstract)        
        out = zeros((len(abstract),len(self.classifier.classes_)))
        for i,t in enumerate(abstract):
            out[i,:] = self.classifier.predict_log_proba(X[i,:])
        return out
    
    def optim_from_probs(self, abstract, write_file = []):
        '''
        # ILP problem 
        '''
        
        probs = self.probabilities(abstract)        
        L = self.labels        
                
        h = tuplelist([(i,l) for i,p in enumerate(probs) for l in L])
        
        # Gurobi model
        m = Model('pippo')
        
        # variables
        X = {}
        for i,p in enumerate(probs):
            for j,l in enumerate(L):
                X[i,l] = m.addVar(obj = p[j], 
                                  vtype = GRB.BINARY, 
                                  name = 'x_%d_%s' %(i,l))
        
        m.update()
        
        # constraints
        #-- 00: one label per token 
        for i,l in h:
            m.addConstr(quicksum(X[i,l] for l in L) == 1, 
                        name = 'con00_%d' %i)
        
        m.update()
        
        # save model
        if write_file: m.write(write_file)
        
        # optimize
        m.setParam(GRB.param.OutputFlag, 0)
        m.modelSense = GRB.MAXIMIZE
        m.optimize()
        
        # solution
        out = []
        sol = m.getAttr('x', X)
        for i in range(len(probs)):
            for l in self.classifier.classes_:    
                if round(sol[i,l]) == 1: out.append(l)
        
        return out



class Classifier_vanilla:
    '''
    # vanilla classifier: includes constraint #01 in the ILP problem
    '''
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def train(self,train_data,verbose = True,**kwargs):
        start = time.time()
        if verbose: print '#---- Training...',
        samples, labels = zip(*[i for a in train_data for i in a])
        v = DictVectorizer()        
        self.vectorize = v.fit(samples)
        X = self.vectorize.fit_transform(samples)
        
        # maximum entropy model
        maxent = linear_model.LogisticRegression(**kwargs)
        self.classifier = maxent.fit(X,labels)
        self.labels = self.classifier.classes_
        if verbose: print '... trained, time elapsed: %s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]
    
    def batch_tagger(self,test_data,excluded, dw, oc_con=True, verbose = True):
        '''
        # returns the predicted tagging of abstracts in 'test_data'
        '''
        if verbose: print '#------ Tagging...'
        tagged_out = []
        j = 0
        for abst,ex in zip(test_data,excluded):
            start = time.time()
            if verbose: print '\t- %d/%d,' % (j+1,len(test_data)), 
            untagged_abst = [tok[0] for tok in abst]  
            tags = self.optim_from_probs(untagged_abst)
            for i in ex:
                tags.insert(i,'O')
            tagged_out.append(tags)
            if verbose: print '%s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]          
            j += 1
        return tagged_out
    
    def batch_probabilities(self,test_data):
        out = []
        for a in test_data:
            out.append(self.probabilities(a))
        return out
    
    def probabilities(self,abstract):
        X = self.vectorize.transform(abstract)        
        out = zeros((len(abstract),len(self.classifier.classes_)))
        for i,t in enumerate(abstract):
            out[i,:] = self.classifier.predict_log_proba(X[i,:])
        return out
    
    def optim_from_probs(self, abstract, write_file = []):
        '''
        # ILP problem
        '''
        
        probs = self.probabilities(abstract)        
        L = self.labels        
                
        h = tuplelist([(i,l) for i,p in enumerate(probs) for l in L])
        
        # Gurobi model
        m = Model('pippo')
        
        # variables
        X = {}
        for i,p in enumerate(probs):
            for j,l in enumerate(L):
                X[i,l] = m.addVar(obj = p[j], 
                                  vtype = GRB.BINARY, 
                                  name = 'x_%d_%s' %(i,l))
        
        m.update()
        
        # constraints
        #-- 00: one label per token 
        for i,l in h:
            m.addConstr(quicksum(X[i,l] for l in L) == 1, 
                        name = 'con00_%d' %i)
        
        #-- 01: one label per abstract
        for l in set(L)-set('O'):
            m.addConstr(quicksum(X[i,l] for i,p in enumerate(probs)) == 1,
                        name = 'con01_%s' %l)
        
        m.update()
        
        # save model
        if write_file: m.write(write_file)
        
        # optimize
        m.setParam(GRB.param.OutputFlag, 0)
        m.modelSense = GRB.MAXIMIZE
        m.optimize()
        
        # solution
        out = []
        sol = m.getAttr('x', X)
        for i in range(len(probs)):
            for l in self.classifier.classes_:    
                if round(sol[i,l]) == 1: out.append(l)
        
        return out



class Classifier:
    '''
    # full model: classifies tokens using all the constraints in the ILP problem
    '''
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def train(self,train_data,verbose = True,**kwargs):
        start = time.time()
        if verbose: print '#---- Training...',
        samples, labels = zip(*[i for a in train_data for i in a])
        v = DictVectorizer()        
        self.vectorize = v.fit(samples)
        X = self.vectorize.fit_transform(samples)
        
        # maximum entropy model
        maxent = linear_model.LogisticRegression(**kwargs)
        self.classifier = maxent.fit(X,labels)
        self.labels = self.classifier.classes_
        if verbose: print '... trained, time elapsed: %s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]
    
    def batch_tagger(self,test_data,excluded, dw, oc_con=True, verbose = True):
        '''
        # returns the predicted tagging of abstracts in 'test_data'
        '''
        if verbose: print '#------ Tagging...'
        tagged_out = []
        j = 0
        for abst,ex in zip(test_data,excluded):
            start = time.time()
            if verbose: print '\t- %d/%d,' % (j+1,len(test_data)), 
            untagged_abst = [tok[0] for tok in abst]  
            self.tags = self.optim_from_probs(untagged_abst, dw, oc_con)
            for i in ex:
                self.tags.insert(i,'O')
            tagged_out.append(self.tags)
            if verbose: print '%s' %str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]          
            j += 1
        return tagged_out
    
    def batch_probabilities(self,test_data):
        out = []
        for a in test_data:
            out.append(self.probabilities(a))
        return out
    
    def probabilities(self,abstract):
        X = self.vectorize.transform(abstract)        
        out = zeros((len(abstract),len(self.classifier.classes_)))
        for i,t in enumerate(abstract):
            out[i,:] = self.classifier.predict_log_proba(X[i,:])
        return out
    
    def optim_from_probs(self, abstract, dw, oc_con, write_file = []):
        '''
        # ILP problem        
        '''
        
        probs = self.probabilities(abstract)        
        L = self.labels        
                
        pcs = tuplelist([(i,a['paragraph-cat']) for i,a in enumerate(abstract)])
        h = tuplelist([(i,l) for i,p in enumerate(probs) for l in L])
        
                
        # Gurobi model
        m = Model('pippo')
        
        # variables
        #- decision variables
        X = {}
        for i,p in enumerate(probs):
            for j,l in enumerate(L):
                X[i,l] = m.addVar(obj = p[j], 
                                  vtype = GRB.BINARY, 
                                  name = 'x_%d_%s' %(i,l))
        
        #- auxiliary variables
        WL = {}
        for l in ['R1','R2']:
            WL[l] = m.addVar(vtype = GRB.INTEGER, name = 'w_%s' %l)
        
        Z = {}
        for l in set(L)-set('O'):
            Z[l] = m.addVar(vtype = GRB.INTEGER, name = 'z_%s' %l)        
        
        D = {}
        D['A1','A2'] = m.addVar(obj = -dw[0], vtype = GRB.INTEGER, name = 'd_A1_A2')
        D['R1','R2'] = m.addVar(obj = -dw[1], vtype = GRB.INTEGER, name = 'd_R1_R2')        
        
        Q = {}
        for l in ['OC','R1','R2']:
                Q[l] = m.addVar(vtype = GRB.INTEGER, name = 'q_%s' %l)
        
        Y = {}
        Y[0] = m.addVar(vtype = GRB.BINARY, name = 'y_0')
        Y[1] = m.addVar(vtype = GRB.BINARY, name = 'y_1')
        
        B = {}
        B[0] = m.addVar(vtype = GRB.BINARY, name = 'b_0')
        B[1] = m.addVar(vtype = GRB.BINARY, name = 'b_1')
        
        m.update()
        
        # constraints
        #-- auxiliary constraints
        #--- WL == index of "_x_" words (or else 101)
        for l in ['R1','R2']:
            m.addConstr(WL[l] == quicksum(X[i,l]*(norm_tags_types[abstract[i]['word'].lower()]
                                                  if abstract[i]['word'].lower() in norm_tags
                                                  else 101)                                         
                                for i in range(len(probs))),
                                name = 'aux00_%s' %l)
        
        #--- Z[l] == index of token labelled as l
        for l in set(L)-set('O'):
            m.addConstr(Z[l] == quicksum(X[i,l]*i for i in range(len(probs))),
                                        name = 'aux01_%s' %l)        
        
        #--- D == distance between A1 and A2 or R1 and R2
        m.addConstr(D['A1','A2'] == Z['A2'] - Z['A1'], name = 'aux03_0')
        m.addConstr(D['R1','R2'] == Z['R2'] - Z['R1'], name = 'aux03_1')  
        
        #--- Q[l] == position of sentence containing token labelled as l
        for l in ['OC','R1','R2']:
            m.addConstr(Q[l] == quicksum(X[i,l]*abstract[i]['sentence-position']
                                            for i in range(len(probs))), 
                        name = 'aux04_%s' %l)
        
        #--- Y
        m.addConstr(Y[0] + Y[1] == 1, name = 'aux05')
        
        #--- B
        m.addConstr(B[0] + B[1] == 1, name = 'aux06')
        
        #-- 00: one label per token 
        for i,l in h:
            m.addConstr(quicksum(X[i,l] for l in L) == 1, 
                        name = 'con00_%d' %i)
        
        #-- 01: one label per abstract
        for l in set(L)-set('O'):
            m.addConstr(quicksum(X[i,l] for i,p in enumerate(probs)) == 1,
                        name = 'con01_%s' %l)
        
        #-- 02: A1 before A2
        m.addConstr(Z['A2'] - Z['A1'] >= 1, name = 'con02')
        
        #-- 03: last of {A1,A2,P} before first of {OC,R1,R2}
        for l0 in ['A1','A2','P']:
            for l1 in ['OC','R1','R2']:
                m.addConstr(Z[l1] - Z[l0] >= 1, name = 'con03_%s_%s' %(l0,l1))
        
#         #-- 04: R1/R2 in 'Results' or 'None'
#         if oc_con: 
#             L04 = ['OC','R1','R2']
#         else:
#             L04 = ['R1','R2']
#         for i,pc in pcs:
#             if pc not in Pc1:
#                 for l in L04:
#                     m.addConstr(X[i,l] == 0, name = 'con04_%d' %i)    
#                 
#         #-- 05: A1/A2/P in ['Background','Methods','Objective','None']
#         for i,pc in pcs:
#             if pc not in Pc0:
#                 for l in ['A1','A2','P']:
#                     m.addConstr(X[i,l] == 0, name = 'con05_%d' %i)
        
        #-- 06
        m.addConstr((Z['A1'] - Z['P'])*Y[0] >= 0, name = 'con06_0')
        m.addConstr((Z['P'] - Z['A2'])*Y[1] >= 0, name = 'con06_0')
        
        #-- 07 R1, R2 == "_x_"
        for l in ['R1','R2']:
            m.addConstr(WL[l] <= 100, name = 'con08_%s' %l)
        
        #-- 08 R1 == R2
        m.addConstr(WL['R1'] == WL['R2'], name = 'con07')
        
#         #-- 09 OC in same sentence as R1
#         if oc_con:
#             m.addConstr((Q['OC'] - Q['R1'])*B[0] == 0, name = 'con09_0')
#             m.addConstr((Q['OC'] - Q['R2'])*B[1] == 0, name = 'con09_1')
        
        m.update()
        
        # save model
        if write_file: m.write(write_file)
        
        # optimize
        m.setParam(GRB.param.OutputFlag, 0)
        m.modelSense = GRB.MAXIMIZE
        m.optimize()
        
        # solution
        out = []
        sol = m.getAttr('x', X)
        for i in range(len(probs)):
            for l in self.classifier.classes_:    
                if round(sol[i,l]) == 1: out.append(l)
        
        return out



def print_output(gold, output):
    '''
    # prints the classifier's output, 'gold' is the actual data in the format
    # returned by preprocess_data
    '''
    for g,o in zip(gold,output):
        print '\n\n#-- %s --#' %g[0]['file']
        print '# %s #' %' '.join(g[0]['title'])
        for tg,to in zip(g,o):
            print '%s%s' %(tg['word'],('('+tg['label']+'|'+to+')' 
                                        if (tg['label']!='O' 
                                            or to!='O')
                                            and tg['label']!=to
                                        else ('('+to+')' if
                                              to!='O'
                                              else ''))),


def load_classifier(model_id):
    '''
    # loads a classifier model from file
    '''
    f = open('saved models/'+model_id+'.pickle')
    return pickle.load(f)


def save_model(model,obj_type,filename):
    '''
    # saves a model in file
    '''
    if obj_type == 'cv':
        f = shelve.open('saved models/'+filename+'.dat')
        f['precision'] = model.precision
        f.close()
        print 'Cross Validation saved in %s' %('saved models/'+filename+'.dat')
        return
    if obj_type == 'cl':
        f = open('saved models/'+filename+'.pickle','wb')
        pickle.dump(model,f)
        print 'Classifier saved in %s' %('saved models/'+filename+'.pickle')



def demo():
    data_raw4 = preprocess_data(load_dat = './data/preprocessed1375357188annotation IV.dat')
    f = shelve.open(r'./data/preprocessed1375357188annotationIV_featured.dat')
    [tagged_abstracts4,untagged_abstracts4,
     tags4,excluded4] = [f['tagged_abstracts4'], f['untagged_abstracts4'],
                         f['tags4'], f['excluded4']]
    f.close()
    dataset4 = [list(a) for a in tagged_abstracts4]
    ntrain = int(len(data_raw4)*0.75)
    dev_set4 = dataset4[:ntrain]
    test_set4 = dataset4[ntrain:]
    
    model = Classifier(feature_extractor)
    model.train(dev_set4,C=0.9)
    model_out = model.batch_tagger(test_set4, excluded4[ntrain:], dw=[5e-4,5e-4])  
    
    print_output(data_raw4[ntrain:],model_out)
