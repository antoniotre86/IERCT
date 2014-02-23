# -*- coding: utf-8 -*-
'''
Created on 15 Feb 2014

@author: Antonio
'''


from __future__ import division
import time, datetime


class HoldOut:
    '''
    Hold Out test of 'classifier' in 'test set'
    '''
    
    def __init__(self,classifier,test_set,excluded,dw=[0,0],oc_con = True, 
                 verbose = True):
        '''
        'classifier' is an object of class classifier_functions.Classifier or 
        classifier_functions.Classifier_vanilla or 
        classifier_function.Classifier_zero
        '''
        
        test_set_tags = [[t[1] for t in a] for a in test_set]
        for a,e in zip(test_set_tags,excluded):
            for j in e:
                a.insert(j,'O')
        test_set_tags = [t for a in test_set_tags for t in a]
        out = classifier.batch_tagger(test_set, excluded, dw, oc_con, verbose)
        out_tags = [tag for a in out for tag in a] 
        
        #self.confmatrix = nltk.ConfusionMatrix(out_tags, test_set_tags)
        self.labels = set(test_set_tags)
        self.TP = {}        
        self.precision = {}
        
        for lab in set(self.labels)-set('O'):
            self.TP[lab] = len([1 for i,j in zip(test_set_tags,out_tags)
                            if i==lab and i==j]) # True Positives
        for lab in set(self.labels)-set('O'):
            self.precision[lab] = self.TP[lab]/len(test_set)
        
    def tabulate_evaluation_measures(self):
        '''
        prints the precisions in a readable format
        '''
        output = {'precision': self.precision}
        print
        print '%-10s' % 'Labels',
        for lab in set(self.labels)-set('O'):
            print '%8s' % lab,
        print
        print '-'*(11+9*(len(self.labels)-1))
        for key in output.keys():
            print '%-10s' % key,
            for lab in set(self.labels)-set('O'):
                print '%8f' % round(output[key][lab],6),
            print
        print '-'*(11+9*(len(self.labels)-1))



class CrossValidation:
    '''
    cross-validation routine to evaluate 'classifier' in 'data'
    '''
    
    def __init__(self,classifier,data,excluded,folds,dw=[0,0],oc_con=True,
                 verbose=True,**kwargs):
        '''
        'classifier' is an object of class classifier_functions.Classifier or 
        classifier_functions.Classifier_vanilla or 
        classifier_function.Classifier_zero
        'data' is a list of examples as returned by classifier_functions.apply_features
        '''
        
        self.labels = set([i[1] for abst in data for i in abst])
        self.TP = {}
        for lab in self.labels:
            self.TP[lab] = 0
        indices = range(len(data))
        if verbose: print
        
        start = time.time()
        for k in xrange(folds):
            if verbose: print '#-- Fold: %d ...' % (k+1)
            ntest = int(len(data)/folds)
            idx_test = (range(k*ntest,(k+1)*ntest) if k<folds-1
                        else range(k*ntest,len(data)))
            idx_train = [i for i in indices if i not in idx_test]
            
            test_set = [list(data[i]) for i in idx_test]
            train_set = [data[i] for i in idx_train]
            ex = [excluded[i] for i in idx_test]
            test_set_tags = [[t[1] for t in a] for a in test_set]
            
            for a,e in zip(test_set_tags,ex):
                for j in e:
                    a.insert(j,'O')
            test_set_tags = [t for a in test_set_tags for t in a]
            
            classifier.train(train_set,verbose,**kwargs)
            out = classifier.batch_tagger(test_set, ex, dw, oc_con, verbose)
            out_tags = [tag for a in out for tag in a] 
            
            for lab in set(self.labels)-set('O'):
                self.TP[lab] = self.TP[lab] + len([1 for i,j in zip(test_set_tags,out_tags)
                        if i==lab and i==j]) # True Positives
            if verbose: print '... time elapsed: %s --#\n' % str(datetime.datetime.fromtimestamp(time.time()-start)).split()[1]
            
        self.precision = {}
        for lab in set(self.labels)-set('O'):
            self.precision[lab] = self.TP[lab]/len(data)
        
    def tabulate_evaluation_measures(self):
        '''
        prints the precisions in a readable format
        '''
        output = {'precision': self.precision}
        print '%-10s' % 'Labels',
        for lab in set(self.labels)-set('O'):
            print '%8s' % lab,
        print
        print '-'*(11+9*(len(self.labels)-1))
        for key in output.keys():
            print '%-10s' % key,
            for lab in set(self.labels)-set('O'):
                print '%8f' % round(output[key][lab],6),
            print
        print '-'*(11+9*(len(self.labels)-1))
        print
