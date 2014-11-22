# -*- coding: utf-8 -*-
'''
Created on 15 Feb 2014

@author: Antonio
'''

import sys, getopt, shelve, os

import preprocessing_functions as pre
import classifier_functions as clf
import classifier_evaluation as ce


def main(args):
    
    if args[0].has_key('-a'):
        annotation = args[0]['-a']
    else:
        annotation = '4'
    if args[0].has_key('-l'):
        filename_load = args[0]['-l']
        assert os.path.exists(filename_load), filename_load + ' does not exist.'
        preprocess_new = False
    else:
        preprocess_new = True
    save_models = args[0].has_key('-s')
    
    if annotation == '1':
        ## - annotation I
        if preprocess_new:
            data_raw1 = pre.preprocess_data(preprocess_from = './data/annotation I/')
        else:
            data_raw1 = shelve.open(filename_load)
        
        tagged_abstracts1,untagged_abstracts1,tags1,excluded1 = clf.apply_features(data_raw1.values(),clf.feature_extractor)
        
        dataset1 = [list(a) for a in tagged_abstracts1]
        ntrain = int(len(data_raw1)*0.75)
        dev_set1 = dataset1[:ntrain]
        test_set1 = dataset1[ntrain:]
        
        # vanilla
        maxent10 = clf.Classifier_vanilla(clf.feature_extractor)
        maxent10.train(dev_set1,C=0.9)
        maxent10_ho = ce.HoldOut(maxent10,test_set1,excluded1[ntrain:],dw=[5e-4,5e-4])
        
        maxent10_cv = ce.CrossValidation(maxent10,dev_set1,excluded1[:ntrain],10,C=0.9)
        
        maxent10_out = maxent10.batch_tagger(test_set1,excluded1[ntrain:], dw=[0,0])  
        
        # full
        maxent11 = clf.Classifier(clf.feature_extractor)
        maxent11.train(dev_set1,C=0.9)
        maxent11_ho = ce.HoldOut(maxent11,test_set1,excluded1[ntrain:],
                                 dw=[5e-4,5e-4],oc_con=False)
        
        maxent11_cv = ce.CrossValidation(maxent11,dev_set1,excluded1[:ntrain],10,dw=[5e-4,5e-4],oc_con=False,C=0.9)
        
        maxent11_out = maxent11.batch_tagger(test_set1, excluded1[ntrain:],dw=[5e-4,5e-4], oc_con=False)
        
        # zero
        maxent12 = clf.Classifier_zero(clf.feature_extractor)
        maxent12.train(dev_set1,C=0.9)
        maxent12_ho = ce.HoldOut(maxent12,test_set1,excluded1[ntrain:],dw=[5e-4,5e-4])
        
        maxent12_cv = ce.CrossValidation(maxent12,dev_set1,excluded1[:ntrain],10,dw=[5e-4,5e-4],C=0.9)
        
        # save models
        if save_models:
            clf.save_model(maxent10_ho,'cv','maxent10_ho')
            clf.save_model(maxent10_cv,'cv','maxent10_cv')
            clf.save_model(maxent11_ho,'cv','maxent11_ho')
            clf.save_model(maxent11_cv,'cv','maxent11_cv')
        
        # print results
        print '# Annotation I'
        print '#- Cross Validation'
        print '#-- Zero model'
        maxent12_cv.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent10_cv.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent11_cv.tabulate_evaluation_measures()
        print '#- Hold OUt'
        print '#-- Zero model'
        maxent12_ho.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent10_ho.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent11_ho.tabulate_evaluation_measures()
        print
    
    
    if annotation == '2':
        ## - annotation II
        
        if preprocess_new:
            data_raw2 = pre.preprocess_data(preprocess_from = './data/annotation II/')
        else:
            data_raw2 = shelve.open(filename_load)
        
        tagged_abstracts2,untagged_abstracts2,tags2,excluded2 = clf.apply_features(data_raw2.values(),clf.feature_extractor)
        
        dataset2 = [list(a) for a in tagged_abstracts2]
        ntrain = int(len(data_raw2)*0.75)
        dev_set2 = dataset2[:ntrain]
        test_set2 = dataset2[ntrain:]
        
        # vanilla
        maxent20 = clf.Classifier_vanilla(clf.feature_extractor)
        maxent20.train(dev_set2,C=0.9)
        maxent20_ho = ce.HoldOut(maxent20,test_set2,excluded2[ntrain:])
        
        maxent20_cv = ce.CrossValidation(maxent20,dev_set2,excluded2[:ntrain],10,C=0.9)
        
        maxent20_out = maxent20.batch_tagger(test_set2, excluded2[ntrain:], dw=[0,0])  
        
        # full
        maxent21 = clf.Classifier(clf.feature_extractor)
        maxent21.train(dev_set2,C=0.9)
        maxent21_ho = ce.HoldOut(maxent21,test_set2,excluded2[ntrain:],dw=[5e-4,5e-4],oc_con = False)
        
        maxent21_cv = ce.CrossValidation(maxent21,dev_set2,excluded2[:ntrain],10,dw=[5e-4,5e-4], oc_con = False, C=0.9)
        
        maxent21_out = maxent21.batch_tagger(test_set2, excluded2[ntrain:], dw=[5e-4,5e-4], oc_con=False)
        
        # zero
        maxent22 = clf.Classifier_zero(clf.feature_extractor)
        maxent22.train(dev_set2,C=0.9)
        maxent22_ho = ce.HoldOut(maxent22,test_set2,excluded2[ntrain:],dw=[5e-4,5e-4])
        
        maxent22_cv = ce.CrossValidation(maxent22,dev_set2,excluded2[:ntrain],10,dw=[5e-4,5e-4],C=0.9)
        
        # save models
        if save_models:
            clf.save_model(maxent20_ho,'cv','maxent20_ho')
            clf.save_model(maxent20_cv,'cv','maxent20_cv')
            clf.save_model(maxent21_ho,'cv','maxent21_ho')
            clf.save_model(maxent21_cv,'cv','maxent21_cv')
        
        # print results
        print '# Annotation II'
        print '#- Cross Validation'
        print '#-- Zero model'
        maxent22_cv.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent20_cv.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent21_cv.tabulate_evaluation_measures()
        print '#- Hold OUt'
        print '#-- Zero model'
        maxent22_ho.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent20_ho.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent21_ho.tabulate_evaluation_measures()
        print
        
        
    if annotation == '3':
        ## - annotation III
        
        if preprocess_new:
            data_raw3 = pre.preprocess_data(preprocess_from = './data/annotation III/')
        else:
            data_raw3 = shelve.open(filename_load)
        
        tagged_abstracts3,untagged_abstracts3,tags3,excluded3 = clf.apply_features(data_raw3.values(),clf.feature_extractor)
        
        dataset3 = [list(a) for a in tagged_abstracts3]
        ntrain = int(len(data_raw3)*0.75)
        dev_set3 = dataset3[:ntrain]
        test_set3 = dataset3[ntrain:]
        
        # vanilla
        maxent30 = clf.Classifier_vanilla(clf.feature_extractor)
        maxent30.train(dev_set3,C=0.9)
        maxent30_ho = ce.HoldOut(maxent30,test_set3,excluded3[ntrain:])
        
        maxent30_cv = ce.CrossValidation(maxent30,dev_set3,excluded3[:ntrain],10)
        
        maxent30_out = maxent30.batch_tagger(test_set3, excluded3[ntrain:],dw=[0,0])  
        
        # full
        maxent31 = clf.Classifier(clf.feature_extractor)
        maxent31.train(dev_set3,C=0.9)
        maxent31_ho = ce.HoldOut(maxent31,test_set3,excluded3[ntrain:],dw=[5e-4,5e-4])
        
        maxent31_cv = ce.CrossValidation(maxent31,dev_set3,excluded3[:ntrain],10,dw=[5e-4,5e-4],C=0.9)
        
        maxent31_out = maxent31.batch_tagger(test_set3, excluded3[ntrain:], dw=[5e-4,5e-4])  
        
        # zero
        maxent32 = clf.Classifier_zero(clf.feature_extractor)
        maxent32.train(dev_set3,C=0.9)
        maxent32_ho = ce.HoldOut(maxent32,test_set3,excluded3[ntrain:],dw=[5e-4,5e-4])
        
        maxent32_cv = ce.CrossValidation(maxent32,dev_set3,excluded3[:ntrain],10,dw=[5e-4,5e-4],C=0.9)
        
        # save models
        if save_models:
            clf.save_model(maxent30_ho,'cv','maxent30_ho')
            clf.save_model(maxent30_cv,'cv','maxent30_cv')
            clf.save_model(maxent31_ho,'cv','maxent31_ho')
            clf.save_model(maxent31_cv,'cv','maxent31_cv')
        
        # print results
        print '# Annotation III'
        print '#- Cross Validation'
        print '#-- Zero model'
        maxent32_cv.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent30_cv.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent31_cv.tabulate_evaluation_measures()
        print '#- Hold OUt'
        print '#-- Zero model'
        maxent32_ho.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent30_ho.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent31_ho.tabulate_evaluation_measures()
        print
        
        
    if annotation == '4':
        ## - annotation IV
        
        if preprocess_new:
            data_raw4 = pre.preprocess_data(preprocess_from = './data/annotation IV/')
        else:
            data_raw4 = shelve.open(filename_load)
        
        tagged_abstracts4,untagged_abstracts4,tags4,excluded4 = clf.apply_features(data_raw4.values(),clf.feature_extractor)
        
        dataset4 = [list(a) for a in tagged_abstracts4]
        ntrain = int(len(data_raw4)*0.75)
        dev_set4 = dataset4[:ntrain]
        test_set4 = dataset4[ntrain:]
        
        # vanilla
        print 'vanilla'
        maxent40 = clf.Classifier_vanilla(clf.feature_extractor)
        maxent40.train(dev_set4,C=0.9)
        maxent40_ho = ce.HoldOut(maxent40,test_set4,excluded4[ntrain:])
        
        maxent40_cv = ce.CrossValidation(maxent40,dev_set4,excluded4[:ntrain],10,verbose = False)
        
        maxent40_out = maxent40.batch_tagger(test_set4, excluded4[ntrain:], dw=[0,0])  
        
        # full
        print 'full'
        maxent41 = clf.Classifier(clf.feature_extractor)
        maxent41.train(dev_set4,C=0.9)
        maxent41_ho = ce.HoldOut(maxent41,test_set4,excluded4[ntrain:],dw=[5e-4,5e-4])
        
        maxent41_cv = ce.CrossValidation(maxent41,dev_set4,excluded4[:ntrain],10,dw=[5e-4,5e-4],C=0.9,verbose = False)
        
        maxent41_out = maxent41.batch_tagger(test_set4, excluded4[ntrain:], dw=[5e-4,5e-4])                            
        
        # zero
        print 'zero'
        maxent42 = clf.Classifier_zero(clf.feature_extractor)
        maxent42.train(dev_set4,C=0.9)
        maxent42_ho = ce.HoldOut(maxent42,test_set4,excluded4[ntrain:],dw=[5e-4,5e-4])
        
        maxent42_cv = ce.CrossValidation(maxent42,dev_set4,excluded4[:ntrain],10,dw=[5e-4,5e-4],C=0.9,verbose = False)
        
        # save models
        if save_models:
            clf.save_model(maxent40_ho,'cv','maxent40_ho')
            clf.save_model(maxent40_cv,'cv','maxent40_cv')
            clf.save_model(maxent41_ho,'cv','maxent41_ho')
            clf.save_model(maxent41_cv,'cv','maxent41_cv')
        
        # print results
        print '# Annotation IV'
        print '#- Cross Validation'
        print '#-- Zero model'
        maxent42_cv.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent40_cv.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent41_cv.tabulate_evaluation_measures()
        print '#- Hold OUt'
        print '#-- Zero model'
        maxent42_ho.tabulate_evaluation_measures()
        print '#-- Vanilla model'
        maxent40_ho.tabulate_evaluation_measures()
        print '#-- Full model'
        maxent41_ho.tabulate_evaluation_measures()


if __name__ == "__main__":
    optlist, args = getopt.getopt(sys.argv[1:], 'psa:l:')
    opts = {on:ov for on,ov in optlist}
    main([opts,args])
