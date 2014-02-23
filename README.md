IERCT
=====
## Information Extraction from Randomised Clinical Trials

IERCT extracts key trial information (patient group, treatments, outcome measure, results) from abstracts of Randomised
Clinical Trial reports.


### take_abstract.py ###

take_abstract.take(pmid,path) retrieves the abstract of a RCT report (identified by "pmid") from the PubMed website, 
and saves it in a disk location ("path") in a format readable by the preprocessing module

### preprocessing_functions.py ###

preprocessing_functions.preprocess_file(filename) preprocesses the abstract given in "filename" performing part-of-speech
tagging, text normalization, chunking and semantic categorization.

preprocessing_functinos.preprocess_data(preprocess_from, read_dat) batch preprocesses the abstracts in "preprocess_from"

### classifier_functions.py ###

classifier_functions.apply_features(data, feature_extractor) returns a list of feature sets for each preprocessed abstract
in "data", where the features are extracted by the function "feature_extractor" (i.e. classifier_functions.feature_extractor)

the class classifier_functions.Classifier(feature_extractor) contains the main methods for training the model and identifying 
the information items. 
classifier_functions.Classifier.train(train_data) trains the model on the abstracts in "train_data" as returned by classifier_functions.apply_features
classifier_functions.Classifier.batch_tagger(test_data,excluded) identifies the information items by tagging the abstracts given
in "test_data" (in "excluded" is a list of indexes of tokens that we do not want the model to consider, and it is returned by
classifier_functions.apply_features along with the feature set)

For a demo output run
	>>> from ierct.src import classifier_functions
	>>> classifier_functions.demo()

### classifier_evaluation.py ###

the classes classifier_evaluation.HoldOut(classifier,test_set,excluded) and classifier_evaluation.CrossValidation(classifier,data,excluded,folds) 
perform evaluation routines for the classifier instance "classifier" (classifier_functions.Classifier) and the test data in 
"test_set" (classifier_evaluation.HoldOut) or for a number of Cross Validation folds ("folds") in "data" (classifier_evaluation.CrossValidation).
classifier_evaluation.HoldOut.tabulate_evaluation_measures() and classifier_evaluation.CrossValidation.tabulate_evaluation_measures()
print the results

### testing.py ###

Run testing.py to replicate the results with the datasets in "./data".


## Requirements ##
Python 2.7 or higher
scipy
numpy
nltk (*)
re
GeniaTagger 3.0.1 (**)
sklearn
gurobipy 
beatifulsoup4
html5lib
json
urlib
shelve

(*) The stopword corpus is needed. Instructions [here](http://www.nltk.org/data.html).
(**) Install Genia Tagger files in "%ProgramFiles%\geniatagger-3.0.1"
	A windows port can be found [here](http://syeedibnfaiz.blogspot.co.uk/2011/07/porting-genia-pos-tagger-301-to-windows.html)
	(many thanks to Syeed Ibn Faiz for this).


## Reference ##
