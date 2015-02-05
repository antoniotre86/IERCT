'''
Created on 28 Jan 2015

@author: Antonio
'''
from ierct.baselines.rule_based import find_tag_postion


class EvaluateRuleBased:
    '''
    classdocs
    '''


    def __init__(self, extractor):
        '''
        Constructor
        '''
        self.extractor = extractor
        
    
    def run(self, abstracts):
        pass
    
    
    def eval_single(self, abstract_file, pred = None):
        
        abstract = open(abstract_file,'rb').read()
        abstract,pmid = self.extractor.preprocess(abstract, keeptags = True, nparagraphs = 999)
        tag_positions = find_tag_postion(abstract)
        
        pred = (pred if pred else self.extractor.extract(abstract_file))
        
        out = [-1,-1,-1]
        out[0] = (any([(pred['patient'][0][0] <= tp <= pred['patient'][0][1]) for tp in tag_positions['patient']])*1
                  if pred['patient'] else -1)
        pic = pred['intervention-comparison']
        if len(pic) == 2:
            out[1] = (any([(pic[0][0][0] <= tp <= pic[0][0][1]) for tp in tag_positions['intervention']])
                      or any([(pic[1][0][0] <= tp <= pic[1][0][1]) for tp in tag_positions['intervention']]))*1
            out[2] = (any([(pic[0][0][0] <= tp <= pic[0][0][1]) for tp in tag_positions['comparison']]) 
                      or any([(pic[1][0][0] <= tp <= pic[1][0][1]) for tp in tag_positions['comparison']]))*1               
        elif len(pic) == 1:
            if any([(pic[0][0][0] <= tp <= pic[0][0][1]) for tp in tag_positions['intervention']]):
                out[1] = 1
            if any([(pic[0][0][0] <= tp <= pic[0][0][1]) for tp in tag_positions['comparison']]):
                out[2] = 1
        else:
            out[1], out[2] = [-1,-1]
        
        return out
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            