'''
Created on 29 Dec 2014

@author: Antonio
'''

from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JNetworkError
import re
import os






class Metamap:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        self.genericObject = self._start_gateway()
        self.semmap = self._load_sem_map()
        
        
    def _start_gateway(self):
        self._gateway = JavaGateway()
        return self._gateway.entry_point.getGenericObject()
    
    
    def _submit(self):
        
        try:
            return self.genericObject.handleSubmission()
        except Py4JNetworkError as e:
            raise e
    
    
    def _load_sem_map(self):
        sm = {}
        with open(r"C:\Users\Antonio\workspace\metamap\SemanticTypes_2013AA.txt", 'rb') as foo:
            for line in foo.readlines():
                lline = line.strip().split('|')
                sm[lline[0]] = lline[2]
        
        return sm
            
    
    def process_file(self, filename):
        
        self.genericObject.setFileField("UpLoad_File", os.path.abspath(filename))
        
        out = self._submit()
        
        return out
    
    
    def process_text(self, text):
        '''
        NOT WORKING
        :param text:
        '''
        self.genericObject.setField("APIText", text)
        
        out = self._submit()
        
        return out
    
    
    def parse_result(self, result):
        pt0 = '[0-9]+\s+(.+\[.+\])'
        pt1 = '\[(.+)\]'
        pt2 = '\(.+\)'
        
        d = {}
        
        for i in re.findall(pt0, result):
            j = re.sub(pt2,'',i)
            t = re.sub(pt1,'',j).lower()
            t = re.sub('\s+$','',t)
            c = re.findall(pt1,j)[0]
            if not d.has_key(t):
                d[t] = c
        
        return d
    
    
    
        