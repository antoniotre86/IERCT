'''
Created on 15 Feb 2014

@author: Antonio
'''

import bs4, re
from urllib import urlopen


def take(pmid,path=r'.\data\retrieved'):
    '''
    retrieves the abstract with id pmid from PUBMED and saves it in path
    in a standard format
    '''
    
    url = 'http://www.ncbi.nlm.nih.gov/pubmed/'+pmid+'?report=xml&format=text'
    xmlpage = urlopen(url).read()
    xmlpage = xmlpage.replace('&lt;','<')
    xmlpage = xmlpage.replace('&gt;','>')
    
    soup = bs4.BeautifulSoup(xmlpage,"html5lib")
    if soup.abstract:
        abstract = soup.abstract.encode('utf-8')
    else:
        print '*',
        soup = bs4.BeautifulSoup(xmlpage,"xml")
        abstract = soup.Abstract.encode('utf-8')
        abstract = re.sub(r'\<Abstract','<abstract',abstract)
        abstract = re.sub(r'\<\/Abstract','</abstract',abstract)
        abstract = re.sub(r'bstractText','bstracttext',abstract)
        abstract = re.sub(r'Label\=','label=',abstract)
        abstract = re.sub(r'NlmCategory\=','nlmcategory=',abstract) 
    
    abstract = re.sub(r'\n\s+','\n',abstract)    
    output = str(soup.pmid)+'\n'+abstract
    out_file = open(path+'\\'+pmid+'.xml','w')
    out_file.write(output)
    out_file.close()


def take_title(pmid):
    '''
    returns the title of the document registered with pmid
    :param pmid:str
    '''
    
    url = 'http://www.ncbi.nlm.nih.gov/pubmed/'+pmid+'?report=xml&format=text'
    xmlpage = urlopen(url).read()
    xmlpage = xmlpage.replace('&lt;','<')
    xmlpage = xmlpage.replace('&gt;','>')
    soup = bs4.BeautifulSoup(xmlpage,"html5lib")
    if soup.articletitle:
        return soup.articletitle.text.encode('utf-8')
    else:
        return 'no title'
    