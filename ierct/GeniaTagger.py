"""
Tokenstream generator based on GENIA tagger. 
http://www-tsujii.is.s.u-tokyo.ac.jp/GENIA/home/wiki.cgi

This wrapper around GENIA manages a single GENIA instance, allowing
for batch tagging without incurring the start-up time for each 
instance being tagged.
"""
import subprocess
import re

RE_NEWLINE = re.compile(r'\n')

class GeniaTagger(object):
    TAGS = ['word', 'base', 'POStag', 'chunktag', 'NEtag']
    def __init__(self, tagger_exe=r'c:\program files\geniatagger-3.0.1\geniatagger.exe', 
                 genia_path=r'c:\program files\geniatagger-3.0.1'):
        # TODO: Handle paths not specified
        self.genia_instance = subprocess.Popen([tagger_exe], cwd=genia_path, 
                                               stdin=subprocess.PIPE, 
                                               stdout=subprocess.PIPE, 
                                               stderr=subprocess.PIPE)
        # Close stderr to avoid deadlocks due to a full buffer.
        self.genia_instance.stderr.close()
    def process(self, text):
        # Strip off newlines, as genia uses them to delimit blocks to process
        proc_text = RE_NEWLINE.sub('', text)
        # Write the text to genia's stdin, terminating with a newline
        self.genia_instance.stdin.write(proc_text + '\n')
        token_stream = []
        # Read all the lines on genia's stdout. Output ends with a blank line.
        line = self.genia_instance.stdout.readline().rstrip()
        range_start = 0
        while line:
            token = dict(zip(self.TAGS, line.split('\t')))
            # Extract the raw word
            word = token['word']
            # Clean up genia's mangling of some tokens.
            if word == '``': word = '"'
            if word == "''": word = '"'
            # Compute where this token starts and ends in the stream
            start, end = re.search(re.escape(word), proc_text[range_start:]).span()
            token['start'] = range_start + start
            token['end'] = range_start + end
            token_stream.append(token)
            range_start = range_start + end
            line = self.genia_instance.stdout.readline().rstrip()
        return token_stream 
