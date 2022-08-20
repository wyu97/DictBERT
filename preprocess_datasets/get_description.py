import re
import json
import spacy
from nltk import word_tokenize

import nltk
nltk.download('punkt')

wikidata_descr = {
    'walmart': 'U.S. discount retailer based in Arkansas.',
    'wyoming': 'least populous state of the United States of America',
    'safeway': 'American supermarket chain',
    'mcdonalds': 'American fast food restaurant chain',
    'washington d.c': 'capital city of the United States',
    'espn': 'American pay television sports network',
    'windows 95': 'operating system from Microsoft'
}
my_mapping = {
    'jumping rope': 'jump rope',
    'eden': 'Eden',
    'contemplating': 'contemplate',
    'rehabilitating': 'rehabilitate',
    'catalog': 'catalogue',
    'works': 'work',
    'hoping': 'hope',
    'wetlands': 'wetland',
    'waiting': 'wait',
    'sunglass': 'sunglasses',
    'centre': 'center',
    'bath room': 'bathroom',
    'phd': 'ph.d.',
    'sunglasses': 'sunglasses',
}

patterns = [
    'plural of ', 
    "past participle of ", 
    "present participle of ",
    "third-person singular simple present indicative form of ",
    "alternative form of ",
    "alternative spelling of ",
    "alternative letter-case form of",
    "obsolete form of",
    "non-oxford british english standard spelling of",
    "obsolete spelling",
]

nlp=spacy.load('en_core_web_sm')


bad_form_of = []
def lemma_first(qc):   
    words = nlp(qc)
    qc_words = [w.text for w in words]
    lemma_word = words[0].lemma_ if words[0].lemma_ != '-PRON-' else words[0].text
    if qc_words[0] == lemma_word:
        return qc, qc_words
    else:
        qc_words[0] = lemma_word
        qc_new = ' '.join(qc_words)
        return qc_new, qc_words


def check_my_rules(meaning):
    
    for p in patterns:
        if p in meaning.lower():
            matched_word = meaning.split(p)[-1]
            # print(meaning, matched_word)
            return matched_word
    return None


def resolve_meaning(qc, wik_dict, round=0):
    
    if round > 3: return None

    qc = qc.lower()
    if qc in wikidata_descr:
        return wikidata_descr[qc]
    if qc == '':
        return None
    if qc in my_mapping:
        print('replacing {} with {}'.format(qc, my_mapping[qc]))
        qc = my_mapping[qc]
    if qc in wik_dict:
        for meaning in wik_dict[qc]:
            if 'senses' in meaning:
                for sense in meaning['senses']:      
                    if 'glosses' in sense:
                        mstr = '{}'.format(sense['glosses'][0])
                        if 'surname' in mstr.lower() or 'given name' in mstr.lower():
                            return 'a surname / given name in English.'
                        qc_new = check_my_rules(mstr)
                        if not qc_new:
                            return mstr
                        else:
                            return resolve_meaning(qc_new, wik_dict, round+1)
    return None


def remove_upprintable_chars(s):
    return ''.join(x for x in s if x.isprintable())


def skip_special_tokens(s):
    if not bool(re.search('[A-Za-z]', s)):
        return False
    elif len(s) < 3: 
        return False
    else:
        return True


def construct_dict_mapping_file(ipath, opath, wik_dict):
    with open(ipath, 'r', encoding='utf-8') as f, \
        open(opath, 'w', encoding='utf-8') as g:
        for _, line in enumerate(f.readlines()):
            vocab, count = line.strip().split('\t')
            if count == 'None': continue
            if int(count) > 3000: continue
            if not skip_special_tokens(vocab): continue
            meaning = resolve_meaning(vocab, wik_dict)

            if not meaning: continue
            
            if not vocab.isprintable():
                vocab = remove_upprintable_chars(vocab)
            if not meaning.isprintable():
                meaning = remove_upprintable_chars(meaning)

            meaning = ' '.join(word_tokenize(meaning))
            line = {'word': vocab, 'text': meaning}
            g.write(f'{json.dumps(line)}\n')


def load_dict(ipath, opath, wikidict):
    return construct_dict_mapping_file(ipath, opath, wikidict)
