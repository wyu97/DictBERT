import re
import os
import json
import string

#  extract_patterns does the following things:
#  - special patterns are extracted
#    - email addresses
#    - urls
#    - files
#  - tokenize by hyphens in words, light-hearted etc.
#
#  !! requires filter_and_cleanup_lines.py.
#  !! requires all lowercase input.
def extract_patterns(line):
    email = r'^([a-z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-z0-9\-]+\.)+))([a-z]{2,4}|[0-9]{1,3})(\]?)$'
    # ref: https://www.w3schools.com/php/php_form_url_email.asp
    url1 = r'^(?:(?:https?|ftp):\/\/|www\.)[-a-z0-9+&@#\/%?=~_|!:,.;]*[-a-z0-9+&@#\/%=~_|]$'
    # simple fooo-bar.com cases without the prefix
    url2 = r'^[^$s]{3}[^$s]*\.(?:com|net|org|edu|gov)$'
    # file: prefix len >=5, suffix given.
    file = r'^[a-z_-][a-z0-9_-]{4}[a-z0-9_-]*\.(?:pdf|mp4|mp3|doc|xls|docx|ppt|pptx|wav|wma|csv|tsv|cpp|py|bat|reg|png|jpg|mov|avi|gif|rtf|txt|bmp|mid)$'
    newline = ''
    for w in line.split():
        w = re.sub(url1, '<url>', w)
        w = re.sub(url2, '<url>', w)
        w = re.sub(email, '<email>', w)
        w = re.sub(file, '<file>', w)
        w = ' - '.join(w.split('-'))
        newline += ' ' + w
    return newline.lstrip()


# pre-process
def pre_cleanup(line):
    line = line.replace('\t', ' ')  # replace tab with spaces
    line = ' '.join(line.strip().split())  # remove redundant spaces
    line = re.sub(r'\.{4,}', '...', line)  # remove extra dots
    line = line.replace('<<', '«').replace('>>', '»')  # group << together
    line = re.sub(' (,:\.\)\]»)', r'\1', line)  # remove space before >>
    line = re.sub('(\[\(«) ', r'\1', line)  # remove space after <<
    line = line.replace(',,', ',').replace(',.', '.')  # remove redundant punctuations
    line = re.sub(r' \*([^\s])', r' \1', line)  # remove redundant asterisks
    return ' '.join(line.strip().split())  # remove redundant spaces


#  post_cleanup does the following things:
#  - remove all backslashes
#  - normalize and remove redundant spaces (including \t, etc.)
#  - tokenize by spaces, and start/end puncts, normalize each word
#    - puncts in the middle are regarded as a part of the word, for example,
#      1.23, y'all, etc.
#  - replace '...' with ' ... '
def post_cleanup(line):
    line = re.sub(r'\\', ' ', line)  # remove all backslashes
    line = re.sub(r'\s\s+', ' ', line) # remove all redundant spaces
    line = re.sub(r'\.\.\.', ' ... ', line)
    line = re.sub(r'\.\.', ' .. ', line)
    newline = ''
    for w in line.split():
        ls = w.lstrip(string.punctuation)
        rs = ls.rstrip(string.punctuation)
        lw = len(w)
        lstart = lw - len(ls)
        rstart = lstart + len(rs)
        for i in range(lstart):
            newline += ' ' + w[i]
        if rs:
            newline += ' ' + rs
        for i in range(rstart, lw):
            newline += ' ' + w[i]
    return newline.lstrip()


def preprocess_text(line):
    line = line.strip().lower()
    line = pre_cleanup(line)
    line = post_cleanup(line)   
    return line


def get_all_files(path):
    if os.path.isfile(path): return [path]
    return [f for d in os.listdir(path)
              for f in get_all_files(os.path.join(path, d))]


task_to_keys = {
    "cola": ("sentence"),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence"),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def main(folder_path):

    all_files = get_all_files(folder_path)
    for idx, file in enumerate(all_files):
        if not file.endswith('.json'): continue
        if file.endswith('prc.json'): continue

        print(f'Pre-processing the file {file}, in total {idx+1}/{len(all_files)} files.')

        new_file_path = file.replace('.json', '.prc.json')

        with open(file, 'r') as f, open(new_file_path, 'w') as g:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)

                if line.get('sentence'):
                    line['sentence'] = preprocess_text(line['sentence'])
                if line.get('sentence1'):
                    line['sentence1'] = preprocess_text(line['sentence1'])
                if line.get('sentence2'):
                    line['sentence2'] = preprocess_text(line['sentence2'])
                if line.get('question'):
                    line['question'] = preprocess_text(line['question'])
                if line.get('question1'):
                    line['question1'] = preprocess_text(line['question1'])
                if line.get('question2'):
                    line['question2'] = preprocess_text(line['question2'])
                if line.get('premise'):
                    line['premise'] = preprocess_text(line['premise'])
                if line.get('hypothesis'):
                    line['hypothesis'] = preprocess_text(line['hypothesis'])

                line = json.dumps(line)
                g.write(f'{line}\n')


if __name__ == '__main__':
    main(os.path.join(os.path.abspath(os.pardir), 'glue_datasets'))
