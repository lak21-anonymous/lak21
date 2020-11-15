import nltk
import numpy as np
import csv
import pkg_resources
import copy

# lemmatizer packages
from utilities import *
from constants import *
from symspellpy import SymSpell, Verbosity
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
nltk.download('wordnet')

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
twd = TreebankWordDetokenizer()
wnl = WordNetLemmatizer()

ENGLISH_ASCII = list(range(128)) + list(range(8210, 8224)) + [160, 180]
CONTRACTIONS = "contractions.csv"
EXTRA_WORDS = "extra_terms.csv"
DICTIONARY_PATH = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
TAGS = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

reader = csv.reader(open(CONTRACTIONS))
contractions = {}
for row in reader:
    contractions[row[0]] = row[1]

def write_counts_file(csv_name, count = 100000000):
    reader = csv.reader(open(csv_name))
    output_file_name = csv_name.split('.')[0] + ".txt"
    with open(output_file_name, "w") as text_file:
        for row in reader:
            line = row[0] + " " + str(count) + "\n"
            text_file.write(line)
    return output_file_name

WORD_SOURCES = [DICTIONARY_PATH, write_counts_file(CONTRACTIONS), write_counts_file(EXTRA_WORDS)]
for word_source in WORD_SOURCES:
    sym_spell.load_dictionary(word_source, term_index=0, count_index=1)
    

english_vocab = set(w.lower() for w in nltk.corpus.words.words()) 
#this is where the bug is, i'm going to leave it unchanged for now
#it should be 
reader = csv.reader(open(EXTRA_WORDS))
#it was
#reader = csv.reader(EXTRA_WORDS)
extra_words = tuple(row[0] for row in reader)
english_vocab.update(extra_words)


def preprocess_with_counts(file):
    data = get_problem_data(file)
    
    processed_results = {}
    
    for problem, results in data.items():
        data[problem] = results[results['Question'].isin(INCLUDED_QUESTIONS)]
    print(answer_counts(data))
    for problem, results in data.items():
        data[problem] = results[~results['Answer'].isin(EXCLUDED_ANSWERS)]
    print(answer_counts(data))
    
    for problem, results in data.items():
        results['english'] = results['Answer'].apply(is_english)
        results = results[results['english']]
        results.drop('english', inplace=True, axis=1)
        data[problem] = results
    print(answer_counts(data))
    for problem, results in data.items():
        results['Original'] = results['Answer']
        results['Answer'] = results['Answer'].apply(normalize)
        
        results['nonsense'] = results['Answer'].apply(is_nonsense)
        nonsense = results[results['nonsense']]
        print(problem)
        print(nonsense.shape[0])
        print(nonsense[nonsense['Original'].str.split().str.len()< 2].shape[0])
        results = results[~results['nonsense']]
        results.drop('nonsense', inplace=True, axis=1)
        results['Manual Tag'] = 'no tag'
        data[problem] = results
    print(answer_counts(data))
        

def preprocess(file):
    data = get_problem_data(file)
    data = filter_questions_and_answers(data)
    
    processed_results = {}
    for problem, results in data.items():
        results['Unique ID'] = results['username'] + results['Answer ID']
        assert results[results['Unique ID'].duplicated()].empty
        
        processed = results[results['Question'].isin(INCLUDED_QUESTIONS)]
        processed = processed[~processed['Answer'].isin(EXCLUDED_ANSWERS)]
        
        processed['english'] = processed['Answer'].apply(is_english)
        processed = processed[processed['english']]
        processed.drop('english', inplace=True, axis=1)
        
        processed['Original'] = processed['Answer']
        processed['Answer'] = processed['Answer'].apply(normalize)
        
        processed['nonsense'] = processed['Answer'].apply(is_nonsense)
        processed = processed[~processed['nonsense']]
        processed.drop('nonsense', inplace=True, axis=1)
        processed['Manual Tag'] = 'no tag'
        
        
        
        processed_results[problem] = processed
    return processed_results

def preprocess_results(results):
    results = results[(results['Question'].isin(INCLUDED_QUESTIONS)) & (~results['Answer'].isin(EXCLUDED_ANSWERS))]
    results['Answer'] = results['Answer'].str.strip()
        
    processed = results[results['Question'].isin(INCLUDED_QUESTIONS)]
    processed = processed[~processed['Answer'].isin(EXCLUDED_ANSWERS)]
        
    processed['english'] = processed['Answer'].apply(is_english)
    processed = processed[processed['english']]

    processed['Original'] = processed['Answer']
    processed['Answer'] = processed['Answer'].apply(normalize)

    processed['nonsense'] = processed['Answer'].apply(is_nonsense)
    processed = processed[~processed['nonsense']]
    processed['Manual Tag'] = 'no tag'
    
    return processed

def filter_results(results):
    results = results[(results['Question'].isin(INCLUDED_QUESTIONS)) & (~results['Answer'].isin(EXCLUDED_ANSWERS))]
    results['Answer'] = results['Answer'].str.strip()
    


def filter_questions_and_answers(data):
    filtered_results = {}
    for problem, results in data.items():
        results = copy.deepcopy(results)
        filtered = results[results['Question'].isin(INCLUDED_QUESTIONS)]
        filtered = filtered[~filtered['Answer'].isin(EXCLUDED_ANSWERS)]
        filtered['Answer'] = filtered['Answer'].str.strip()
        filtered_results[problem] = filtered
    return filtered_results


def is_english(text):
    return all(ord(char) in ENGLISH_ASCII for char in text )


def keep_english_results(data):
    cleaned_results = {}
    for problem, results in data.items():
        cleaned_results[problem] = results[is_english(cleaned['Answer'])]
    return cleaned_results
       
    
def spellcheck(sentence):
    suggestions = sym_spell.lookup_compound(sentence, max_edit_distance=2)
    return suggestions[0]._term if suggestions else sentence


def spellcheck_results(data):
    cleaned_results = {}  
    for problem, results in data.items():
        cleaned = copy.deepcopy(results)
        cleaned['Answer'] = cleaned['Answer'].apply(spellcheck)
        cleaned_results[problem] = cleaned
    return cleaned_results


def expand_contraction(word):
    return contractions[word] if word in contractions else word


def expand_sentence_contractions(sentence):
    return twd.detokenize([expand_contraction(word) for word in sentence.split(' ')])


def expand_contraction_results(data):
    cleaned_results = {}  
    for problem, results in data.items():
        cleaned = copy.deepcopy(results)
        cleaned['Answer'] = cleaned['Answer'].apply(expand_sentence_contractions)
        cleaned_results[problem] = cleaned
    return cleaned_results

def lemmatize_sentence(text):
    def get_wordnet_tag(tag):       
        return TAGS[tag[0]] if tag[0] in TAGS else ''
    
    def lemmatize(tagged_word):
        if not tagged_word[1]: 
            return tagged_word[0]
        return wnl.lemmatize(tagged_word[0],tagged_word[1])
    
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    tagged_tokens = [(token[0], get_wordnet_tag(token[1])) for token in tagged_tokens]
    
    return twd.detokenize([lemmatize(token) for token in tagged_tokens])


def lemmatize_data(data):
    cleaned_results = {}  
    for problem, results in data.items():
        cleaned = copy.deepcopy(results)
        cleaned['Answer'] = cleaned['Answer'].apply(lemmatize_sentence)
        cleaned_results[problem] = cleaned
    return cleaned_results


def is_nonsense(text):
    if len(text) < 2: return True
    
    count = 0
    tokens = text.split()
    for word in tokens:
        if word not in english_vocab:
            count += 1

    return count*2 > len(tokens)


def remove_nonsense(data):
    cleaned_results = {}
    for problem, results in data.items():
        cleaned_results[problem] = results[~is_nonsense(cleaned['Answer'])]
    return cleaned_results


def normalize(text):
    text = spellcheck(text)
    text = expand_sentence_contractions(text)
    text = lemmatize_sentence(text)
    return text


