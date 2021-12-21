import os
import re
import copy
import gzip
import pickle
import string
from typing import List, Iterable
from pipeline import PipelineStage
from contractions import CONTRACTIONS


UNK_LABEL = 'UNK'


class CleanText(PipelineStage):

    def __init__(self):
        bad_regex_list = [
            # web links
            r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+(:[0-9]+)?|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)',
        ]
        self.bad_regex = re.compile('|'.join(bad_regex_list))

    def clean_txt(self, origin_txt):
        txt = origin_txt.strip()
        txt = re.sub(pattern=self.bad_regex, repl=" ", string=txt)
        txt = txt.strip()
        return txt

    def transform(self, X: str) -> str:
        return self.clean_txt(str(X))


class LowerText(PipelineStage):

    def transform(self, X: str) -> str:
        return X.lower()


class CleanPunct(PipelineStage):

    def transform(self, X: str) -> str:
        return X.translate(str.maketrans('', '', string.punctuation))


class CleanPunctLight(PipelineStage):

    def transform(self, X: str) -> str:
        return X.translate(str.maketrans('', '', '!"#$%&()*+/:;<=>@[\\]^_`{|}~'))


class CleanDigits(PipelineStage):

    def transform(self, X: str) -> str:
        return X.translate(str.maketrans('', '', string.digits))


class GeneralizeDayNumber(PipelineStage):
    """
    Replace day number by label
    """

    def __init__(self):
        self._label = 'DAY_NUMBER'

        day_regex_list = [
            '([1-31]|[0-2][0-9]|[3][0,1])',
        ]
        self.day_regex = re.compile('|'.join(day_regex_list))

    def transform(self, X: str) -> str:
        tokens = []
        for token in X.split():
            tokens.append(self._label if re.fullmatch(self.day_regex, token) else token)
        return ' '.join(tokens)


class GeneralizeYear(PipelineStage):
    """
    Replace year (determined in reasonable way) by label
    """

    def __init__(self):
        self._label = 'YEAR'

        year_regex_list = [
            '([1-2][0-9]{3})',
        ]
        self.year_regex = re.compile('|'.join(year_regex_list))

    def transform(self, X: str) -> str:
        tokens = []
        for token in X.split():
            tokens.append(self._label if re.fullmatch(self.year_regex, token) else token)
        return ' '.join(tokens)


class GeneralizeTimeOfTheDay(PipelineStage):
    """
    Replace time of the day by label
    """

    def __init__(self):
        self._label = 'TIME_OF_THE_DAY'
        self.synonyms = ('afternoon', 'arvo', 'bedtime'
                                              'day', 'daylight', 'daytime',
                         'eve', 'evening', 'mealtime',
                         'morning', 'night', 'nighttime',
                         'tonight', 'lunchtime'
                         )

    def transform(self, X: str) -> str:
        return ' '.join([self._label
                         if token.translate(str.maketrans('', '', string.punctuation)) in self.synonyms else token
                         for token in X.split()
                         if token.strip()])


class GeneralizeDayOfWeek(PipelineStage):
    """
    Replace day of week by label
    """

    def __init__(self):
        self._label = 'DAY_OF_WEEK'
        self.synonyms = ('monday', 'mon', 'mo',
                         'tuesday', 'tue', 'tu',
                         'wednesday', 'wed', 'we',
                         'thursday', 'thu', 'th',
                         'friday', 'fri', 'fr',
                         'saturday', 'sat', 'sa',
                         'sunday', 'sun', 'su',
                         )

    def transform(self, X: str) -> str:
        return ' '.join([self._label
                         if token.translate(str.maketrans('', '', string.punctuation)) in self.synonyms else token
                         for token in X.split()
                         if token.strip()])


class GeneralizeMonth(PipelineStage):
    """
    Replace month by label
    """

    def __init__(self):
        self._label = 'MONTH'
        self.synonyms = ('january', 'jan',
                         'february', 'feb',
                         'march', 'mar',
                         'april', 'apr',
                         'may',
                         'june', 'jun',
                         'july', 'jul',
                         'august', 'aug',
                         'september', 'sep', 'sept',
                         'october', 'oct',
                         'november', 'nov',
                         'december', 'dec',
                         )

    def transform(self, X: str) -> str:
        return ' '.join([self._label
                         if token.translate(str.maketrans('', '', string.punctuation)) in self.synonyms else token
                         for token in X.split()
                         if token.strip()])


class GeneralizeEnts(PipelineStage):
    """
    Replace words by recognized named entities
    PERSON, NORP, FACILITY, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LANGUAGE, LAW, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
    """

    def __init__(self, nlp_model=None, ignore_ents=None):
        self.nlp = nlp_model
        self.ignore_ents = ignore_ents if ignore_ents is not None else []

    def label(self, label_):
        return label_

    def transform(self, X: str) -> str:
        doc = self.nlp(X)
        for ent in reversed(doc.ents):
            label = self.label(ent.label_)
            if label in self.ignore_ents:
                X = X[:ent.start_char] + ent.text + X[ent.end_char:]
            else:
                X = X[:ent.start_char] + label + X[ent.end_char:]
        return X


class Decontract(PipelineStage):
    """
    Replace contractions with their full versions
    """

    def transform(self, X: str) -> str:
        X = X.replace("’", "'")
        for token in X.split():
            decontraction = CONTRACTIONS.get(token)
            if decontraction:
                X = X.replace(token, decontraction)

        return X


class SkipPastTenses(PipelineStage):
    """
    Detect past tense in sentence, skip it, if detected.
    We do dependency parsing and looking at the sentence root.
    If it is VBD (verb, past tense), we suppose it is past tense.
    If not, we iterate dependency children and check if there is an auxiliary verb having VBD form
    """

    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def detect_past_tense(self, sentence):
        sents = list(self.nlp(sentence).sents)

        if not sents:
            return False

        sent_token = sents[-1]  # analyze last sentence only
        if sent_token.root.dep_ == 'ROOT':
            if (sent_token.root.tag_ == "VBD" or
                    (any(w.dep_ == "aux" and w.tag_ == "VBD" for w in sent_token.root.children) and
                     len([x for x in sent_token.root.children]) > 1)):
                # It may be a complex sentence, let's check the tenses of constitutes
                is_past_tense = True
                for child in sent_token.root.children:
                    if child.pos_ == 'VERB':
                        is_past_tense = not child.tag_ == 'VBP'
                return is_past_tense

        return False

    def transform(self, X: str) -> str:
        sentence = X if type(X) == str else ' '.join(X)
        return '' if self.detect_past_tense(sentence) else X


class TokenizeSplit(PipelineStage):

    def transform(self, X: str) -> List[str]:
        doc = X.split()
        return [token for token in doc
                if token.strip()]


class CountOfTokens(PipelineStage):

    def __init__(self, n):
        self.n = n

    def transform(self, X: str) -> List[str]:
        doc = X.split()
        return '' if len(doc) <= self.n else X


class JoinTokens(PipelineStage):

    def transform(self, X: List[str]) -> str:
        return ' '.join([token for token in X]).strip()


class Detokenize(PipelineStage):

    def transform(self, X: List[str]) -> str:
        return ' '.join(X)


class StopWords(PipelineStage):

    def __init__(self):
        self.stop_words = (
            'op',
        )

    def transform(self, X: List[str]) -> List[str]:
        return [token if token not in self.stop_words else UNK_LABEL for token in X]


class OneChar(PipelineStage):

    def transform(self, X: List[str]) -> List[str]:
        return [token if len(token) > 1 or token in ('i', 'a') else UNK_LABEL for token in X]


class OmittedPrepositions(PipelineStage):
    """
    On/At/In
    There are cases when prepositions must be omitted and when omission is optional
    Prepositions of time are omitted
    1. before the words: last, next, this, that, some, every (We met last month. We meet every day.)
    2. "at", "on", "in" are optional in some cases (but only these three prepositions):
        - when the phrase refers to times at more than one remove from the present:
            (on) the day before yesterday, (in) the January before last.

        - in postmodified phrases containing "the" the preposition is optional in American English:
            We met the day of the conference., We met the spring of 1983.
            However: We met in the spring. (can not be omitted because there is nothing after the prepositional phrase.)

        - in phrases which identify a time before or after a given time in the past or future:
            (in) the previous spring (the spring before the time in question) (at/on) the following weekend, (on)the next day.

    Our strategy:
        - define if token is a verb
        - is it used ever with prepositions of time? Look at the collected stats for that case
        - add preposition right after the token
    """

    def __init__(self, nlp_model=None, storage_path: str = None):
        """
        preps: list of possible prepositions {'on': 0, 'at': 1, 'in': 2}
        preps_stat: occurrences of preps with tokens {'token1': (12, 2, 4), ... }

        """
        self.name = 'prepositions_of_time'
        self.file_name = f'{self.name}.mdl'
        self.storage_path = storage_path

        self.nlp = copy.deepcopy(nlp_model)
        self.nlp.disable_pipes('ner', 'lemmatizer')

        self.prepositions = ('on', 'at', 'in')

        regex_list = ['(.* +on(( +)|$).*)', '(.* +at(( +)|$).*)', '(.* +in(( +)|$).*)']
        self.preposition_regex = re.compile('|'.join(regex_list))

        # Stat dictionary
        # { 'token': (n1, n2, n3, n4), ...}
        # n1 total occurrences of token
        # n2 occurrences with on
        # n3 occurrences with at
        # n4 occurrences with in
        self.stat = {}

        """ 
        We believe preposition can be omitted, if occurrence of verb without preposition 
        is more often than occurrence with it, but no more than self.threshold times (i.e. 5-27 times, not 30-50)
        P.S. have to be tuned for every training dataset  
        """
        self.threshold = 27

        self.load(self.storage_path)

    def save(self, storage_path: str = None):

        if storage_path:
            self.storage_path = storage_path

        if self.storage_path:
            file = os.path.join(self.storage_path, self.file_name)
            with gzip.open(file, 'wb') as f:
                tmp = {
                    'stat': self.stat,
                    'threshold': self.threshold,
                }
                pickle.dump(tmp, f, pickle.HIGHEST_PROTOCOL)

    def load(self, storage_path: str = None):

        if storage_path:
            self.storage_path = storage_path

        if self.storage_path:
            file = os.path.join(self.storage_path, self.file_name)
            with gzip.open(file, 'rb') as f:
                tmp = pickle.load(f)
                self.stat = tmp['stat']
                # self.threshold = tmp['threshold']

    def fit(self, X: Iterable[str], Y=None):
        self.stat = {}
        for txt in X:
            if not re.fullmatch(self.preposition_regex, str(txt)):
                continue

            tokens = txt.split()
            for i, t in enumerate(tokens[1:]):
                if t not in self.prepositions:
                    continue
                prev_token = tokens[i]
                prep_indx = self.prepositions.index(t) + 1
                if prev_token in self.stat.keys():
                    val = list(self.stat[prev_token])
                    val[prep_indx] += 1
                    self.stat.update({prev_token: tuple(val)})
                else:
                    val = [0, 0, 0, 0]
                    val[prep_indx] = 1
                    self.stat[prev_token] = tuple(val)

        keys = self.stat.keys()
        for txt in X:
            for token in txt.split():
                if token in keys:
                    val = list(self.stat[token])
                    val[0] += 1
                    self.stat[token] = tuple(val)

        return self

    def transform(self, X: List[str]) -> List[str]:

        if not X:
            return X

        doc = self.nlp(X[-1])
        if doc:
            last_tok = doc[-1]
            if last_tok.pos_ == 'VERB':
                cnts = self.stat.get(last_tok.text, None)
                if cnts:
                    max_index = cnts[1:].index(max(cnts[1:])) + 1
                    cnt_with_prep = cnts[max_index]
                    cnt_total = cnts[0]
                    factor = cnt_total / cnt_with_prep if cnt_with_prep else self.threshold
                    if factor <= self.threshold:
                        X.append(self.prepositions[max_index - 1])
        return X
