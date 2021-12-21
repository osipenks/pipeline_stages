# This is a sample script.
from nlp import nlp
from pipeline_stages import CleanText, LowerText, CleanDigits, TokenizeSplit, StopWords, OneChar, OmittedPrepositions, Decontract, SkipPastTenses, JoinTokens, GeneralizeEnts, CleanPunctLight, GeneralizeDayOfWeek
from pipeline import Pipeline


class SomePipeline(Pipeline):
    """
    Preprocess text
    """
    def __init__(self, storage_path=None):
        super().__init__()
        self.nlp = nlp
        self.nlp.disable_pipes('lemmatizer')
        self.storage_path = storage_path

        # Some pipeline stages may parse doc and share it with other stages
        self.nlp_doc = None

        self.pipe = [
            ('clean_text', CleanText()),
            ('generalize_dow', GeneralizeDayOfWeek()),
            ('lower_text', LowerText()),
            ('decontract', Decontract()),
            ('clean_punct', CleanPunctLight()),
            ('generalize', GeneralizeEnts(nlp_model=self.nlp, ignore_ents=['CARDINAL'])),
            ('clean_digits', CleanDigits()),
            ('tokenize', TokenizeSplit()),
            ('stop_words', StopWords()),
            ('strip_one_chars', OneChar()),
            ('omitted_prepositions', OmittedPrepositions(nlp_model=self.nlp, storage_path=storage_path)),
            ('skip_past_tenses', SkipPastTenses(nlp_model=self.nlp)),
            ('tokens_to_string', JoinTokens()),
        ]


if __name__ == '__main__':

    sentence = "Hello, man. It's a beautiful, beautiful friday test"
    pipe = SomePipeline('./models')

    print(pipe.transform(sentence))
