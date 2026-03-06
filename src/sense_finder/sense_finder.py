# sense_finder/sense_finder.py
from nltk.corpus import wordnet

# If WordNet is not downloaded yet, uncomment this:
# import nltk
# nltk.download('wordnet')

def find_senses(word):
    """
    Given a word, return two possible meanings (senses) for the pun.
    
    Returns:
    {
        "pun_word": str,
        "sense_a": str,
        "sense_b": str
    }
    """
    synsets = wordnet.synsets(word)

    if len(synsets) >= 2:
        sense_a = synsets[0].definition()
        sense_b = synsets[1].definition()
    elif len(synsets) == 1:
        sense_a = synsets[0].definition()
        sense_b = synsets[0].definition()  # fallback if only one sense exists
    else:
        # No synsets found
        sense_a = ""
        sense_b = ""

    return {
        "pun_word": word,
        "sense_a": sense_a,
        "sense_b": sense_b
    }
