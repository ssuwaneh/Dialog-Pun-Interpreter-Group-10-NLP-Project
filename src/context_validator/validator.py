"""
Context Validator Module
Role: Determine whether each sense of a pun word is valid in the sentence context
"""

from sentence_transformers import SentenceTransformer, util

# Load lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def validate_context(sentence, pun_word, senses):
    """
    sentence: str
    pun_word: str
    senses: list[str] (two meanings)

    returns:
    {
        "sense_1_valid": True/False,
        "sense_2_valid": True/False,
        "similarity_scores": (score1, score2)
    }
    """

    # Replace the pun word with each sense
    sentence_1 = sentence.replace(pun_word, senses[0])
    sentence_2 = sentence.replace(pun_word, senses[1])

    # Encode sentences
    emb_original = model.encode(sentence, convert_to_tensor=True)
    emb_1 = model.encode(sentence_1, convert_to_tensor=True)
    emb_2 = model.encode(sentence_2, convert_to_tensor=True)

    # Compute similarity
    score1 = util.cos_sim(emb_original, emb_1).item()
    score2 = util.cos_sim(emb_original, emb_2).item()

    # Simple threshold rule (can adjust later)
    threshold = 0.60

    return {
        "sense_1_valid": score1 > threshold,
        "sense_2_valid": score2 > threshold,
        "similarity_scores": (score1, score2)
    }


# Example test
if __name__ == "__main__":
    sentence = "I used to be a banker but I lost interest."
    pun_word = "interest"
    senses = ["financial earnings", "personal curiosity"]

    result = validate_context(sentence, pun_word, senses)
    print(result)
