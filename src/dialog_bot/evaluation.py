# evaluation for the dialog bot
# checks if the bot's responses actually reference the right pun info

from dialog_bot import analyze_pun, chat

# (pun sentence, question, keywords we expect to see in the response)
#TBD with puns we choose for demo
TEST_CASES = []


def keyword_hit_rate(response, expected):
    response_lower = response.lower()
    hits = sum(1 for kw in expected if kw in response_lower)
    return hits / len(expected)


def run_evaluation():
    results = []
    for sentence, question, keywords in TEST_CASES:
        analysis = analyze_pun(sentence)
        response = chat(sentence, question, [], analysis)
        score = keyword_hit_rate(response, keywords)
        results.append({"sentence": sentence, "question": question,
                        "response": response[:200], "keyword_score": score})

        print(f"Q: {question}")
        print(f"A: {response[:200]}")
        print(f"Keyword coverage: {score:.0%}")
        print("-" * 60)

    avg = sum(r["keyword_score"] for r in results) / len(results)
    print(f"\nOverall keyword coverage: {avg:.0%}")
    return results


if __name__ == "__main__":
    run_evaluation()
