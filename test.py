"""Stage 6 check: does retrieval find the right file, and how low do unrelated scores go?"""
import asyncio  # embed() and search() are async, so we need an event loop

from app.embeddings import KnowledgeBase  # the class under test

CASES = [  # (query, file that should win, or None for an unrelated question)
    ("What is the rate limit?", "acme_api.txt"),  # shares keywords with the document
    ("How many calls can I make per minute?", "acme_api.txt"),  # paraphrase: few shared words
    ("When does my login token expire?", "acme_api.txt"),  # paraphrase of the JWT sentence
    ("How do I undo a bad release?", "acme_deployment.txt"),  # paraphrase of the rollback sentence
    ("What test coverage is required?", "acme_deployment.txt"),  # keyword question
    ("How long do I have to submit receipts?", "acme_office.txt"),  # paraphrase of the expense rule
    ("What is the capital of France?", None),  # unrelated: should score low everywhere
]


async def main():
    kb = KnowledgeBase()  # an empty knowledge base, built on first search
    hits, total = 0, 0  # counters for the top-1 hit rate
    unrelated_max = 0.0  # highest score seen for an unrelated query
    for query, expected in CASES:  # run every test query
        results = await kb.search(query, top_k=3)  # top 3 matches
        print(f"\nQ: {query}")  # show the query
        for r in results:  # show each match with its score
            print(f"   {r['score']:.4f}  {r['source']:<22} {r['text'][:60]}...")
        if expected is None:  # unrelated question: remember how high it scored
            unrelated_max = max(unrelated_max, results[0]["score"])
        else:  # related question: did the right file rank first?
            total += 1
            hits += results[0]["source"] == expected
    print(f"\nTop-1 hit rate: {hits}/{total}")  # the number for your evidence log
    print(f"Highest score for an unrelated query: {unrelated_max:.4f}")  # use this to calibrate MIN_SCORE


asyncio.run(main())  # run the async function
