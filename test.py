"""Stage 7 check: runs offline. It ends with 'All checks passed' when everything behaves."""
from app.context import (  # the pieces under test
    ContextBudget, ConversationManager, SessionStore, count_tokens, fit_to_budget, reorder_lost_in_middle,
)


def words(n: int) -> str:  # a string with exactly n words
    return " ".join(["w"] * n)


# 1. token counting: 10 words -> 13 tokens, empty -> 0
assert count_tokens(words(10)) == 13
assert count_tokens("") == 0
print("PASS token counting")

# 2. lost-in-the-middle: best first, second best last, weakest in the middle
order = reorder_lost_in_middle(["A", "B", "C", "D", "E"], [0.95, 0.60, 0.20, 0.80, 0.50])
print("reordered:", order)  # expected A (best) ... D (second best), with C (weakest) in the middle
assert order == ["A", "B", "C", "E", "D"]
print("PASS reorder")

# 3. fit_to_budget: drops the weakest whole chunk, keeps the original order
kept, kept_scores = fit_to_budget([words(10), words(10), words(10)], [0.9, 0.5, 0.7], max_tokens=30)  # each chunk costs 13 tokens
print("kept scores:", kept_scores)
assert kept_scores == [0.9, 0.7]  # the 0.5 chunk was dropped, and the order is preserved
print("PASS fit_to_budget")

# 4. budget: component caps and priority order
budget = ContextBudget(max_tokens=1000, generation_reserve=200)  # 800 tokens available for input
sys_text, sys_tokens = budget.allocate("system_prompt", words(100), max_tokens=50)  # over its cap, so it gets trimmed
q_text, q_tokens = budget.allocate("user_query", words(10))  # fits easily
ctx_text, ctx_tokens = budget.allocate("retrieved_context", words(1000), max_tokens=5000)  # trimmed to what is left
hist_text, hist_tokens = budget.allocate("history", "hello there")  # starved: nothing is left
print(budget.report())
assert sys_tokens <= 50  # component cap respected
assert budget.used() <= budget.available  # window limit respected
assert hist_text == ""  # a late component gets nothing when earlier ones used the space
print("PASS budget")

# 5. history compression: stays within budget, keeps the newest turn, creates summaries
manager = ConversationManager(max_history_tokens=200, keep_recent=4)
for i in range(20):  # 20 turns of about 26 tokens each
    manager.add_turn("user" if i % 2 == 0 else "assistant", f"message {i} " + words(18))
print(f"turns kept: {len(manager.turns)}, summaries: {len(manager.summaries)}, tokens: {manager.token_count()}")
assert manager.token_count() <= 200  # compressed into the budget
assert len(manager.turns) >= 4  # the recent turns survive
assert manager.turns[-1]["content"].startswith("message 19")  # the newest turn is intact
assert len(manager.summaries) >= 1  # older turns became summaries
print("PASS history compression")

# 6. sessions: the least recently used one is evicted
store = SessionStore(max_sessions=2)
store.get("a")
store.get("b")
store.get("a")  # touch "a" so "b" becomes the oldest
store.get("c")  # a third session pushes out the oldest ("b")
assert "b" not in store and "a" in store and "c" in store and len(store) == 2
print("PASS session eviction")

print("\nAll checks passed")
