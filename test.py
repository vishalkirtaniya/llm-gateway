# import re
# for reply in ["0.85", "0.3", "1.0", "1", ".7"]:
#     print(reply, "| yours:", re.search(r"\d*?\d+", reply).group(), "| fixed:", re.search(r"\d*\.?\d+", reply).group())

from app.structured import parse_structured  # the function under test
import json
from app.llm_client import LLMError

GOOD = '{"answer":"Train, by 60 minutes","confidence":0.9,"key_points":["12:15","13:15"],"needs_human_review":false}'  # a valid reply
CASES = {  # name -> raw model text
    "valid": GOOD,
    "fenced": f"```json\n{GOOD}\n```",  # code fences should be tolerated
    "chatty": f"Here is the JSON: {GOOD}",  # a preamble should be tolerated
    "confidence_7": GOOD.replace("0.9", "7"),  # out of range
    "missing_field": '{"answer":"x","confidence":0.5,"key_points":[]}',  # needs_human_review is missing
    "empty_answer": GOOD.replace("Train, by 60 minutes", ""),  # violates min_length
    "six_points": GOOD.replace('["12:15","13:15"]', '["a","b","c","d","e","f"]'),  # violates max_length
    "not_json": "I think the train arrives first.",  # no JSON at all
    "truncated": '{"answer": "Train, by',  # cut off mid-string
}

EXPECTED_FAILURES = {
    "confidence_7",
    "missing_field",
    "empty_answer",
    "six_points",
    "not_json",
    "truncated",
}

for name, raw in CASES.items():
    try:
        parse_structured(raw)

        if name in EXPECTED_FAILURES:
            print(f"FAIL  {name}: Expected an exception")
        else:
            print(f"PASS  {name}")

    except Exception as exc:
        if name in EXPECTED_FAILURES:
            print(f"PASS  {name}: Correctly rejected")
        else:
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
