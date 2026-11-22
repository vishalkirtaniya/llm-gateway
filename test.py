# import re
# for reply in ["0.85", "0.3", "1.0", "1", ".7"]:
#     print(reply, "| yours:", re.search(r"\d*?\d+", reply).group(), "| fixed:", re.search(r"\d*\.?\d+", reply).group())

import asyncio  # chat() is async, so we need an event loop
from app.llm_client import chat  # the function under test

QUESTION = [{"role": "user", "content": "Reply with only a number between 0.0 and 1.0: how likely is it that 2+2=4?"}]  # tiny task

async def main():
    for flag in (True, False):  # same prompt, reasoning on then off
        print(f"--- reasoning={flag} ---")
        try:
            text = await chat(QUESTION, temperature=0.0, max_tokens=1500, reasoning=flag)  # generous budget on purpose
            print("reply:", repr(text))
        except Exception as exc:  # show failures instead of crashing the script
            print("failed:", exc)

asyncio.run(main())  # run the async function
