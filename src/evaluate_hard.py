# src/evaluate_hard.py — harder test set, designed to find failures.

from rag import ask
import time


# Each item: question, list of strings that must ALL appear (must_contain),
# and a "mode": "all" (every string required), or "refuse" (must decline).
test_set = [
    {
        "question": "What new product offerings were announced during First Quarter 2025 and Second Quarter 2025?",
        "must_contain": ["MacBook Pro", "Mac mini", "iMac", "iPad mini", "iPhone 16e", "iPad Air", "MacBook Air", "Mac Studio"],
        "mode": "all", "category": "prose_list"
    },
    {
        "question": "What were Apple's net sales in Japan in 2025 and what was the percent change?",
        "must_contain": ["28,703", "15"],
        "mode": "all", "category": "nested_table_multipart"
    },
    {
        "question": "What is the total number of employees in Google?",
        "must_contain": [],
        "mode": "refuse", "category": "not_present"
    },
    {
        "question": "Why did Apple's iPhone net sales increase during 2025 compared to 2024?",
        "must_contain": ["Pro"],
        "mode": "all", "category": "prose_reasoning"
    },
    {
        "question": "Which product category had the highest net sales in 2025 and which had the lowest?",
        "must_contain": ["iPhone", "iPad"],
        "mode": "all", "category": "table_comparison"
    },
    {
        "question": "What was the difference between basic earnings per share and diluted earnings per share in 2025?",
        "must_contain": ["0.03"],
        "mode": "all", "category": "arithmetic"
    },
    {
        "question": "How profitable was Apple in its most recent fiscal year?",
        "must_contain": ["112,010"],
        "mode": "all", "category": "ambiguous_phrasing"
    },
]

# Phrases that signal a (correct) refusal.
REFUSE_PHRASES = ["don't have enough information", "do not have enough information",
                  "not provided", "not contain", "no information", "cannot find",
                  "isn't in", "is not in", "unable to"]

def score(item, answer):
    ans_low = answer.lower()
    if item["mode"] == "refuse":
        return any(p in ans_low for p in REFUSE_PHRASES)
    # mode == "all": every required string must appear (commas/spaces normalised)
    norm = answer.replace(",", "").replace(" ", "")
    for s in item["must_contain"]:
        s_norm = s.replace(",", "").replace(" ", "")
        if s_norm not in norm and s not in answer:
            return False
    return True

def run(n_results=8):
    passed = 0
    for item in test_set:
        answer, _ = ask(item["question"], n_results=n_results)
        ok = score(item, answer)
        passed += ok
        print(f"{'✓' if ok else '✗'}  [{item['category']}] {item['question']}")
        print(f"     got: {answer[:160].strip()}\n")
        time.sleep(15)
    print(f"Score@{n_results}: {passed/len(test_set):.0%}  ({passed}/{len(test_set)})")

if __name__ == "__main__":
    run(n_results=8)