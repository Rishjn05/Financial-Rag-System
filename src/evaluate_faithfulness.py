# src/evaluate_faithfulness.py
# Measures answer correctness: does the full RAG pipeline produce the right value?

from rag import ask   # reuse your full pipeline from rag.py

# Each question now carries the ground-truth value the answer MUST contain.
test_set = [
    {"question": "What was Apple's net income in 2025?",          "expected": "112,010"},
    {"question": "What were Apple's total net sales in 2025?",    "expected": "416,161"},
    {"question": "What were Apple's net sales in China?",         "expected": "64,377"},
    {"question": "What was Apple's net income in 2024?",          "expected": "93,736"},
    # add several more, mixing easy and year-specific (drift-prone) ones
]

def evaluate_faithfulness(n_results=8):
    correct = 0
    for item in test_set:
        answer, _ = ask(item["question"], n_results=n_results)
        # Is the ground-truth value present in the generated answer?
        is_correct = item["expected"] in answer.replace(",", "").replace(" ", "") \
                     or item["expected"] in answer
        correct += is_correct
        mark = "✓" if is_correct else "✗"
        print(f"{mark}  {item['question']}")
        print(f"     expected: {item['expected']}  |  got: {answer[:120].strip()}\n")
    print(f"Answer correctness@{n_results}: {correct/len(test_set):.0%}  ({correct}/{len(test_set)})")

if __name__ == "__main__":
    evaluate_faithfulness(n_results=8)