import json
from question_generator_module import generate_question


test_inputs = [
    {
        "domain": "Machine Learning",
        "difficulty": "medium",
        "previous_questions": []
    },
    {
        "domain": "DSA",
        "difficulty": "hard",
        "previous_questions": []
    },
    {
        "domain": "DBMS",
        "difficulty": "easy",
        "previous_questions": []
    }
]


previous_questions = []

for payload in test_inputs:
    payload["previous_questions"] = previous_questions

    result = generate_question(payload)
    previous_questions.append(result)

    print("\nGenerated Hugging Face LLM Question:")
    print(json.dumps(result, indent=2))
