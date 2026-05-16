"""
Setup:
1. Install:
   pip install "huggingface_hub[inference]"

2. Create a Hugging Face token:
   https://huggingface.co/settings/tokens

3. Set token:

   Windows PowerShell:
   $env:HF_TOKEN="your_huggingface_token_here"

   macOS/Linux:
   export HF_TOKEN="your_huggingface_token_here"

4. Run:
   python question_generator_module_hf.py
"""

import os
import json
import random
from typing import Dict, List, Optional, Any

from huggingface_hub import InferenceClient


FALLBACK_DOMAIN = "Generic"
VALID_DIFFICULTIES = ["easy", "medium", "hard"]

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


DOMAIN_MAPPING = {
    "DSA": {
        "Arrays": ["Two Pointer", "Sliding Window", "Prefix Sum", "Kadane Algorithm"],
        "Linked Lists": ["Cycle Detection", "Merge Linked Lists", "Reverse Linked List"],
        "Trees": ["BST", "Traversal", "Lowest Common Ancestor"],
        "Graphs": ["BFS", "DFS", "Shortest Path"],
        "Dynamic Programming": ["Knapsack", "Memoization", "LCS"]
    },
    "Machine Learning": {
        "Supervised Learning": ["Regression", "Classification", "Decision Tree"],
        "Unsupervised Learning": ["Clustering", "Dimensionality Reduction", "K-Means"],
        "Model Evaluation": ["Confusion Matrix", "Precision Recall", "ROC AUC"],
        "Feature Engineering": ["Scaling", "Encoding", "Feature Selection"],
        "Neural Networks": ["Activation Functions", "Backpropagation", "Loss Functions"]
    },
    "DBMS": {
        "SQL Queries": ["Joins", "Nested Queries", "Aggregate Functions"],
        "Normalization": ["1NF", "2NF", "3NF"],
        "Transactions": ["ACID", "Concurrency Control", "Deadlock"],
        "Indexing": ["B-Tree Index", "Hash Index", "Primary Index"],
        "Relational Model": ["Keys", "Constraints", "Relationships"]
    },
    "Operating Systems": {
        "Process Management": ["Process States", "PCB", "Context Switching"],
        "CPU Scheduling": ["FCFS", "SJF", "Round Robin"],
        "Memory Management": ["Paging", "Segmentation", "Virtual Memory"],
        "Synchronization": ["Semaphore", "Mutex", "Deadlock"],
        "File Systems": ["File Allocation", "Directory Structure", "Disk Scheduling"]
    },
    "Computer Networks": {
        "Network Models": ["OSI Model", "TCP IP Model", "Encapsulation"],
        "Transport Layer": ["TCP", "UDP", "Congestion Control"],
        "Network Layer": ["IP Addressing", "Subnetting", "Routing"],
        "Application Layer": ["HTTP", "DNS", "SMTP"],
        "Network Security": ["Firewall", "Encryption", "SSL TLS"]
    },
    "Generic": {
        "Programming Fundamentals": ["Variables", "Functions", "Control Flow"],
        "Problem Solving": ["Algorithm Design", "Dry Run", "Complexity"],
        "Technical Communication": ["Explanation", "Documentation", "Requirement Analysis"]
    }
}


def validate_mapping(mapping: Dict[str, Dict[str, List[str]]]) -> None:
    if not mapping:
        raise ValueError("Question generation blocked: domain mapping is empty.")

    for domain, topics in mapping.items():
        if not topics:
            raise ValueError(f"Domain '{domain}' has no topics.")

        for topic, subtopics in topics.items():
            if not subtopics:
                raise ValueError(f"Topic '{topic}' in domain '{domain}' has no subtopics.")


def normalize_difficulty(difficulty: str) -> str:
    difficulty = str(difficulty).lower().strip()

    if difficulty not in VALID_DIFFICULTIES:
        return "medium"

    return difficulty


def select_domain(mapping: Dict[str, Dict[str, List[str]]], requested_domain: Optional[str]) -> str:
    if requested_domain in mapping:
        return requested_domain

    if FALLBACK_DOMAIN in mapping:
        return FALLBACK_DOMAIN

    return random.choice(list(mapping.keys()))


def select_topic_and_subtopic(
    mapping: Dict[str, Dict[str, List[str]]],
    domain: str,
    previous_questions: List[Dict[str, Any]]
) -> tuple:
    used_subtopics = {
        q.get("subtopic")
        for q in previous_questions
        if q.get("domain") == domain and q.get("subtopic")
    }

    available_pairs = []

    for topic, subtopics in mapping[domain].items():
        for subtopic in subtopics:
            if subtopic not in used_subtopics:
                available_pairs.append((topic, subtopic))

    if not available_pairs:
        for topic, subtopics in mapping[domain].items():
            for subtopic in subtopics:
                available_pairs.append((topic, subtopic))

    return random.choice(available_pairs)


def build_prompt(
    domain: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    previous_questions: List[Dict[str, Any]],
    candidate_answer: Optional[str] = None
) -> str:
    difficulty_instruction = {
        "easy": "Generate a conceptual interview question. It should test basic understanding.",
        "medium": "Generate an applied interview question. It should test practical use of the concept.",
        "hard": "Generate a problem-solving interview question. It should require deeper reasoning or implementation thinking."
    }

    previous_question_texts = [
        q.get("question", "")
        for q in previous_questions
        if q.get("question")
    ]

    prompt = f"""
Generate exactly one interview-style question.

Domain: {domain}
Topic: {topic}
Subtopic: {subtopic}
Difficulty: {difficulty}

Difficulty Rule:
{difficulty_instruction[difficulty]}

Question Rules:
1. The question must strictly belong to the given domain, topic, and subtopic.
2. The question must not be vague or overly open-ended.
3. The question must not repeat any previous question.
4. The question must be suitable for an interview.
5. Do not include the answer.
6. Return only valid JSON.
7. Do not add markdown, explanation, comments, or extra text.

Previous Questions:
{json.dumps(previous_question_texts, indent=2)}
"""

    if candidate_answer:
        prompt += f"""

Context-Aware Rule:
The candidate previously answered:
"{candidate_answer}"

Generate a follow-up interview question that checks deeper understanding without repeating the previous question.
"""

    prompt += f"""

Return JSON exactly in this format:
{{
  "question": "...",
  "difficulty": "{difficulty}",
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "domain": "{domain}"
}}
"""

    return prompt.strip()


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    # Extract first JSON object if model adds extra text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def call_huggingface_llm(prompt: str, model: str = DEFAULT_HF_MODEL) -> Dict[str, Any]:
    """
    Calls a Hugging Face-hosted LLM using InferenceClient chat completions.
    """

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Create a Hugging Face token and set it before running."
        )

    client = InferenceClient(
        model=model,
        token=hf_token
    )

    completion = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an expert technical interviewer. Always return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=250,
        temperature=0.7
    )

    raw_text = completion.choices[0].message.content
    cleaned_text = clean_json_text(raw_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        raise ValueError(f"Hugging Face model did not return valid JSON. Raw response was: {raw_text}")


def is_repeated_question(question: str, previous_questions: List[Dict[str, Any]]) -> bool:
    old_questions = {
        q.get("question", "").strip().lower()
        for q in previous_questions
        if q.get("question")
    }

    return question.strip().lower() in old_questions


def generate_question(
    input_payload: Dict[str, Any],
    mapping: Optional[Dict[str, Dict[str, List[str]]]] = None,
    model: str = DEFAULT_HF_MODEL
) -> Dict[str, Any]:
    
    mapping = mapping or DOMAIN_MAPPING
    validate_mapping(mapping)

    requested_domain = input_payload.get("domain")
    difficulty = normalize_difficulty(input_payload.get("difficulty", "medium"))
    previous_questions = input_payload.get("previous_questions", [])
    candidate_answer = input_payload.get("candidate_answer")

    domain = select_domain(mapping, requested_domain)
    topic, subtopic = select_topic_and_subtopic(mapping, domain, previous_questions)

    prompt = build_prompt(
        domain=domain,
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        previous_questions=previous_questions,
        candidate_answer=candidate_answer
    )

    result = call_huggingface_llm(prompt, model=model)

    retry_count = 0
    while is_repeated_question(result.get("question", ""), previous_questions) and retry_count < 3:
        topic, subtopic = select_topic_and_subtopic(mapping, domain, previous_questions)

        prompt = build_prompt(
            domain=domain,
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            previous_questions=previous_questions,
            candidate_answer=candidate_answer
        )

        result = call_huggingface_llm(prompt, model=model)
        retry_count += 1

    return {
        "question": result.get("question"),
        "difficulty": result.get("difficulty", difficulty),
        "topic": result.get("topic", topic),
        "subtopic": result.get("subtopic", subtopic),
        "domain": result.get("domain", domain)
    }


if __name__ == "__main__":
    sample_input = {
        "domain": "Machine Learning",
        "difficulty": "medium",
        "previous_questions": []
    }

    output = generate_question(sample_input)
    print(json.dumps(output, indent=2))
