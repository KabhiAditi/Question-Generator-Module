# GPT/LLM Question Generator Module

## Overview

The GPT/LLM Question Generator Module is a domain-aware interview question generation system designed for AI-driven interview and evaluation platforms.

The module dynamically generates technical interview questions using a Large Language Model (LLM) backend integrated through Hugging Face Inference Providers.

The system supports:
- Domain-based generation
- Difficulty-aware generation
- Topic and subtopic variation
- Duplicate prevention
- Structured JSON outputs
- API-ready architecture

---

#  Objectives

The objective of this module is to:
- Generate structured interview-style technical questions
- Ensure domain relevance
- Control difficulty level
- Avoid repeated questions
- Support integration with APIs and evaluation systems

---

# Features

## Domain-Based Generation

Supports domains such as:
- DSA
- Machine Learning
- DBMS
- Operating Systems
- Computer Networks

---

## Difficulty-Aware Question Generation

| Difficulty | Question Type |
|---|---|
| Easy | Conceptual |
| Medium | Applied |
| Hard | Problem-Solving |

---

## Topic & Subtopic Variation

The system dynamically selects:

```text
Domain → Topic → Subtopic
```

to ensure varied question generation.

---

## Duplicate Prevention

The module:
- tracks previous questions
- avoids repeated subtopics
- retries generation if duplicate detected

---

## API-Compatible JSON Output

Example Output:

```json
{
  "question": "How would you apply classification to solve a real-world problem?",
  "difficulty": "medium",
  "topic": "Supervised Learning",
  "subtopic": "Classification",
  "domain": "Machine Learning"
}
```

---

# Technologies Used

- Python
- Hugging Face Inference API
- huggingface_hub
- JSON
- Flask
- Prompt Engineering

---

# Project Structure

```text
project/
│
├── question_generator_module.py
├── test_question_generator.py
├── flask_api.py
├── README.md
```

---

# Installation

## Step 1 — Install Dependencies

```bash
pip install "huggingface_hub[inference]"
```

---

## Step 2 — Create Hugging Face Token

Generate token from:

https://huggingface.co/settings/tokens

Enable:

```text
Inference → Make calls to Inference Providers
```

---

## Step 3 — Set Environment Variable

### Windows PowerShell

```powershell
$env:HF_TOKEN="your_token_here"
```

### macOS/Linux

```bash
export HF_TOKEN="your_token_here"
```

---

# Running the Module

## Run Main Generator

```bash
python question_generator_module_hf.py
```

---

## Run Tests

```bash
python test_hf_question_generator.py
```

---

## Run Flask API

```bash
python flask_api_hf.py
```

---

# Workflow

```text
Input Payload
    ↓
Validate Domain Mapping
    ↓
Normalize Difficulty
    ↓
Select Domain
    ↓
Select Topic & Subtopic
    ↓
Build Prompt
    ↓
Call Hugging Face LLM
    ↓
Validate JSON Response
    ↓
Check Duplicate Questions
    ↓
Return Final JSON Output
```

---

# Sample Input

```json
{
  "domain": "Machine Learning",
  "difficulty": "medium",
  "previous_questions": []
}
```

---

# Sample Output

```json
{
  "question": "How would you apply classification to solve a real-world problem?",
  "difficulty": "medium",
  "topic": "Supervised Learning",
  "subtopic": "Classification",
  "domain": "Machine Learning"
}
```

---

# Future Enhancements

- Semantic similarity checking
- User-adaptive difficulty
- Database integration
- Multi-model support
- Advanced evaluation engine
- Voice-based interview support

---

# References

- https://huggingface.co/
- https://huggingface.co/docs
- https://flask.palletsprojects.com/
- https://docs.python.org/3/

