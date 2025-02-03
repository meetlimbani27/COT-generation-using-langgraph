# COT Generation using LangGraph

## Branch Information
⚠️ **Note**: This is the *resolution* branch. Other branches contain different functionality:
- `main`: Base implementation
- `optimization`: Performance improvements
- `validation`: Data validation features

## Project Overview
Chain-of-Thought (COT) generation implementation using LangGraph for structured reasoning processes. Handles complex question answering through step-by-step reasoning decomposition.

## Features
- Dynamic reasoning graph construction
- Multi-step question decomposition
- Intermediate validation checks
- Configurable resolution strategies

## Installation
```bash
git clone https://github.com/your-repo/COT-generation-using-langgraph.git
cd COT-generation-using-langgraph
pip install -r requirements.txt
```

## Key Files
- `cot.py`: Core reasoning graph implementation
- `question.py`: Question processing and decomposition
- `test.py`: Validation test cases
- `conversation_log.txt`: Example interaction log

## Usage
```python
from question import process_question
from cot import build_reasoning_graph

question = "Explain how photosynthesis works in plants"
reasoning_chain = build_reasoning_graph(process_question(question))
print(reasoning_chain.execute())
```

## Requirements
Please create a requirements.txt file with necessary dependencies. Common requirements might include:
```
langgraph>=0.1.0
python-dotenv>=1.0.0
```
