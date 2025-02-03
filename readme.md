# COT Generation CLI (langcli branch)

## Branch Specifics
⚠️ **CLI Focus**: This branch contains command-line interface implementation for the COT generation system. Other branches include:
- `resolution`: Reasoning path resolution strategies
- `main`: Base API implementation
- `webui`: Browser-based interface

## CLI Features
```
Usage: cotcli [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose  Enable debug output
  
Commands:
  generate   Create reasoning chain from input
  validate   Check reasoning path consistency
  export     Save output to specified format
  config     Manage runtime parameters
```

## Installation
```bash
git checkout langcli
pip install -e .
```

## Example Workflow
```bash
# Generate reasoning chain from file input
cotcli generate --input=question.txt --format=markdown

# Validate existing reasoning path
cotcli validate --chain=output.json

# Export to different formats
cotcli export --input=chain.json --format=pdf
```

## Configuration
```bash
cotcli config set --model=gpt-4 --temperature=0.7
cotcli config list
```

## LangGraph CLI Visualization (langcli branch)

## Branch Purpose
✨ **Graph Visualization**: This branch specializes in visual graph analysis using LangGraph's CLI tools. Features:
- Web-based graph visualization
- Interactive node inspection
- Real-time modification preview

## Web Interface Setup
```bash
# Install CLI with visualization extras
pip install "langgraph[cli]"

# Start visualization server
langgraph serve --port 8501
```

Access the interface at: `http://localhost:8501`

## Key Commands
```bash
# Visualize a chain-of-thought graph
langgraph visualize ./cot_chain.json

# Generate web report from graph
langgraph export ./output/ --format web

# Live monitoring mode
langgraph monitor --watch ./graphs/
```

## Web UI Features
- 2D/3D graph layout toggling
- Node property inspection panel
- Edge weight visualization
- Export as PNG/SVG/PDF
