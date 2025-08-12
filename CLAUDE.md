# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **federated AI analysis system** designed to reduce hallucination through multi-agent consensus. The system analyzes 3D cube structures using 5 specialized nodes that cross-validate each other's findings and intelligently escalate to humans when confidence is insufficient.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies (no requirements.txt - install manually)
pip install openai pydantic

# Configure OpenAI API
export OPENAI_API_KEY="your-api-key-here"
```

### Running Complete Analysis
```bash
# Run complete federation analysis pipeline (RECOMMENDED)
python run_analysis.py cube_puzzle.png

# Run with any cube puzzle image
python run_analysis.py your_cube_image.png
```

### Running Individual Nodes (Advanced)
```bash
# Run individual nodes (requires previous node outputs)
python node1.py  # Needs node1_output.json
python node2.py  # Needs node1_output.json, node2_output.json
python node3.py  # Needs node1_output.json, node2_output.json
python node4.py  # Needs previous outputs
python node5.py  # Needs all previous outputs
```

### Testing Individual Components
```python
# Test a single node
from node1 import Node1StructureAnalyzer
analyzer = Node1StructureAnalyzer()
result = await analyzer.analyze_structure("cube_puzzle.png")

# Or use the complete pipeline programmatically
from run_analysis import CompleteFederationPipeline
pipeline = CompleteFederationPipeline()
result = await pipeline.run_complete_analysis("your_image.png")
```

## Architecture Overview

### Federation Design Pattern
The system implements a **sequential dependency chain** where each node builds upon previous analyses:

1. **Node 1 (Structure Analysis)** → Determines base dimensions and theoretical cube count
2. **Node 2 (Systematic Counting)** → Layer-by-layer visible cube enumeration  
3. **Node 3 (Void Detection)** → Missing cube pattern analysis via internal faces
4. **Node 4 (Perspective Correction)** → Occlusion and depth compensation
5. **Node 5 (Consensus Synthesis)** → Cross-validation and uncertainty quantification

### Key Architectural Concepts

**Structured Output Validation**: Every node uses Pydantic models with custom validators to ensure data consistency and mathematical correctness (e.g., `total_cubes = x * y * z`).

**Federation Orchestrator**: The `FederationOrchestrator` class manages node registration, execution sequencing, and result aggregation. It stores intermediate results for downstream nodes to access.

**Anti-Hallucination Pattern**: Instead of forcing confident answers, the system:
- Detects contradictions between specialized analyzers
- Quantifies system confidence based on inter-node agreement  
- Defers to human judgment when uncertainty exceeds thresholds (typically <0.5 confidence)
- Provides transparent reasoning chains for all conclusions

**Evidence-Based Reasoning**: All nodes must provide:
- Confidence scores (0.0-1.0) with evidence
- Detailed reasoning paths explaining methodology
- Specific findings that can be cross-validated

### Data Flow Pattern
```
cube_puzzle.png → Node1 → node1_output.json → Node2 → node2_output.json → ... → federation_final_answer.json
```

Each output JSON contains structured Pydantic model data that serves as input context for subsequent nodes.

### Critical Implementation Details

**OpenAI Integration**: Nodes use GPT-5 with structured output via `response_format={"type": "json_object"}` and manual schema definition for consistent parsing.

**Contradiction Detection**: Node 5 systematically compares findings across all previous nodes, identifying irreconcilable differences and calculating aggregate system confidence.

**Human Escalation Logic**: When contradictions cannot be resolved or system confidence falls below threshold, the system outputs `"answer_type": "DEFER_TO_HUMAN"` rather than guessing.

## Working with This Codebase

### Adding New Analysis Nodes
New nodes should inherit the pattern of:
1. Pydantic output model with validators
2. Async analysis method taking image path + previous results
3. OpenAI API call with structured output schema
4. Evidence-based confidence scoring

### Modifying Federation Logic
The orchestrator in `node1.py` contains the basic framework. For complex federations, consider:
- Parallel execution for independent analyses
- Dynamic node weighting based on historical accuracy
- Adversarial validation nodes that challenge consensus

### Understanding the Anti-Hallucination Approach
This system demonstrates **responsible AI** by acknowledging uncertainty rather than providing confident but potentially incorrect answers. The final output (`federation_final_answer.json`) shows the system correctly identified contradictions and deferred to human judgment with 41.7% confidence - a significant advancement over traditional AI approaches that would force a potentially wrong answer.

When working with this code, preserve this uncertainty quantification pattern as it's the core innovation distinguishing this approach from standard AI systems.