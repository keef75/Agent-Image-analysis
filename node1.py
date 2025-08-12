"""
Node 1: Structure Analysis for Cube Counting Federation
Analyzes 3D structure and determines base cube dimensions
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructureAnalysisOutput(BaseModel):
    """Pydantic model for Node 1 structured output"""
    analysis_type: str = Field(default="structure_decomposition", description="Type of analysis performed")
    base_cube_dimensions: Dict[str, int] = Field(
        description="X, Y, Z dimensions of the base cube structure"
    )
    total_theoretical_cubes: int = Field(
        description="Total number of unit cubes if structure were complete"
    )
    structure_type: str = Field(
        description="Description of the structure type (e.g., '3x3x3 cube with internal voids')"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence level in analysis (0.0 to 1.0)"
    )
    reasoning_path: str = Field(
        description="Detailed explanation of the analysis process"
    )
    
    model_config = {
        "json_schema_extra": {
            "required": ["base_cube_dimensions", "total_theoretical_cubes", "structure_type", "confidence", "reasoning_path"]
        }
    }
    
    @field_validator('total_theoretical_cubes')
    @classmethod
    def validate_cube_math(cls, v, info):
        """Ensure total cubes matches dimensions"""
        if info.data and 'base_cube_dimensions' in info.data:
            dims = info.data['base_cube_dimensions']
            expected = dims.get('x', 1) * dims.get('y', 1) * dims.get('z', 1)
            if v != expected:
                raise ValueError(f"Total cubes {v} doesn't match dimensions {dims}")
        return v

class Node1StructureAnalyzer:
    """Node 1: Analyzes overall structure and determines base cube dimensions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "node_1_structure_analysis"
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_analysis_prompt(self) -> str:
        """Create the prompt template for structure analysis"""
        return """
You are analyzing a 3D structure to determine its base dimensions and total theoretical cube count.

Looking at this image showing a cube structure with some missing pieces, analyze:

1. DIMENSIONS: What are the X, Y, Z dimensions of the complete structure this appears to be based on?
   - Look for grid patterns, visible edges, and geometric relationships
   - Common structures are 3x3x3, 4x4x4, 5x5x5, etc.

2. TOTAL COUNT: How many unit cubes would the complete structure contain?
   - This should equal X * Y * Z

3. STRUCTURE TYPE: Describe what type of structure this is
   - Include details about visible patterns, symmetries, or internal voids

4. REASONING: Explain your analysis step-by-step
   - What visual cues led to your dimension determination?
   - How did you account for perspective and occlusion?

5. CONFIDENCE: How certain are you in this analysis? (0.0 = very uncertain, 1.0 = completely certain)

Be systematic and explicit about your reasoning. Focus on identifying the underlying grid structure rather than counting individual visible cubes.

Please respond with a JSON object containing the following fields:
- analysis_type: string
- base_cube_dimensions: object with x, y, z integer values
- total_theoretical_cubes: integer
- structure_type: string
- confidence: number between 0.0 and 1.0
- reasoning_path: string
        """.strip()
    
    async def analyze_structure(self, image_path: str) -> StructureAnalysisOutput:
        """Analyze the structure and return structured output"""
        
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Create the prompt
            prompt = self.create_analysis_prompt()
            
            logger.info(f"Node 1: Starting structure analysis for {image_path}")
            
            # Define the schema manually for OpenAI API
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis performed"
                    },
                    "base_cube_dimensions": {
                        "type": "object",
                        "description": "X, Y, Z dimensions of the base cube structure",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "z": {"type": "integer"}
                        },
                        "required": ["x", "y", "z"],
                        "additionalProperties": False
                    },
                    "total_theoretical_cubes": {
                        "type": "integer",
                        "description": "Total number of unit cubes if structure were complete"
                    },
                    "structure_type": {
                        "type": "string",
                        "description": "Description of the structure type (e.g., '3x3x3 cube with internal voids')"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level in analysis (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "reasoning_path": {
                        "type": "string",
                        "description": "Detailed explanation of the analysis process"
                    }
                },
                "required": ["base_cube_dimensions", "total_theoretical_cubes", "structure_type", "confidence", "reasoning_path"],
                "additionalProperties": False
            }
            
            # Call OpenAI API with structured output
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            analysis_data = json.loads(response_text)
            
            # Create Pydantic model instance
            analysis = StructureAnalysisOutput(**analysis_data)
            
            logger.info(f"Node 1: Analysis complete. Dimensions: {analysis.base_cube_dimensions}, "
                       f"Total cubes: {analysis.total_theoretical_cubes}, "
                       f"Confidence: {analysis.confidence}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Node 1: Error during analysis: {str(e)}")
            raise

class FederationOrchestrator:
    """Basic orchestration framework for managing node communication"""
    
    def __init__(self):
        self.nodes = {}
        self.execution_graph = []
        self.results = {}
        
    def register_node(self, node_id: str, node_instance):
        """Register a node in the federation"""
        self.nodes[node_id] = node_instance
        logger.info(f"Registered node: {node_id}")
        
    def add_execution_step(self, node_id: str, inputs: Dict[str, Any]):
        """Add a step to the execution graph"""
        self.execution_graph.append({
            "node_id": node_id,
            "inputs": inputs
        })
        
    async def execute_node(self, node_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node and store results"""
        
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not registered")
            
        node = self.nodes[node_id]
        
        # For Node 1, we expect an image_path input
        if node_id == "node_1_structure_analysis":
            result = await node.analyze_structure(inputs["image_path"])
            
        # Store result for other nodes to use
        self.results[node_id] = result
        
        logger.info(f"Executed {node_id} successfully")
        return result.model_dump() if hasattr(result, 'model_dump') else result
        
    async def execute_federation(self) -> Dict[str, Any]:
        """Execute all nodes in the federation according to the graph"""
        
        execution_results = {}
        
        for step in self.execution_graph:
            node_id = step["node_id"]
            inputs = step["inputs"]
            
            # Add previous results to inputs if needed
            inputs["previous_results"] = self.results
            
            result = await self.execute_node(node_id, inputs)
            execution_results[node_id] = result
            
        return execution_results
        
    def get_node_result(self, node_id: str) -> Optional[Any]:
        """Get the result from a specific node"""
        return self.results.get(node_id)

# Example usage and testing
async def test_node1_cube_analysis():
    """Test Node 1 with the cube counting problem"""
    
    # Initialize Node 1
    node1 = Node1StructureAnalyzer()
    
    # Initialize orchestrator
    orchestrator = FederationOrchestrator()
    orchestrator.register_node("node_1_structure_analysis", node1)
    
    # Add execution step
    orchestrator.add_execution_step(
        "node_1_structure_analysis", 
        {"image_path": "cube_puzzle.png"}  # Path to your cube image
    )
    
    try:
        # Execute the analysis
        results = await orchestrator.execute_federation()
        
        # Print results
        print("=" * 50)
        print("NODE 1 STRUCTURE ANALYSIS RESULTS")
        print("=" * 50)
        
        node1_result = results["node_1_structure_analysis"]
        print(f"Analysis Type: {node1_result['analysis_type']}")
        print(f"Base Dimensions: {node1_result['base_cube_dimensions']}")
        print(f"Total Theoretical Cubes: {node1_result['total_theoretical_cubes']}")
        print(f"Structure Type: {node1_result['structure_type']}")
        print(f"Confidence: {node1_result['confidence']:.2f}")
        print(f"Reasoning: {node1_result['reasoning_path']}")
        
        # Save results for next nodes
        with open("node1_output.json", "w") as f:
            json.dump(node1_result, f, indent=2)
            
        print("\nResults saved to node1_output.json")
        return node1_result
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        return None

if __name__ == "__main__":
    import asyncio
    
    # Run the test
    result = asyncio.run(test_node1_cube_analysis())