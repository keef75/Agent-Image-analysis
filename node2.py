"""
Node 2: Systematic Cube Counting for Cube Counting Federation
Performs methodical grid-by-grid counting of visible cubes
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

class LayerCount(BaseModel):
    """Individual layer counting data"""
    layer: int = Field(description="Layer number (1=front, 2=middle, 3=back)")
    visible_cubes: int = Field(description="Number of cubes visible in this layer")
    counting_method: str = Field(description="How this layer was counted")
    notes: str = Field(description="Any observations about this layer")
    
    model_config = {
        "json_schema_extra": {
            "required": ["layer", "visible_cubes", "counting_method", "notes"]
        }
    }
    
    @field_validator('layer', mode='before')
    @classmethod
    def parse_layer_number(cls, v):
        """Convert descriptive layer names to integers"""
        if isinstance(v, str):
            # Extract number from strings like "Layer 1 (front)" or "Layer 2"
            import re
            match = re.search(r'(\d+)', v)
            if match:
                return int(match.group(1))
            # Fallback: try to map common descriptions
            v_lower = v.lower()
            if 'front' in v_lower or 'first' in v_lower:
                return 1
            elif 'middle' in v_lower or 'second' in v_lower:
                return 2
            elif 'back' in v_lower or 'third' in v_lower:
                return 3
            else:
                raise ValueError(f"Could not parse layer number from: {v}")
        return v

class SystematicCountingOutput(BaseModel):
    """Pydantic model for Node 2 structured output"""
    analysis_type: str = Field(default="systematic_counting")
    layer_by_layer_count: List[LayerCount] = Field(
        description="Detailed breakdown of counting by layers or sections"
    )
    total_visible_cubes: int = Field(
        description="Total number of cubes currently visible/present"
    )
    counting_method: str = Field(
        description="Overall methodology used for systematic counting"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level in counting accuracy (0.0 to 1.0)"
    )
    potential_occlusion_issues: List[str] = Field(
        description="Areas where occlusion might affect count accuracy"
    )
    node1_comparison: Dict[str, Any] = Field(
        description="Comparison with Node 1's dimensional analysis"
    )
    
    model_config = {
        "json_schema_extra": {
            "required": ["layer_by_layer_count", "total_visible_cubes", "counting_method", "confidence", "potential_occlusion_issues", "node1_comparison"]
        }
    }
    
    @field_validator('total_visible_cubes')
    @classmethod
    def validate_layer_sum(cls, v, info):
        """Ensure total matches sum of layer counts"""
        if info.data and 'layer_by_layer_count' in info.data:
            layer_sum = sum(layer.visible_cubes for layer in info.data['layer_by_layer_count'])
            if v != layer_sum:
                logger.warning(f"Total visible cubes {v} doesn't match layer sum {layer_sum}")
        return v

class Node2SystematicCounter:
    """Node 2: Performs systematic cube-by-cube counting"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "node_2_systematic_counting"
        
    def load_node1_output(self, filepath: str = "node1_output.json") -> Dict[str, Any]:
        """Load Node 1's analysis results"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Node 1 output file not found: {filepath}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Node 1 output: {filepath}")
            raise
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_counting_prompt(self, node1_data: Dict[str, Any]) -> str:
        """Create the prompt template for systematic counting"""
        return f"""
You are performing systematic cube counting on a 3D structure image. 

CONTEXT FROM NODE 1:
Node 1 analyzed this as: {node1_data['structure_type']}
Claimed dimensions: {node1_data['base_cube_dimensions']}
Claimed total theoretical cubes: {node1_data['total_theoretical_cubes']}

YOUR TASK: Count the ACTUAL VISIBLE/PRESENT cubes systematically

METHODOLOGY:
1. LAYER-BY-LAYER ANALYSIS: Break the structure into depth layers (front to back)
   - Layer 1 (front): Count cubes in the frontmost visible layer
   - Layer 2 (middle): Count cubes in the middle layer(s)  
   - Layer 3 (back): Count cubes in the back layer(s)

2. SYSTEMATIC GRID COUNTING: For each layer, count methodically
   - Go row by row, column by column
   - Account for visible vs hidden cubes
   - Note any internal voids or missing sections

3. OCCLUSION ANALYSIS: Identify areas where perspective might hide cubes
   - What cubes might be behind visible ones?
   - What areas show internal structure (striped/hatched regions)?

4. VERIFICATION: Check your counting
   - Does the count make geometric sense?
   - Do the dimensions match what you observe?

5. COMPARE WITH NODE 1: 
   - Does your count support Node 1's dimensional analysis?
   - If not, what discrepancies do you observe?

Be extremely methodical. Count visible cubes only, but note where you suspect hidden cubes exist.

Please respond with a JSON object containing the following fields:
- analysis_type: string
- layer_by_layer_count: array of objects with:
  * layer: integer (1=front, 2=middle, 3=back)
  * visible_cubes: integer
  * counting_method: string
  * notes: string
- total_visible_cubes: integer
- counting_method: string
- confidence: number between 0.0 and 1.0
- potential_occlusion_issues: array of strings
- node1_comparison: object (will be filled in by the system)

IMPORTANT: The 'layer' field must be an integer (1, 2, 3), not a string description.
        """.strip()
    
    async def count_cubes_systematically(self, image_path: str, node1_output: Dict[str, Any]) -> SystematicCountingOutput:
        """Perform systematic cube counting and return structured output"""
        
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Create the prompt with Node 1 context
            prompt = self.create_counting_prompt(node1_output)
            
            logger.info(f"Node 2: Starting systematic counting for {image_path}")
            logger.info(f"Node 2: Using Node 1 context - claimed {node1_output['total_theoretical_cubes']} total cubes")
            
            # Define the schema manually for OpenAI API
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis performed"
                    },
                    "layer_by_layer_count": {
                        "type": "array",
                        "description": "Detailed breakdown of counting by layers or sections",
                        "items": {
                            "type": "object",
                            "properties": {
                                "layer": {"type": "integer", "description": "Layer number (1=front, 2=middle, 3=back)"},
                                "visible_cubes": {"type": "integer", "description": "Number of cubes visible in this layer"},
                                "counting_method": {"type": "string", "description": "How this layer was counted"},
                                "notes": {"type": "string", "description": "Any observations about this layer"}
                            },
                            "required": ["layer", "visible_cubes", "counting_method", "notes"]
                        }
                    },
                    "total_visible_cubes": {
                        "type": "integer",
                        "description": "Total number of cubes currently visible/present"
                    },
                    "counting_method": {
                        "type": "string",
                        "description": "Overall methodology used for systematic counting"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level in counting accuracy (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "potential_occlusion_issues": {
                        "type": "array",
                        "description": "Areas where occlusion might affect count accuracy",
                        "items": {"type": "string"}
                    },
                    "node1_comparison": {
                        "type": "object",
                        "description": "Comparison with Node 1's dimensional analysis",
                        "additionalProperties": False
                    }
                },
                "required": ["layer_by_layer_count", "total_visible_cubes", "counting_method", "confidence", "potential_occlusion_issues", "node1_comparison"]
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
            try:
                counting_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Node 2: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                raise
            
            # Create Pydantic model instance
            try:
                counting = SystematicCountingOutput(**counting_data)
            except Exception as e:
                logger.error(f"Node 2: Failed to create Pydantic model: {e}")
                logger.error(f"Counting data: {counting_data}")
                raise
            
            # Add Node 1 comparison data
            counting.node1_comparison = {
                "node1_total_theoretical": node1_output['total_theoretical_cubes'],
                "node1_dimensions": node1_output['base_cube_dimensions'],
                "node2_visible_count": counting.total_visible_cubes,
                "missing_cubes_if_node1_correct": node1_output['total_theoretical_cubes'] - counting.total_visible_cubes,
                "dimensional_consistency": self.check_dimensional_consistency(counting, node1_output)
            }
            
            logger.info(f"Node 2: Counting complete. Visible cubes: {counting.total_visible_cubes}, "
                       f"Node 1 claimed total: {node1_output['total_theoretical_cubes']}, "
                       f"Confidence: {counting.confidence}")
            
            return counting
            
        except Exception as e:
            logger.error(f"Node 2: Error during counting: {str(e)}")
            raise
    
    def check_dimensional_consistency(self, counting: SystematicCountingOutput, node1_data: Dict[str, Any]) -> str:
        """Check if the counting is consistent with Node 1's dimensional analysis"""
        
        node1_total = node1_data['total_theoretical_cubes']
        visible_count = counting.total_visible_cubes
        
        if visible_count > node1_total:
            return "INCONSISTENT: Visible cubes exceed Node 1's total theoretical count"
        elif visible_count == node1_total:
            return "PERFECT_MATCH: All theoretical cubes are visible (no missing cubes)"
        else:
            missing = node1_total - visible_count
            return f"CONSISTENT: {missing} cubes missing from Node 1's {node1_total} total"

# Extended orchestrator to handle Node 2
class FederationOrchestrator:
    """Enhanced orchestration framework for managing multi-node communication"""
    
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
        
        # Route to appropriate node method
        if node_id == "node_1_structure_analysis":
            result = await node.analyze_structure(inputs["image_path"])
        elif node_id == "node_2_systematic_counting":
            # Node 2 needs both image and Node 1 results
            node1_data = inputs.get("node1_data") or self.results.get("node_1_structure_analysis")
            if not node1_data:
                raise ValueError("Node 2 requires Node 1 results")
            
            # Convert Pydantic model to dict if needed
            if hasattr(node1_data, 'model_dump'):
                node1_data = node1_data.model_dump()
            elif hasattr(node1_data, 'dict'):
                node1_data = node1_data.dict()
                
            result = await node.count_cubes_systematically(inputs["image_path"], node1_data)
            
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

# Test function for Node 2
async def test_node2_with_node1():
    """Test Node 2 using Node 1's existing output"""
    
    # Initialize Node 2
    node2 = Node2SystematicCounter()
    
    # Load Node 1's output
    node1_data = node2.load_node1_output("node1_output.json")
    
    # Run Node 2 analysis
    try:
        result = await node2.count_cubes_systematically("cube_puzzle.png", node1_data)
        
        # Print results
        print("=" * 60)
        print("NODE 2 SYSTEMATIC COUNTING RESULTS")
        print("=" * 60)
        
        print(f"Analysis Type: {result.analysis_type}")
        print(f"Total Visible Cubes: {result.total_visible_cubes}")
        print(f"Counting Method: {result.counting_method}")
        print(f"Confidence: {result.confidence:.2f}")
        
        print("\nLayer-by-Layer Breakdown:")
        for layer in result.layer_by_layer_count:
            print(f"  Layer {layer.layer}: {layer.visible_cubes} cubes ({layer.counting_method})")
            if layer.notes:
                print(f"    Notes: {layer.notes}")
        
        print(f"\nNode 1 Comparison:")
        comparison = result.node1_comparison
        print(f"  Node 1 claimed total: {comparison['node1_total_theoretical']}")
        print(f"  Node 2 visible count: {comparison['node2_visible_count']}")
        print(f"  Missing cubes (if Node 1 correct): {comparison['missing_cubes_if_node1_correct']}")
        print(f"  Consistency check: {comparison['dimensional_consistency']}")
        
        if result.potential_occlusion_issues:
            print(f"\nOcclusion Issues:")
            for issue in result.potential_occlusion_issues:
                print(f"  - {issue}")
        
        # Save results for next nodes
        with open("node2_output.json", "w") as f:
            json.dump(result.model_dump() if hasattr(result, 'model_dump') else result, f, indent=2)
            
        print("\nResults saved to node2_output.json")
        return result
        
    except Exception as e:
        print(f"Error during Node 2 execution: {str(e)}")
        return None

# Two-node federation test
async def test_two_node_federation():
    """Test the full two-node pipeline"""
    
    # This assumes node1.py has already run and created node1_output.json
    # We'll just test Node 2 standalone for now
    return await test_node2_with_node1()

if __name__ == "__main__":
    import asyncio
    
    # Run Node 2 test
    result = asyncio.run(test_two_node_federation())