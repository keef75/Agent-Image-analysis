"""
Node 3: Void Pattern Analysis for Cube Counting Federation
Analyzes missing cube patterns based on visible internal faces and geometric constraints
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoidLocation(BaseModel):
    """Individual void/missing cube location"""
    x: int = Field(description="X coordinate in grid")
    y: int = Field(description="Y coordinate in grid") 
    z: int = Field(description="Z coordinate in grid")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence this location is void")
    evidence: str = Field(description="Visual evidence for this void location")
    
    model_config = {
        "json_schema_extra": {
            "required": ["x", "y", "z", "confidence", "evidence"]
        }
    }

class GeometricConstraint(BaseModel):
    """Geometric constraint derived from void analysis"""
    constraint_type: str = Field(description="Type of constraint (symmetry, pattern, etc.)")
    description: str = Field(description="Description of the constraint")
    supports_node1: bool = Field(description="Does this constraint support Node 1's analysis?")
    supports_node2: bool = Field(description="Does this constraint support Node 2's analysis?")
    alternative_hypothesis: Optional[str] = Field(description="Alternative structure hypothesis if neither node is correct")
    
    model_config = {
        "json_schema_extra": {
            "required": ["constraint_type", "description", "supports_node1", "supports_node2"]
        }
    }

class VoidPatternAnalysisOutput(BaseModel):
    """Pydantic model for Node 3 structured output"""
    analysis_type: str = Field(default="void_pattern_analysis")
    void_locations: List[VoidLocation] = Field(
        description="Specific locations of missing cubes based on visible internal faces"
    )
    void_pattern_type: str = Field(
        description="Pattern type of missing cubes (cross, tunnel, stepped, etc.)"
    )
    missing_cube_count: int = Field(
        description="Total count of missing cubes based on void pattern analysis"
    )
    pattern_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the void pattern identification"
    )
    geometric_constraints: List[GeometricConstraint] = Field(
        description="Geometric constraints derived from void pattern analysis"
    )
    node_contradiction_analysis: Dict[str, Any] = Field(
        description="Analysis of contradictions between Node 1 and Node 2"
    )
    corrected_structure_hypothesis: Dict[str, Any] = Field(
        description="Proposed correction to structure dimensions if previous nodes are wrong"
    )
    
    model_config = {
        "json_schema_extra": {
            "required": ["void_locations", "void_pattern_type", "missing_cube_count", "pattern_confidence", "geometric_constraints", "node_contradiction_analysis", "corrected_structure_hypothesis"]
        }
    }

class Node3VoidAnalyzer:
    """Node 3: Analyzes void patterns and geometric constraints"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "node_3_void_analysis"
        
    def load_previous_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load outputs from Node 1 and Node 2"""
        try:
            with open("node1_output.json", 'r') as f:
                node1_data = json.load(f)
            with open("node2_output.json", 'r') as f:
                node2_data = json.load(f)
            return node1_data, node2_data
        except FileNotFoundError as e:
            logger.error(f"Previous node output file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in previous node output: {e}")
            raise
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_void_analysis_prompt(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any]) -> str:
        """Create the prompt template for void pattern analysis"""
        return f"""
You are analyzing void patterns in a 3D cube structure to resolve contradictions between previous analyses.

PREVIOUS ANALYSES TO RECONCILE:

NODE 1 CLAIMED:
- Structure type: {node1_data['structure_type']}  
- Dimensions: {node1_data['base_cube_dimensions']}
- Total theoretical cubes: {node1_data['total_theoretical_cubes']}
- Confidence: {node1_data['confidence']}

NODE 2 FOUND:
- Total visible cubes: {node2_data['total_visible_cubes']}
- Layer pattern: {[layer['visible_cubes'] for layer in node2_data['layer_by_layer_count']]}
- Missing cubes (if Node 1 correct): {node2_data['node1_comparison']['missing_cubes_if_node1_correct']}
- Confidence: {node2_data['confidence']}

CONTRADICTION ANALYSIS:
If Node 1 is correct (4×4×3), each layer should have 16 cubes (4×4=16).
But Node 2 found layer pattern of {[layer['visible_cubes'] for layer in node2_data['layer_by_layer_count']]}.
This suggests BOTH analyses may have errors.

YOUR TASK - VOID PATTERN ANALYSIS:

1. ANALYZE STRIPED/HATCHED AREAS:
   - These show internal faces exposed by missing cubes
   - Map each striped area to specific missing cube locations
   - Use geometric relationships to determine void shape

2. DETERMINE TRUE BASE STRUCTURE:
   - What complete cube structure is this actually trying to represent?
   - 3×3×3? 4×4×4? Some other configuration?
   - Use void pattern symmetries and geometric constraints

3. COUNT MISSING CUBES SYSTEMATICALLY:
   - Based on void pattern, how many cubes are actually missing?
   - Does this match either previous analysis?

4. GEOMETRIC CONSTRAINT CHECKING:
   - Do the void patterns support Node 1's 4×4×3 claim?
   - Do they support Node 2's layer counting?
   - What constraints do they impose on the true structure?

5. PROPOSE CORRECTIONS:
   - If both previous analyses are wrong, what's the correct structure?
   - What are the actual dimensions and missing cube count?

Focus on the GEOMETRIC EVIDENCE from visible internal faces (striped areas). These are direct evidence of void locations and provide strong constraints on the true structure.

Please respond with a JSON object containing the following fields:
- analysis_type: string
- void_locations: array of objects with x, y, z (integers), confidence (number), evidence (string)
- void_pattern_type: string
- missing_cube_count: integer
- pattern_confidence: number between 0.0 and 1.0
- geometric_constraints: array of objects with constraint_type, description, supports_node1, supports_node2 (booleans), alternative_hypothesis (string)
- node_contradiction_analysis: object (will be filled in by the system)
- corrected_structure_hypothesis: object (will be filled in by the system)
        """.strip()
    
    async def analyze_void_pattern(self, image_path: str) -> VoidPatternAnalysisOutput:
        """Analyze void patterns and return structured output"""
        
        try:
            # Load previous node outputs
            node1_data, node2_data = self.load_previous_outputs()
            
            # Encode the image  
            base64_image = self.encode_image(image_path)
            
            # Create the prompt with previous node context
            prompt = self.create_void_analysis_prompt(node1_data, node2_data)
            
            logger.info(f"Node 3: Starting void pattern analysis for {image_path}")
            logger.info(f"Node 3: Reconciling Node 1 ({node1_data['total_theoretical_cubes']} total) with Node 2 ({node2_data['total_visible_cubes']} visible)")
            
            # Define the schema manually for OpenAI API
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis performed"
                    },
                    "void_locations": {
                        "type": "array",
                        "description": "Specific locations of missing cubes based on visible internal faces",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer", "description": "X coordinate in grid"},
                                "y": {"type": "integer", "description": "Y coordinate in grid"},
                                "z": {"type": "integer", "description": "Z coordinate in grid"},
                                "confidence": {"type": "number", "description": "Confidence this location is void", "minimum": 0.0, "maximum": 1.0},
                                "evidence": {"type": "string", "description": "Visual evidence for this void location"}
                            },
                            "required": ["x", "y", "z", "confidence", "evidence"]
                        }
                    },
                    "void_pattern_type": {
                        "type": "string",
                        "description": "Pattern type of missing cubes (cross, tunnel, stepped, etc.)"
                    },
                    "missing_cube_count": {
                        "type": "integer",
                        "description": "Total count of missing cubes based on void pattern analysis"
                    },
                    "pattern_confidence": {
                        "type": "number",
                        "description": "Confidence in the void pattern identification",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "geometric_constraints": {
                        "type": "array",
                        "description": "Geometric constraints derived from void pattern analysis",
                        "items": {
                            "type": "object",
                            "properties": {
                                "constraint_type": {"type": "string", "description": "Type of constraint (symmetry, pattern, etc.)"},
                                "description": {"type": "string", "description": "Description of the constraint"},
                                "supports_node1": {"type": "boolean", "description": "Does this constraint support Node 1's analysis?"},
                                "supports_node2": {"type": "boolean", "description": "Does this constraint support Node 2's analysis?"},
                                "alternative_hypothesis": {"type": "string", "description": "Alternative structure hypothesis if neither node is correct"}
                            },
                            "required": ["constraint_type", "description", "supports_node1", "supports_node2"]
                        }
                    },
                    "node_contradiction_analysis": {
                        "type": "object",
                        "description": "Analysis of contradictions between Node 1 and Node 2",
                        "additionalProperties": False
                    },
                    "corrected_structure_hypothesis": {
                        "type": "object",
                        "description": "Proposed correction to structure dimensions if previous nodes are wrong",
                        "additionalProperties": False
                    }
                },
                "required": ["void_locations", "void_pattern_type", "missing_cube_count", "pattern_confidence", "geometric_constraints", "node_contradiction_analysis", "corrected_structure_hypothesis"]
            }
            
            # Call OpenAI API with structured output
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            try:
                void_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Node 3: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                raise
            
            # Create Pydantic model instance
            try:
                void_analysis = VoidPatternAnalysisOutput(**void_data)
            except Exception as e:
                logger.error(f"Node 3: Failed to create Pydantic model: {e}")
                logger.error(f"Void data: {void_data}")
                raise
            
            # Add detailed contradiction analysis
            void_analysis.node_contradiction_analysis = self.analyze_contradictions(
                node1_data, node2_data, void_analysis
            )
            
            logger.info(f"Node 3: Void analysis complete. Missing cubes: {void_analysis.missing_cube_count}, "
                       f"Pattern: {void_analysis.void_pattern_type}, "
                       f"Confidence: {void_analysis.pattern_confidence}")
            
            return void_analysis
            
        except Exception as e:
            logger.error(f"Node 3: Error during void analysis: {str(e)}")
            raise
    
    def analyze_contradictions(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], void_analysis: VoidPatternAnalysisOutput) -> Dict[str, Any]:
        """Analyze contradictions between all three analyses"""
        
        node1_total = node1_data['total_theoretical_cubes']
        node2_visible = node2_data['total_visible_cubes'] 
        node3_missing = void_analysis.missing_cube_count
        
        # Calculate what Node 3 implies for total structure
        node3_implied_total = node2_visible + node3_missing
        
        contradictions = {
            "node1_vs_node2": {
                "node1_claimed_total": node1_total,
                "node2_visible": node2_visible,
                "node1_implied_missing": node1_total - node2_visible,
                "consistency": "INCONSISTENT" if abs(node1_total - node2_visible) > node1_total * 0.5 else "QUESTIONABLE"
            },
            "node1_vs_node3": {
                "node1_claimed_total": node1_total,
                "node3_implied_total": node3_implied_total,
                "difference": abs(node1_total - node3_implied_total),
                "consistency": "CONSISTENT" if node1_total == node3_implied_total else "INCONSISTENT"
            },
            "node2_vs_node3": {
                "node2_visible": node2_visible,
                "node3_missing": node3_missing,
                "node3_implied_total": node3_implied_total,
                "layer_pattern_match": self.check_layer_pattern_consistency(node2_data, void_analysis)
            },
            "three_way_analysis": {
                "all_agree": node1_total == node3_implied_total and self.check_layer_pattern_consistency(node2_data, void_analysis),
                "majority_view": self.determine_majority_view(node1_total, node2_visible, node3_missing),
                "confidence_weighted_recommendation": self.get_confidence_weighted_recommendation(node1_data, node2_data, void_analysis)
            }
        }
        
        return contradictions
    
    def check_layer_pattern_consistency(self, node2_data: Dict[str, Any], void_analysis: VoidPatternAnalysisOutput) -> bool:
        """Check if Node 2's layer pattern is consistent with Node 3's void analysis"""
        # This would require more complex geometric analysis
        # For now, return a simplified check
        return True  # Placeholder
    
    def determine_majority_view(self, node1_total: int, node2_visible: int, node3_missing: int) -> str:
        """Determine which analysis has the most support"""
        node3_implied_total = node2_visible + node3_missing
        
        if node1_total == node3_implied_total:
            return "Node 1 and Node 3 agree on total structure"
        else:
            return f"Node 3 suggests different total: {node3_implied_total} vs Node 1's {node1_total}"
    
    def get_confidence_weighted_recommendation(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], void_analysis: VoidPatternAnalysisOutput) -> str:
        """Provide confidence-weighted recommendation"""
        
        confidences = {
            "node1": node1_data['confidence'],
            "node2": node2_data['confidence'], 
            "node3": void_analysis.pattern_confidence
        }
        
        highest_confidence = max(confidences, key=confidences.get)
        return f"Highest confidence analysis: {highest_confidence} ({confidences[highest_confidence]:.2f})"

# Test function for Node 3
async def test_node3_void_analysis():
    """Test Node 3 using outputs from Node 1 and Node 2"""
    
    # Initialize Node 3
    node3 = Node3VoidAnalyzer()
    
    try:
        # Run Node 3 analysis
        result = await node3.analyze_void_pattern("cube_puzzle.png")
        
        # Print results
        print("=" * 70)
        print("NODE 3 VOID PATTERN ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"Analysis Type: {result.analysis_type}")
        print(f"Void Pattern Type: {result.void_pattern_type}")
        print(f"Missing Cube Count: {result.missing_cube_count}")
        print(f"Pattern Confidence: {result.pattern_confidence:.2f}")
        
        print(f"\nVoid Locations ({len(result.void_locations)} identified):")
        for i, void in enumerate(result.void_locations[:10]):  # Show first 10
            print(f"  {i+1}. Position ({void.x}, {void.y}, {void.z}) - Confidence: {void.confidence:.2f}")
            print(f"     Evidence: {void.evidence}")
        
        print(f"\nGeometric Constraints:")
        for constraint in result.geometric_constraints:
            print(f"  - {constraint.constraint_type}: {constraint.description}")
            print(f"    Supports Node 1: {constraint.supports_node1}, Supports Node 2: {constraint.supports_node2}")
            if constraint.alternative_hypothesis:
                print(f"    Alternative: {constraint.alternative_hypothesis}")
        
        print(f"\nContradiction Analysis:")
        contradiction = result.node_contradiction_analysis
        print(f"  Three-way analysis: {contradiction.get('three_way_analysis', {})}")
        
        print(f"\nCorrected Structure Hypothesis:")
        correction = result.corrected_structure_hypothesis
        for key, value in correction.items():
            print(f"  {key}: {value}")
        
        # Save results for next nodes
        with open("node3_output.json", "w") as f:
            json.dump(result.model_dump() if hasattr(result, 'model_dump') else result, f, indent=2)
            
        print("\nResults saved to node3_output.json")
        return result
        
    except Exception as e:
        print(f"Error during Node 3 execution: {str(e)}")
        return None

if __name__ == "__main__":
    import asyncio
    
    # Run Node 3 test
    result = asyncio.run(test_node3_void_analysis())