"""
Node 4: Cross-Verification Calculator for Cube Counting Federation
Verifies consistency between different counting approaches and flags contradictions
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsistencyCheck(BaseModel):
    """Individual consistency check between analyses"""
    check_name: str = Field(description="Name of the consistency check")
    description: str = Field(description="What this check verifies")
    passed: bool = Field(description="Did this check pass?")
    expected_value: Optional[Union[float, int, str]] = Field(description="Expected value if consistent")
    actual_value: Optional[Union[float, int, str]] = Field(description="Actual observed value")
    discrepancy_magnitude: Optional[float] = Field(description="How large is the discrepancy?")
    severity: str = Field(description="MINOR, MODERATE, MAJOR, CRITICAL")
    
    model_config = {
        "json_schema_extra": {
            "required": ["check_name", "description", "passed", "severity"]
        }
    }

    @field_validator('expected_value', 'actual_value', mode='before')
    @classmethod
    def coerce_expected_actual(cls, v):
        """Allow numbers or simple strings like 'Possible'; coerce numeric-like strings to numbers."""
        if v is None:
            return v
        # Pass through numbers
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try to coerce numeric-looking strings
            try:
                # int first to avoid converting '01' to 1.0 unnecessarily
                if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                    return int(s)
                return float(s)
            except ValueError:
                return s  # keep as descriptive string like 'Possible'
        return v

class DiscrepancyAnalysis(BaseModel):
    """Analysis of discrepancies between nodes"""
    discrepancy_type: str = Field(description="Type of discrepancy found")
    nodes_involved: List[str] = Field(description="Which nodes have this discrepancy")
    description: str = Field(description="Detailed description of the discrepancy") 
    potential_causes: List[str] = Field(description="Possible reasons for this discrepancy")
    resolution_difficulty: str = Field(description="EASY, MODERATE, HARD, IMPOSSIBLE")
    
    model_config = {
        "json_schema_extra": {
            "required": ["discrepancy_type", "nodes_involved", "description", "potential_causes", "resolution_difficulty"]
        }
    }
    
    @field_validator('nodes_involved', mode='before')
    @classmethod
    def convert_node_numbers_to_strings(cls, v):
        """Convert node numbers to string format"""
        if isinstance(v, list):
            return [f"node{node}" if isinstance(node, int) else str(node) for node in v]
        return v

class MathematicalVerification(BaseModel):
    """Mathematical verification of counting relationships"""
    total_theoretical_claims: Dict[str, int] = Field(description="Each node's claim about total cubes")
    visible_cube_claims: Dict[str, int] = Field(description="Each node's claim about visible cubes")
    missing_cube_claims: Dict[str, int] = Field(description="Each node's claim about missing cubes")
    mathematical_consistency: Dict[str, bool] = Field(description="Whether each node's math adds up")
    cross_node_consistency: Dict[str, str] = Field(description="Consistency between different nodes")
    
    model_config = {
        "json_schema_extra": {
            "required": ["total_theoretical_claims", "visible_cube_claims", "missing_cube_claims", "mathematical_consistency", "cross_node_consistency"]
        }
    }
    
    @field_validator('*', mode='before')
    @classmethod
    def handle_nested_structure(cls, v):
        """Handle nested structure from AI response"""
        if isinstance(v, dict):
            # If the AI returned a nested structure like {'node_1': {'total_cubes': 48}}, 
            # extract the values we need
            if 'node_1' in v and isinstance(v['node_1'], dict):
                # Extract total cubes from nested structure
                total_claims = {}
                visible_claims = {}
                missing_claims = {}
                consistency = {}
                cross_consistency = {}
                
                for node_key, node_data in v.items():
                    if isinstance(node_data, dict):
                        if 'calculated_total' in node_data:
                            total_claims[node_key] = node_data['calculated_total']
                        if 'visible_cubes' in node_data:
                            visible_claims[node_key] = node_data['visible_cubes']
                        if 'missing_cubes' in node_data:
                            missing_claims[node_key] = node_data['missing_cubes']
                        if 'dimensions_check' in node_data:
                            consistency[f"{node_key}_dimensions"] = node_data['dimensions_check']
                
                # Return a dict that matches our expected structure
                return {
                    'total_theoretical_claims': total_claims,
                    'visible_cube_claims': visible_claims,
                    'missing_cube_claims': missing_claims,
                    'mathematical_consistency': consistency,
                    'cross_node_consistency': cross_consistency
                }
        return v

class ConfidenceAnalysis(BaseModel):
    """Analysis of confidence levels vs actual agreement"""
    node_confidences: Dict[str, float] = Field(description="Confidence level of each node")
    agreement_level: float = Field(description="How much do the nodes actually agree?")
    confidence_calibration: str = Field(description="Are high-confidence claims actually reliable?")
    uncertainty_recommendation: str = Field(description="Should the system be more uncertain?")
    
    model_config = {
        "json_schema_extra": {
            "required": ["node_confidences", "agreement_level", "confidence_calibration", "uncertainty_recommendation"]
        }
    }
    
    @field_validator('*', mode='before')
    @classmethod
    def handle_nested_confidence_structure(cls, v):
        """Handle nested confidence structure from AI response"""
        if isinstance(v, dict):
            # If the AI returned a nested structure, extract what we need
            if 'node_1' in v and isinstance(v['node_1'], (int, float)):
                # Extract confidence values
                confidences = {}
                for key, value in v.items():
                    if key.startswith('node_') and isinstance(value, (int, float)):
                        confidences[key] = float(value)
                
                # Look for agreement level and other fields
                agreement = v.get('agreement_level', 0.5)
                calibration = v.get('confidence_calibration', 'UNKNOWN')
                recommendation = v.get('uncertainty_recommendation', 'UNKNOWN')
                
                # If we found the overall_confidence field, use it for calibration
                if 'overall_confidence' in v:
                    calibration = v['overall_confidence']
                
                return {
                    'node_confidences': confidences,
                    'agreement_level': float(agreement) if isinstance(agreement, (int, float)) else 0.5,
                    'confidence_calibration': str(calibration),
                    'uncertainty_recommendation': str(recommendation)
                }
        return v

class CrossVerificationOutput(BaseModel):
    """Pydantic model for Node 4 structured output"""
    analysis_type: str = Field(default="cross_verification")
    consistency_checks: List[ConsistencyCheck] = Field(
        description="Detailed consistency checks between analyses"
    )
    discrepancies: List[DiscrepancyAnalysis] = Field(
        description="Major discrepancies found between nodes"
    )
    mathematical_verification: MathematicalVerification = Field(
        description="Mathematical consistency analysis"
    )
    confidence_analysis: ConfidenceAnalysis = Field(
        description="Analysis of confidence vs agreement"
    )
    overall_system_health: str = Field(
        description="HEALTHY, CONCERNING, PROBLEMATIC, CRITICAL"
    )
    contradiction_severity: str = Field(
        description="How severe are the contradictions?"
    )
    recommended_action: str = Field(
        description="What should the system do next?"
    )
    confidence_in_verification: float = Field(
        ge=0.0, le=1.0,
        description="How confident is this verification analysis?"
    )
    
    model_config = {
        "json_schema_extra": {
            "required": ["consistency_checks", "discrepancies", "mathematical_verification", "confidence_analysis", "overall_system_health", "contradiction_severity", "recommended_action", "confidence_in_verification"]
        }
    }

class Node4CrossVerifier:
    """Node 4: Cross-verifies consistency between all previous analyses"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "node_4_cross_verification"
        
    def load_all_previous_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load outputs from all previous nodes"""
        try:
            with open("node1_output.json", 'r') as f:
                node1_data = json.load(f)
            with open("node2_output.json", 'r') as f:
                node2_data = json.load(f)
            with open("node3_output.json", 'r') as f:
                node3_data = json.load(f)
            return node1_data, node2_data, node3_data
        except FileNotFoundError as e:
            logger.error(f"Previous node output file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in previous node output: {e}")
            raise
    
    def create_verification_prompt(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], node3_data: Dict[str, Any]) -> str:
        """Create the prompt template for cross-verification"""
        return f"""
You are performing cross-verification analysis on three different cube counting approaches that have produced contradictory results.

ANALYSIS SUMMARY TO VERIFY:

NODE 1 (STRUCTURE ANALYSIS):
- Claimed structure: {node1_data['structure_type']}
- Dimensions: {node1_data['base_cube_dimensions']}
- Total theoretical cubes: {node1_data['total_theoretical_cubes']}
- Confidence: {node1_data['confidence']}

NODE 2 (SYSTEMATIC COUNTING):
- Total visible cubes: {node2_data['total_visible_cubes']}
- Layer breakdown: {[layer['visible_cubes'] for layer in node2_data['layer_by_layer_count']]}
- Confidence: {node2_data['confidence']}
- Implied missing (using Node 1's total): {node2_data['node1_comparison']['missing_cubes_if_node1_correct']}

NODE 3 (VOID PATTERN ANALYSIS):
- Missing cube count: {node3_data['missing_cube_count']}
- Void pattern: {node3_data['void_pattern_type']}
- Confidence: {node3_data['pattern_confidence']}
- Implied total structure: {node2_data['total_visible_cubes'] + node3_data['missing_cube_count']} cubes

CRITICAL CONTRADICTIONS TO ANALYZE:

1. TOTAL STRUCTURE SIZE DISAGREEMENT:
   - Node 1 claims: {node1_data['total_theoretical_cubes']} total cubes
   - Node 2 + Node 3 imply: {node2_data['total_visible_cubes'] + node3_data['missing_cube_count']} total cubes
   - Difference: {abs(node1_data['total_theoretical_cubes'] - (node2_data['total_visible_cubes'] + node3_data['missing_cube_count']))} cubes

2. MATHEMATICAL CONSISTENCY:
   - Does Node 1's claimed dimensions actually yield the claimed total?
   - Does visible + missing = claimed total for each analysis?
   - Are the layer patterns geometrically possible?

3. CONFIDENCE vs ACCURACY:
   - All nodes are highly confident but disagree dramatically
   - What does this say about the reliability of individual analyses?

YOUR VERIFICATION TASKS:

1. CHECK MATHEMATICAL CONSISTENCY:
   - Verify each node's internal arithmetic
   - Check if total = visible + missing for each analysis
   - Identify mathematical errors or impossibilities

2. IDENTIFY MAJOR DISCREPANCIES:
   - Where do the analyses contradict each other most severely?
   - Which discrepancies could be resolved vs fundamental contradictions?

3. CONFIDENCE CALIBRATION ANALYSIS:
   - High confidence + major disagreement = poorly calibrated uncertainty
   - Should the system be much less confident given these contradictions?

4. SYSTEM HEALTH ASSESSMENT:
   - When expert analyses disagree this much, what does it mean?
   - Is this a solvable contradiction or fundamental analysis failure?

5. RECOMMEND NEXT STEPS:
   - Can these contradictions be resolved through additional analysis?
   - Should the system defer to human judgment?
   - Are there objective verification methods we haven't tried?

Focus on identifying WHY these three approaches yielded such different results and whether the contradictions can be resolved or represent fundamental analysis failures.

Please respond with a JSON object containing the following fields:
- analysis_type: string
- consistency_checks: array of objects with check_name, description, passed (boolean), expected_value, actual_value, discrepancy_magnitude, severity
- discrepancies: array of objects with discrepancy_type, nodes_involved (array of strings like "node1", "node2"), description, potential_causes (array), resolution_difficulty
- mathematical_verification: object (will be filled in by the system)
- confidence_analysis: object (will be filled in by the system)
- overall_system_health: string (HEALTHY, CONCERNING, PROBLEMATIC, CRITICAL)
- contradiction_severity: string
- recommended_action: string
- confidence_in_verification: number between 0.0 and 1.0

IMPORTANT: For nodes_involved, use string format like "node1", "node2", "node3" (not numbers).
        """.strip()
    
    async def verify_cross_consistency(self) -> CrossVerificationOutput:
        """Perform cross-verification analysis and return structured output"""
        
        try:
            # Load all previous node outputs
            node1_data, node2_data, node3_data = self.load_all_previous_outputs()
            
            # Create the verification prompt
            prompt = self.create_verification_prompt(node1_data, node2_data, node3_data)
            
            logger.info(f"Node 4: Starting cross-verification analysis")
            logger.info(f"Node 4: Analyzing contradictions between {node1_data['total_theoretical_cubes']}, {node2_data['total_visible_cubes']}, and {node3_data['missing_cube_count']} cube counts")
            
            # Define the schema manually for OpenAI API
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis performed"
                    },
                    "consistency_checks": {
                        "type": "array",
                        "description": "Detailed consistency checks between analyses",
                        "items": {
                            "type": "object",
                            "properties": {
                                "check_name": {"type": "string", "description": "Name of the consistency check"},
                                "description": {"type": "string", "description": "What this check verifies"},
                                "passed": {"type": "boolean", "description": "Did this check pass?"},
                                "expected_value": {"oneOf": [{"type": "number"}, {"type": "string"}], "description": "Expected value if consistent"},
                                "actual_value": {"oneOf": [{"type": "number"}, {"type": "string"}], "description": "Actual observed value"},
                                "discrepancy_magnitude": {"type": "number", "description": "How large is the discrepancy?"},
                                "severity": {"type": "string", "description": "MINOR, MODERATE, MAJOR, CRITICAL"}
                            },
                            "required": ["check_name", "description", "passed", "severity"]
                        }
                    },
                    "discrepancies": {
                        "type": "array",
                        "description": "Major discrepancies found between nodes",
                        "items": {
                            "type": "object",
                            "properties": {
                                "discrepancy_type": {"type": "string", "description": "Type of discrepancy found"},
                                "nodes_involved": {"type": "array", "description": "Which nodes have this discrepancy", "items": {"type": "string"}},
                                "description": {"type": "string", "description": "Detailed description of the discrepancy"},
                                "potential_causes": {"type": "array", "description": "Possible reasons for this discrepancy", "items": {"type": "string"}},
                                "resolution_difficulty": {"type": "string", "description": "EASY, MODERATE, HARD, IMPOSSIBLE"}
                            },
                            "required": ["discrepancy_type", "nodes_involved", "description", "potential_causes", "resolution_difficulty"]
                        }
                    },
                    "mathematical_verification": {
                        "type": "object",
                        "description": "Mathematical consistency analysis",
                        "additionalProperties": False
                    },
                    "confidence_analysis": {
                        "type": "object",
                        "description": "Analysis of confidence vs agreement",
                        "additionalProperties": False
                    },
                    "overall_system_health": {
                        "type": "string",
                        "description": "HEALTHY, CONCERNING, PROBLEMATIC, CRITICAL"
                    },
                    "contradiction_severity": {
                        "type": "string",
                        "description": "How severe are the contradictions?"
                    },
                    "recommended_action": {
                        "type": "string",
                        "description": "What should the system do next?"
                    },
                    "confidence_in_verification": {
                        "type": "number",
                        "description": "How confident is this verification analysis?",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["consistency_checks", "discrepancies", "mathematical_verification", "confidence_analysis", "overall_system_health", "contradiction_severity", "recommended_action", "confidence_in_verification"]
            }
            
            # Call OpenAI API with structured output
            response = self.client.chat.completions.create(
                model="gpt-5",  # Using consistent model for verification
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            try:
                verification_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Node 4: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                raise

            # Pre-compute required sections and inject if missing/empty
            computed_math = self.compute_mathematical_verification(node1_data, node2_data, node3_data)
            computed_conf = self.compute_confidence_analysis(node1_data, node2_data, node3_data, None)

            if not isinstance(verification_data.get("mathematical_verification"), dict) or not verification_data["mathematical_verification"]:
                verification_data["mathematical_verification"] = computed_math.model_dump() if hasattr(computed_math, 'model_dump') else computed_math
            if not isinstance(verification_data.get("confidence_analysis"), dict) or not verification_data["confidence_analysis"]:
                verification_data["confidence_analysis"] = computed_conf.model_dump() if hasattr(computed_conf, 'model_dump') else computed_conf

            # Create Pydantic model instance
            try:
                verification = CrossVerificationOutput(**verification_data)
            except Exception as e:
                logger.error(f"Node 4: Failed to create Pydantic model: {e}")
                logger.error(f"Verification data: {verification_data}")
                raise
            
            logger.info(f"Node 4: Verification complete. System health: {verification.overall_system_health}, "
                       f"Contradiction severity: {verification.contradiction_severity}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Node 4: Error during verification: {str(e)}")
            raise
    
    def compute_mathematical_verification(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], node3_data: Dict[str, Any]) -> MathematicalVerification:
        """Compute mathematical consistency checks"""
        
        # Extract key numbers
        node1_total = node1_data['total_theoretical_cubes']
        node1_dims = node1_data['base_cube_dimensions']
        node2_visible = node2_data['total_visible_cubes']
        node3_missing = node3_data['missing_cube_count']
        
        # Compute expected values
        node1_expected_total = node1_dims['x'] * node1_dims['y'] * node1_dims['z']
        node23_implied_total = node2_visible + node3_missing
        
        return MathematicalVerification(
            total_theoretical_claims={
                "node1": node1_total,
                "node2_plus_node3": node23_implied_total
            },
            visible_cube_claims={
                "node2": node2_visible
            },
            missing_cube_claims={
                "node3": node3_missing,
                "node1_implied": max(0, node1_total - node2_visible)
            },
            mathematical_consistency={
                "node1_dimensions_math": node1_total == node1_expected_total,
                "node2_plus_node3_math": True,  # By definition
                "cross_node_agreement": node1_total == node23_implied_total
            },
            cross_node_consistency={
                "node1_vs_node23": "CONSISTENT" if node1_total == node23_implied_total else f"INCONSISTENT: {abs(node1_total - node23_implied_total)} cube difference",
                "overall": "MAJOR_CONTRADICTIONS" if abs(node1_total - node23_implied_total) > 10 else "MINOR_DIFFERENCES"
            }
        )
    
    def compute_confidence_analysis(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], node3_data: Dict[str, Any], verification: Optional[CrossVerificationOutput] = None) -> ConfidenceAnalysis:
        """Analyze confidence levels vs actual agreement"""
        
        confidences = {
            "node1": node1_data['confidence'],
            "node2": node2_data['confidence'],
            "node3": node3_data['pattern_confidence']
        }
        
        # Compute agreement level based on how close the analyses are
        node1_total = node1_data['total_theoretical_cubes']
        node23_total = node2_data['total_visible_cubes'] + node3_data['missing_cube_count']
        
        agreement_level = 1.0 - min(1.0, abs(node1_total - node23_total) / max(node1_total, node23_total))
        avg_confidence = statistics.mean(confidences.values())
        
        # Confidence calibration analysis
        if avg_confidence > 0.8 and agreement_level < 0.5:
            calibration = "POORLY_CALIBRATED: High confidence but low agreement"
            uncertainty_rec = "INCREASE_UNCERTAINTY: System should be much less confident"
        elif avg_confidence > 0.6 and agreement_level < 0.7:
            calibration = "OVERCONFIDENT: Moderate confidence but significant disagreement"
            uncertainty_rec = "MODERATE_UNCERTAINTY: Some caution warranted"
        else:
            calibration = "REASONABLE: Confidence roughly matches agreement"
            uncertainty_rec = "MAINTAIN_CURRENT: Confidence levels seem appropriate"
        
        return ConfidenceAnalysis(
            node_confidences=confidences,
            agreement_level=agreement_level,
            confidence_calibration=calibration,
            uncertainty_recommendation=uncertainty_rec
        )

# Test function for Node 4
async def test_node4_cross_verification():
    """Test Node 4 using outputs from all previous nodes"""
    
    # Initialize Node 4
    node4 = Node4CrossVerifier()
    
    try:
        # Run Node 4 verification
        result = await node4.verify_cross_consistency()
        
        # Print results
        print("=" * 80)
        print("NODE 4 CROSS-VERIFICATION ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"Analysis Type: {result.analysis_type}")
        print(f"Overall System Health: {result.overall_system_health}")
        print(f"Contradiction Severity: {result.contradiction_severity}")
        print(f"Recommended Action: {result.recommended_action}")
        print(f"Verification Confidence: {result.confidence_in_verification:.2f}")
        
        print(f"\nConsistency Checks ({len(result.consistency_checks)} performed):")
        for check in result.consistency_checks:
            status = "✓ PASS" if check.passed else "✗ FAIL"
            print(f"  {status} {check.check_name} ({check.severity})")
            print(f"    {check.description}")
            if check.expected_value is not None and check.actual_value is not None:
                print(f"    Expected: {check.expected_value}, Actual: {check.actual_value}")
        
        print(f"\nMajor Discrepancies ({len(result.discrepancies)} found):")
        for disc in result.discrepancies:
            print(f"  - {disc.discrepancy_type} ({disc.resolution_difficulty})")
            print(f"    Nodes: {', '.join(disc.nodes_involved)}")
            print(f"    {disc.description}")
            print(f"    Potential causes: {', '.join(disc.potential_causes)}")
        
        print(f"\nMathematical Verification:")
        math_check = result.mathematical_verification
        print(f"  Total cube claims: {math_check.total_theoretical_claims}")
        print(f"  Visible cube claims: {math_check.visible_cube_claims}")
        print(f"  Missing cube claims: {math_check.missing_cube_claims}")
        print(f"  Cross-node consistency: {math_check.cross_node_consistency}")
        
        print(f"\nConfidence Analysis:")
        conf_analysis = result.confidence_analysis
        print(f"  Node confidences: {conf_analysis.node_confidences}")
        print(f"  Actual agreement level: {conf_analysis.agreement_level:.2f}")
        print(f"  Calibration: {conf_analysis.confidence_calibration}")
        print(f"  Recommendation: {conf_analysis.uncertainty_recommendation}")
        
        # Save results for Node 5
        with open("node4_output.json", "w") as f:
            json.dump(result.model_dump() if hasattr(result, 'model_dump') else result, f, indent=2)
            
        print("\nResults saved to node4_output.json")
        return result
        
    except Exception as e:
        print(f"Error during Node 4 execution: {str(e)}")
        return None

if __name__ == "__main__":
    import asyncio
    
    # Run Node 4 test
    result = asyncio.run(test_node4_cross_verification())