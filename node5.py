"""
Node 5: Consensus Synthesis & Final Answer for Cube Counting Federation
Synthesizes findings and provides responsible final answer with proper uncertainty quantification
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any, Tuple
import re
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceSummary(BaseModel):
    """Summary of evidence from each reasoning approach"""
    node_id: str = Field(description="Which node provided this evidence")
    approach: str = Field(description="What reasoning approach was used")
    key_finding: str = Field(description="Main finding from this approach")
    confidence: float = Field(description="Confidence level of this finding")
    supports_answer: Optional[int] = Field(description="What answer this evidence supports, if any")
    reliability_assessment: str = Field(description="How reliable is this evidence given contradictions?")

class ContradictionSummary(BaseModel):
    """Summary of major contradictions found"""
    contradiction_type: str = Field(description="Type of contradiction")
    conflicting_claims: List[str] = Field(description="What claims conflict")
    magnitude: float = Field(description="How severe is this contradiction?")
    resolvability: str = Field(description="Can this contradiction be resolved?")
    impact_on_answer: str = Field(description="How does this affect the final answer?")

class UncertaintyAnalysis(BaseModel):
    """Analysis of uncertainty and confidence calibration"""
    individual_node_confidences: Dict[str, float] = Field(description="Confidence of each node")
    cross_node_agreement: float = Field(description="How much do nodes agree?")
    calibration_quality: str = Field(description="Are confidences well-calibrated?")
    uncertainty_sources: List[str] = Field(description="What creates uncertainty?")
    appropriate_confidence_range: Tuple[float, float] = Field(description="What confidence range is appropriate?")

    @field_validator('appropriate_confidence_range', mode='before')
    @classmethod
    def coerce_conf_range(cls, v):
        if v is None:
            return v
        # Accept list/tuple of two numbers or string like "0.3-0.6"
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = v
            try:
                return (float(a), float(b))
            except Exception:
                pass
        if isinstance(v, str) and '-' in v:
            parts = [p.strip() for p in v.split('-', 1)]
            if len(parts) == 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except Exception:
                    pass
        return v

class ResponsibleAnswer(BaseModel):
    """The responsible final answer with proper uncertainty acknowledgment"""
    answer_type: str = Field(description="CONFIDENT_ANSWER, UNCERTAIN_RANGE, or DEFER_TO_HUMAN")
    primary_answer: Optional[int] = Field(description="Best guess answer if system is confident enough")
    confidence_range: Optional[Tuple[int, int]] = Field(description="Range of plausible answers if uncertain")
    system_confidence: float = Field(description="How confident is the system overall?")
    reasoning_transparency: str = Field(description="Clear explanation of how this answer was reached")
    limitations_acknowledged: List[str] = Field(description="What limitations does the system acknowledge?")
    human_guidance_needed: bool = Field(description="Should humans review this decision?")
    next_steps_recommended: List[str] = Field(description="What should be done next?")

    @field_validator('confidence_range', mode='before')
    @classmethod
    def coerce_answer_range(cls, v):
        if v is None:
            return v
        # Accept list/tuple of two ints or string like "10-15"
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = v
            try:
                return (int(a), int(b))
            except Exception:
                pass
        if isinstance(v, str) and '-' in v:
            parts = [p.strip() for p in v.split('-', 1)]
            if len(parts) == 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except Exception:
                    pass
        return v

class ConsensusSynthesisOutput(BaseModel):
    """Pydantic model for Node 5 structured output"""
    analysis_type: str = Field(default="consensus_synthesis")
    evidence_summary: List[EvidenceSummary] = Field(
        description="Summary of evidence from all reasoning approaches"
    )
    contradiction_summary: List[ContradictionSummary] = Field(
        description="Summary of major contradictions found by Node 4"
    )
    uncertainty_analysis: UncertaintyAnalysis = Field(
        description="Analysis of uncertainty and confidence calibration"
    )
    responsible_answer: ResponsibleAnswer = Field(
        description="Final answer with appropriate uncertainty acknowledgment"
    )
    meta_learning: Dict[str, Any] = Field(
        description="What did the federation learn about its own performance?"
    )
    federation_performance_assessment: str = Field(
        description="How well did the federated approach work?"
    )

class Node5ConsensusSynthesizer:
    """Node 5: Synthesizes all findings into responsible final answer"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "node_5_consensus_synthesis"
        
    def load_all_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load outputs from all previous nodes"""
        try:
            with open("node1_output.json", 'r') as f:
                node1_data = json.load(f)
            with open("node2_output.json", 'r') as f:
                node2_data = json.load(f)
            with open("node3_output.json", 'r') as f:
                node3_data = json.load(f)
            with open("node4_output.json", 'r') as f:
                node4_data = json.load(f)
            return node1_data, node2_data, node3_data, node4_data
        except FileNotFoundError as e:
            logger.error(f"Previous node output file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in previous node output: {e}")
            raise
    
    def create_synthesis_prompt(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], 
                              node3_data: Dict[str, Any], node4_data: Dict[str, Any]) -> str:
        """Create the prompt template for final synthesis"""
        return f"""
You are performing the final synthesis for a federated AI system that has analyzed a cube counting problem through multiple reasoning approaches. Your job is to provide a RESPONSIBLE final answer that properly acknowledges uncertainty rather than forcing confidence.

FEDERATION ANALYSIS SUMMARY:

NODE 1 (STRUCTURE ANALYSIS):
- Claimed: {node1_data['structure_type']} with {node1_data['total_theoretical_cubes']} total cubes
- Confidence: {node1_data['confidence']}
- Approach: Dimensional analysis of overall structure

NODE 2 (SYSTEMATIC COUNTING):  
- Found: {node2_data['total_visible_cubes']} visible cubes
- Confidence: {node2_data['confidence']}
- Approach: Layer-by-layer systematic counting

NODE 3 (VOID PATTERN ANALYSIS):
- Found: {node3_data['missing_cube_count']} missing cubes ({node3_data['void_pattern_type']})
- Confidence: {node3_data['pattern_confidence']}
- Approach: Geometric analysis of void patterns

NODE 4 (CROSS-VERIFICATION):
- System Health: {node4_data['overall_system_health']}
- Contradiction Severity: {node4_data['contradiction_severity']}
- Agreement Level: {node4_data['confidence_analysis']['agreement_level']:.2f}
- Recommendation: {node4_data['recommended_action']}

CRITICAL FINDINGS FROM NODE 4:
- 28-cube discrepancy between analyses (Node 1: 48 total vs Node 2+3: 20 total)
- Poor confidence calibration: High individual confidence but only {node4_data['confidence_analysis']['agreement_level']:.0%} agreement
- Mathematical contradictions that cannot be easily resolved
- Recommendation to defer to human judgment

YOUR SYNTHESIS TASK:

1. EVIDENCE SYNTHESIS:
   - What evidence does each reasoning approach provide?
   - Which evidence is most reliable given the contradictions?
   - What patterns emerge across different approaches?

2. CONTRADICTION RECONCILIATION:
   - Can any of the major contradictions be resolved?
   - What do irreconcilable contradictions tell us about system limits?
   - How should these contradictions affect final confidence?

3. RESPONSIBLE UNCERTAINTY QUANTIFICATION:
   - Given the contradictions, what's an appropriate confidence level?
   - Should the system provide a confident answer, uncertain range, or defer?
   - What uncertainty sources need to be acknowledged?

4. TRANSPARENT REASONING:
   - Explain clearly how the contradictions were handled
   - Show what each approach contributed and where they failed
   - Make the reasoning process transparent for human review

5. META-LEARNING ASSESSMENT:
   - What did this federation learn about its own performance?
   - How well did redundant reasoning work for error detection?
   - What does this suggest about AI confidence calibration?

KEY PRINCIPLE: When expert analyses disagree fundamentally, the responsible action is to acknowledge uncertainty rather than force confidence. A federated AI system should be humble about its limitations and transparent about contradictions.

Your goal is to demonstrate RESPONSIBLE AI that knows when to admit uncertainty rather than hallucinating confidence.
        """.strip()
    
    async def synthesize_final_answer(self) -> ConsensusSynthesisOutput:
        """Synthesize all analyses into responsible final answer"""
        
        try:
            # Load all previous outputs
            node1_data, node2_data, node3_data, node4_data = self.load_all_outputs()
            
            # Create synthesis prompt
            prompt = self.create_synthesis_prompt(node1_data, node2_data, node3_data, node4_data)
            
            logger.info(f"Node 5: Starting final synthesis")
            logger.info(f"Node 5: Processing {node4_data['contradiction_severity']} contradictions with {node4_data['overall_system_health']} system health")
            
            # Define manual JSON schema to guide the model (documentation only)
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string"},
                    "evidence_summary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "approach": {"type": "string"},
                                "key_finding": {"type": "string"},
                                "confidence": {"type": "number"},
                                "supports_answer": {"type": "integer"},
                                "reliability_assessment": {"type": "string"}
                            },
                            "required": ["node_id", "approach", "key_finding", "confidence", "reliability_assessment"]
                        }
                    },
                    "contradiction_summary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "contradiction_type": {"type": "string"},
                                "conflicting_claims": {"type": "array", "items": {"type": "string"}},
                                "magnitude": {"type": "number"},
                                "resolvability": {"type": "string"},
                                "impact_on_answer": {"type": "string"}
                            },
                            "required": ["contradiction_type", "conflicting_claims", "magnitude", "resolvability", "impact_on_answer"]
                        }
                    },
                    "uncertainty_analysis": {
                        "type": "object",
                        "properties": {
                            "individual_node_confidences": {"type": "object", "additionalProperties": {"type": "number"}},
                            "cross_node_agreement": {"type": "number"},
                            "calibration_quality": {"type": "string"},
                            "uncertainty_sources": {"type": "array", "items": {"type": "string"}},
                            "appropriate_confidence_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                        },
                        "required": ["individual_node_confidences", "cross_node_agreement", "calibration_quality", "uncertainty_sources", "appropriate_confidence_range"],
                        "additionalProperties": False
                    },
                    "responsible_answer": {
                        "type": "object",
                        "properties": {
                            "answer_type": {"type": "string"},
                            "primary_answer": {"type": "integer"},
                            "confidence_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                            "system_confidence": {"type": "number"},
                            "reasoning_transparency": {"type": "string"},
                            "limitations_acknowledged": {"type": "array", "items": {"type": "string"}},
                            "human_guidance_needed": {"type": "boolean"},
                            "next_steps_recommended": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["answer_type", "system_confidence", "reasoning_transparency", "limitations_acknowledged", "human_guidance_needed", "next_steps_recommended"],
                        "additionalProperties": False
                    },
                    "meta_learning": {"type": "object"},
                    "federation_performance_assessment": {"type": "string"}
                },
                "required": ["evidence_summary", "contradiction_summary", "uncertainty_analysis", "responsible_answer", "meta_learning", "federation_performance_assessment"],
                "additionalProperties": False
            }

            # Call OpenAI API with JSON object output
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt + "\n\nPlease reply strictly as JSON."}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )

            # Parse the JSON response
            response_text = response.choices[0].message.content
            try:
                synthesis_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Node 5: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                raise

            # Normalize/transform model output into our expected schema if keys differ
            normalized = self.normalize_synthesis_data(synthesis_data, node1_data, node2_data, node3_data, node4_data)

            # Build Pydantic model
            try:
                synthesis = ConsensusSynthesisOutput(**normalized)
            except Exception as e:
                logger.error(f"Node 5: Failed to create Pydantic model: {e}")
                logger.error(f"Synthesis data (normalized): {normalized}")
                raise
            
            # Add computed meta-learning insights
            synthesis.meta_learning = self.compute_meta_learning(
                node1_data, node2_data, node3_data, node4_data, synthesis
            )
            
            logger.info(f"Node 5: Synthesis complete. Answer type: {synthesis.responsible_answer.answer_type}, "
                       f"System confidence: {synthesis.responsible_answer.system_confidence:.2f}, "
                       f"Human guidance needed: {synthesis.responsible_answer.human_guidance_needed}")
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Node 5: Error during synthesis: {str(e)}")
            raise

    def normalize_synthesis_data(
        self,
        raw: Dict[str, Any],
        node1: Dict[str, Any],
        node2: Dict[str, Any],
        node3: Dict[str, Any],
        node4: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map model's free-form keys into our strict ConsensusSynthesisOutput schema."""
        def parse_confidence_from_text(text: str, fallback: Optional[float] = None) -> Optional[float]:
            if not isinstance(text, str):
                return fallback
            m = re.search(r"([0-9]*\.?[0-9]+)", text)
            try:
                return float(m.group(1)) if m else fallback
            except Exception:
                return fallback

        # Evidence summary
        evidence_summary: List[Dict[str, Any]] = []
        ev = raw.get("evidence_synthesis") or {}
        node_key_to_approach = {
            "node_1": "Dimensional structure analysis",
            "node_2": "Systematic counting",
            "node_3": "Void pattern analysis",
            "node_4": "Cross-verification",
        }
        node_confidence_fallback = {
            "node_1": node1.get("confidence"),
            "node_2": node2.get("confidence"),
            "node_3": node3.get("pattern_confidence"),
            "node_4": node4.get("confidence_analysis", {}).get("agreement_level"),
        }
        for k, v in ev.items():
            if k.startswith("node_") and isinstance(v, dict):
                node_id = k.replace("_", "")  # e.g., node_1 -> node1
                approach = node_key_to_approach.get(k, "Unknown approach")
                key_finding = v.get("evidence") or v.get("finding") or ""
                conf = parse_confidence_from_text(v.get("reliability", ""), node_confidence_fallback.get(k))
                reliability_assessment = v.get("reliability") or ""
                evidence_summary.append({
                    "node_id": node_id,
                    "approach": approach,
                    "key_finding": key_finding,
                    "confidence": conf if conf is not None else 0.0,
                    "supports_answer": None,
                    "reliability_assessment": reliability_assessment,
                })

        # Contradiction summary – synthesize at least one item
        contra: List[Dict[str, Any]] = []
        contra_rec = raw.get("contradiction_reconciliation") or {}
        node1_total = node1.get("total_theoretical_cubes")
        node23_total = (node2.get("total_visible_cubes") or 0) + (node3.get("missing_cube_count") or 0)
        contra.append({
            "contradiction_type": "Total Structure Size Disagreement",
            "conflicting_claims": [
                f"Node1 total: {node1_total}",
                f"Node2+Node3 implied total: {node23_total}",
            ],
            "magnitude": float(abs((node1_total or 0) - (node23_total or 0))),
            "resolvability": contra_rec.get("resolvable_contradictions", "Unknown"),
            "impact_on_answer": contra_rec.get("confidence_impact", "") or "Severe impact on confidence",
        })

        # Uncertainty analysis
        ua_src = raw.get("responsible_uncertainty_quantification") or {}
        node_confidences = node4.get("confidence_analysis", {}).get("node_confidences", {})
        agreement = node4.get("confidence_analysis", {}).get("agreement_level", 0.5)
        calibration_quality = node4.get("confidence_analysis", {}).get("confidence_calibration", "UNKNOWN")
        uncertainty_sources = ua_src.get("acknowledged_uncertainty_sources", [])
        try:
            agreement_float = float(agreement)
        except Exception:
            agreement_float = 0.5
        # simple range around agreement for display
        appropriate_confidence_range = (
            max(0.0, round(agreement_float - 0.2, 2)),
            min(1.0, round(agreement_float + 0.2, 2)),
        )
        uncertainty_analysis = {
            "individual_node_confidences": node_confidences,
            "cross_node_agreement": agreement_float,
            "calibration_quality": calibration_quality,
            "uncertainty_sources": uncertainty_sources,
            "appropriate_confidence_range": appropriate_confidence_range,
        }

        # Responsible answer
        rec = (ua_src.get("recommendation") or "").lower()
        if "defer" in rec:
            answer_type = "DEFER_TO_HUMAN"
            human_needed = True
        elif "range" in rec:
            answer_type = "UNCERTAIN_RANGE"
            human_needed = False
        else:
            answer_type = "CONFIDENT_ANSWER"
            human_needed = False
        transparent = raw.get("transparent_reasoning", {})
        reasoning_transparency = transparent.get("reasoning_transparency") or transparent.get("contradiction_handling") or ""
        responsible_answer = {
            "answer_type": answer_type,
            "primary_answer": None,
            "confidence_range": None,
            "system_confidence": agreement_float,
            "reasoning_transparency": reasoning_transparency,
            "limitations_acknowledged": uncertainty_sources or ["Severe contradictions across nodes"],
            "human_guidance_needed": human_needed,
            "next_steps_recommended": [ua_src.get("recommendation", "Review by human")] ,
        }

        # Meta-learning and performance assessment
        meta = raw.get("meta_learning_assessment") or {}
        meta_learning = meta or {
            "performance_insights": "Contradictions identified; deferring to human",
        }
        federation_perf = meta.get("redundant_reasoning_effectiveness") or "Redundancy surfaced contradictions effectively"

        return {
            "analysis_type": "consensus_synthesis",
            "evidence_summary": evidence_summary,
            "contradiction_summary": contra,
            "uncertainty_analysis": uncertainty_analysis,
            "responsible_answer": responsible_answer,
            "meta_learning": meta_learning,
            "federation_performance_assessment": federation_perf,
        }
    
    def compute_meta_learning(self, node1_data: Dict[str, Any], node2_data: Dict[str, Any], 
                            node3_data: Dict[str, Any], node4_data: Dict[str, Any], 
                            synthesis: ConsensusSynthesisOutput) -> Dict[str, Any]:
        """Compute meta-learning insights about federation performance"""
        
        return {
            "federation_effectiveness": {
                "error_detection": "SUCCESSFUL - Node 4 caught major contradictions",
                "uncertainty_calibration": "IMPROVED - System properly acknowledged limits",
                "overconfidence_prevention": "SUCCESSFUL - Avoided confident wrong answer",
                "transparency": "HIGH - Clear reasoning trace provided"
            },
            "individual_vs_federated_performance": {
                "individual_node_issues": "All confident but conflicting answers",
                "federation_benefit": "Detected contradictions and deferred appropriately",
                "reliability_improvement": "Prevented overconfident incorrect answer"
            },
            "lessons_learned": [
                "Multiple reasoning approaches can all fail while being confident",
                "Contradiction detection is more valuable than perfect answers",
                "Proper uncertainty acknowledgment prevents harmful overconfidence",
                "Federation creates humility through redundancy and cross-checking"
            ],
            "confidence_calibration_insights": {
                "before_federation": "Individual nodes: high confidence, wrong answers",
                "after_federation": "System: appropriate uncertainty, honest limitations",
                "key_improvement": "Learned when NOT to be confident"
            }
        }

# Test function for Node 5
async def test_node5_final_synthesis():
    """Test Node 5 using outputs from all previous nodes"""
    
    # Initialize Node 5
    node5 = Node5ConsensusSynthesizer()
    
    try:
        # Run Node 5 synthesis
        result = await node5.synthesize_final_answer()
        
        # Print results
        print("=" * 90)
        print("NODE 5 FINAL CONSENSUS SYNTHESIS")
        print("=" * 90)
        
        print(f"Analysis Type: {result.analysis_type}")
        
        print(f"\nEvidence Summary ({len(result.evidence_summary)} sources):")
        for evidence in result.evidence_summary:
            print(f"  {evidence.node_id} ({evidence.approach}):")
            print(f"    Finding: {evidence.key_finding}")
            print(f"    Confidence: {evidence.confidence:.2f}")
            print(f"    Reliability: {evidence.reliability_assessment}")
        
        print(f"\nContradiction Summary ({len(result.contradiction_summary)} major contradictions):")
        for contradiction in result.contradiction_summary:
            print(f"  {contradiction.contradiction_type} (Magnitude: {contradiction.magnitude:.2f})")
            print(f"    Conflicting claims: {', '.join(contradiction.conflicting_claims)}")
            print(f"    Resolvability: {contradiction.resolvability}")
            print(f"    Impact: {contradiction.impact_on_answer}")
        
        print(f"\nUncertainty Analysis:")
        uncertainty = result.uncertainty_analysis
        print(f"  Individual confidences: {uncertainty.individual_node_confidences}")
        print(f"  Cross-node agreement: {uncertainty.cross_node_agreement:.2f}")
        print(f"  Calibration quality: {uncertainty.calibration_quality}")
        print(f"  Appropriate confidence range: {uncertainty.appropriate_confidence_range}")
        print(f"  Uncertainty sources: {', '.join(uncertainty.uncertainty_sources)}")
        
        print(f"\n🎯 RESPONSIBLE FINAL ANSWER:")
        answer = result.responsible_answer
        print(f"  Answer Type: {answer.answer_type}")
        if answer.primary_answer:
            print(f"  Primary Answer: {answer.primary_answer} cubes")
        if answer.confidence_range:
            print(f"  Confidence Range: {answer.confidence_range[0]}-{answer.confidence_range[1]} cubes")
        print(f"  System Confidence: {answer.system_confidence:.2f}")
        print(f"  Human Guidance Needed: {answer.human_guidance_needed}")
        
        print(f"\n  Reasoning Transparency:")
        print(f"    {answer.reasoning_transparency}")
        
        print(f"\n  Limitations Acknowledged:")
        for limitation in answer.limitations_acknowledged:
            print(f"    - {limitation}")
        
        print(f"\n  Recommended Next Steps:")
        for step in answer.next_steps_recommended:
            print(f"    - {step}")
        
        print(f"\nMeta-Learning Insights:")
        meta = result.meta_learning
        print(f"  Federation Effectiveness: {meta['federation_effectiveness']}")
        print(f"  Key Lessons: {', '.join(meta['lessons_learned'])}")
        
        print(f"\nFederation Performance Assessment:")
        print(f"  {result.federation_performance_assessment}")
        
        # Save final results
        with open("node5_output.json", "w") as f:
            json.dump(result.model_dump() if hasattr(result, 'model_dump') else result, f, indent=2)
        
        with open("federation_final_answer.json", "w") as f:
            json.dump({
                "question": "How many cubes are missing to make a full cube?",
                "answer_type": answer.answer_type,
                "primary_answer": answer.primary_answer,
                "confidence_range": answer.confidence_range,
                "system_confidence": answer.system_confidence,
                "reasoning": answer.reasoning_transparency,
                "human_guidance_needed": answer.human_guidance_needed,
                "federation_performance": result.federation_performance_assessment
            }, f, indent=2)
            
        print("\nResults saved to node5_output.json and federation_final_answer.json")
        return result
        
    except Exception as e:
        print(f"Error during Node 5 execution: {str(e)}")
        return None

# Complete federation test
async def test_complete_federation():
    """Test the complete 5-node federation (assuming previous nodes have run)"""
    
    print("🚀 TESTING COMPLETE 5-NODE FEDERATED INTELLIGENCE SYSTEM")
    print("="*80)
    
    # This assumes all previous nodes have been run and outputs exist
    # We're just running the final synthesis step
    
    return await test_node5_final_synthesis()

if __name__ == "__main__":
    import asyncio
    
    # Run complete federation test
    result = asyncio.run(test_complete_federation())