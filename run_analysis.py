#!/usr/bin/env python3
"""
Federated Cube Analysis - Complete Pipeline Runner
Runs all 5 nodes in sequence for any cube puzzle image

Usage:
    python run_analysis.py <image_path>
    python run_analysis.py cube_puzzle.png
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import all nodes
from node1 import Node1StructureAnalyzer, FederationOrchestrator
from node2 import Node2SystematicCounter
from node3 import Node3VoidAnalyzer
from node4 import Node4CrossVerifier
from node5 import Node5ConsensusSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteFederationPipeline:
    """Complete pipeline orchestrator for federated cube analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize all nodes and orchestrator"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize all nodes
        self.nodes = {
            "node1": Node1StructureAnalyzer(self.api_key),
            "node2": Node2SystematicCounter(self.api_key),
            "node3": Node3VoidAnalyzer(self.api_key),
            "node4": Node4CrossVerifier(self.api_key),
            "node5": Node5ConsensusSynthesizer(self.api_key)
        }
        
        # Track results
        self.results = {}
        self.output_files = {
            "node1": "node1_output.json",
            "node2": "node2_output.json", 
            "node3": "node3_output.json",
            "node4": "node4_output.json",
            "node5": "node5_output.json",
            "federation": "federation_final_answer.json"
        }
    
    def save_node_result(self, node_id: str, result: Any) -> None:
        """Save node result to JSON file"""
        output_file = self.output_files[node_id]
        
        # Convert Pydantic model to dict if needed
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result
            
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved {node_id} results to {output_file}")
        self.results[node_id] = result_dict
    
    async def run_node1(self, image_path: str) -> Dict[str, Any]:
        """Run Node 1: Structure Analysis"""
        logger.info("🔍 Starting Node 1: Structure Analysis")
        
        result = await self.nodes["node1"].analyze_structure(image_path)
        self.save_node_result("node1", result)
        
        logger.info(f"Node 1 Complete - Dimensions: {result.base_cube_dimensions}, "
                   f"Total: {result.total_theoretical_cubes}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node1"]
    
    async def run_node2(self, image_path: str) -> Dict[str, Any]:
        """Run Node 2: Systematic Counting"""
        logger.info("📊 Starting Node 2: Systematic Counting")
        
        # Node 2 needs Node 1's output
        node1_data = self.results["node1"]
        result = await self.nodes["node2"].count_cubes_systematically(image_path, node1_data)
        self.save_node_result("node2", result)
        
        logger.info(f"Node 2 Complete - Visible Cubes: {result.total_visible_cubes}, "
                   f"Layers: {len(result.layer_by_layer_count)}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node2"]
    
    async def run_node3(self, image_path: str) -> Dict[str, Any]:
        """Run Node 3: Void Pattern Analysis"""
        logger.info("🕳️ Starting Node 3: Void Pattern Analysis")
        
        result = await self.nodes["node3"].analyze_void_pattern(image_path)
        self.save_node_result("node3", result)
        
        logger.info(f"Node 3 Complete - Void Locations: {len(result.void_locations)}, "
                   f"Constraints: {len(result.geometric_constraints)}, "
                   f"Confidence: {result.pattern_confidence:.2f}")
        return self.results["node3"]
    
    async def run_node4(self, image_path: str) -> Dict[str, Any]:
        """Run Node 4: Cross Verification"""
        logger.info("🔍 Starting Node 4: Cross Verification")
        # Node 4 doesn't need the image path - it analyzes existing outputs
        
        result = await self.nodes["node4"].verify_cross_consistency()
        self.save_node_result("node4", result)
        
        logger.info(f"Node 4 Complete - System Health: {result.overall_system_health}, "
                   f"Contradiction Severity: {result.contradiction_severity}, "
                   f"Verification Confidence: {result.confidence_in_verification:.2f}")
        return self.results["node4"]
    
    async def run_node5(self, image_path: str) -> Dict[str, Any]:
        """Run Node 5: Consensus Synthesis"""
        logger.info("🤝 Starting Node 5: Consensus Synthesis")
        # Node 5 doesn't need the image path - it synthesizes existing results
        
        result = await self.nodes["node5"].synthesize_final_answer()
        self.save_node_result("node5", result)
        
        # Save the final federation answer
        ra = result.responsible_answer
        final_answer = {
            "question": "How many cubes are missing to make a full cube?",
            "answer_type": ra.answer_type,
            "primary_answer": ra.primary_answer,
            "confidence_range": ra.confidence_range,
            "system_confidence": ra.system_confidence,
            "reasoning": ra.reasoning_transparency,
            "human_guidance_needed": ra.human_guidance_needed,
            "federation_performance": self._assess_federation_performance()
        }
        
        with open(self.output_files["federation"], 'w') as f:
            json.dump(final_answer, f, indent=2)
        
        logger.info(f"Node 5 Complete - Answer Type: {ra.answer_type}, "
                   f"System Confidence: {ra.system_confidence:.3f}, "
                   f"Human Needed: {ra.human_guidance_needed}")
        
        return final_answer
    
    def _assess_federation_performance(self) -> str:
        """Assess how well the federation performed"""
        if len(self.results) < 4:
            return "Incomplete federation execution"
        
        confidences = []
        if "node1" in self.results:
            confidences.append(self.results["node1"].get('confidence', 0.0))
        if "node2" in self.results:
            confidences.append(self.results["node2"].get('confidence', 0.0))
        if "node3" in self.results:
            confidences.append(self.results["node3"].get('pattern_confidence', 0.0))
        if "node4" in self.results:
            # prefer verification confidence, fallback to agreement level
            c4 = self.results["node4"].get('confidence_in_verification', None)
            if c4 is None:
                c4 = self.results["node4"].get('confidence_analysis', {}).get('agreement_level', 0.0)
            confidences.append(c4 or 0.0)
        
        if not confidences:
            return "No confidence data available"
        
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        if confidence_variance > 0.1:
            return "High disagreement detected - effective contradiction identification"
        elif avg_confidence > 0.8:
            return "High consensus - reliable analysis"
        else:
            return "Moderate consensus - some uncertainty present"
    
    async def run_complete_analysis(self, image_path: str) -> Dict[str, Any]:
        """Run the complete federated analysis pipeline"""
        
        # Validate image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"🚀 Starting Complete Federated Analysis for: {image_path}")
        logger.info("=" * 60)
        
        try:
            # Run all nodes in sequence
            await self.run_node1(image_path)
            await self.run_node2(image_path)
            await self.run_node3(image_path)
            await self.run_node4(image_path)
            final_result = await self.run_node5(image_path)
            
            logger.info("=" * 60)
            logger.info("🎉 FEDERATION ANALYSIS COMPLETE!")
            logger.info("=" * 60)
            
            # Print final summary
            self.print_final_summary(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def print_final_summary(self, final_result: Dict[str, Any]) -> None:
        """Print a formatted summary of results"""
        
        print("\n" + "=" * 60)
        print("📋 FEDERATED ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"🎯 Question: {final_result['question']}")
        print(f"📊 Answer Type: {final_result['answer_type']}")
        
        if final_result['primary_answer']:
            print(f"💡 Primary Answer: {final_result['primary_answer']} cubes")
        
        if final_result['confidence_range']:
            print(f"📈 Confidence Range: {final_result['confidence_range']}")
        
        print(f"🎗️  System Confidence: {final_result['system_confidence']:.1%}")
        print(f"👤 Human Guidance Needed: {'Yes' if final_result['human_guidance_needed'] else 'No'}")
        
        print(f"\n🔍 Reasoning:")
        print(f"   {final_result['reasoning']}")
        
        print(f"\n⚡ Federation Performance:")
        print(f"   {final_result['federation_performance']}")
        
        print("\n📁 Generated Files:")
        for node_id, filename in self.output_files.items():
            if Path(filename).exists():
                print(f"   ✅ {filename}")
        
        print("=" * 60)

def main():
    """Main entry point for command-line usage"""
    
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <image_path>")
        print("Example: python run_analysis.py cube_puzzle.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    try:
        # Create and run pipeline
        pipeline = CompleteFederationPipeline()
        result = asyncio.run(pipeline.run_complete_analysis(image_path))
        
        print(f"\n✅ Analysis complete! Check federation_final_answer.json for results.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()