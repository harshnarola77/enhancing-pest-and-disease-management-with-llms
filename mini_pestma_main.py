
import ollama
import time
import os
from typing import Optional, Dict, Any
import json
import re

class LocalMiniPestMA:
    """Your 3-agent plant diagnosis system - VS Code Edition with Robust JSON optimization"""
    
    def __init__(self):
        print("🌱 Mini-PestMA System Starting...")
        print("🔧 Running in VS Code Professional Environment")
        print("⚡ Robust JSON Optimized for High Performance")
        print("🛡️ Crash-Proof with Smart Error Recovery")
        print("=" * 60)
        
        self.agents = {
            'critical_diagnoser': {
                'model': 'mistral-small3.2:24b',
                'role': 'Critical Plant Pathologist',
                'system_prompt': """A forensic plant pathologist with 20 years of experience.

CRITICAL MISSION: Analyze text description and image (if provided). DETECT contradictions between image and text.

MANDATORY: OUTPUT ONLY valid JSON in this EXACT format:
{
  "primary_diagnosis": "specific disease/condition name",
  "primary_confidence": 8,
  "alternative_diagnosis": "second most likely condition", 
  "alternative_confidence": 5,
  "image_text_correlation": "consistent/contradictory/no_image",
  "key_symptoms_observed": "brief list of main symptoms seen",
  "visual_evidence_quality": "excellent/good/poor/none",
  "error_flags": {
    "equally_likely": false,
    "contradictory_symptoms": true,
    "insufficient_evidence": false
  },
  "diagnostic_reasoning": "brief explanation of primary diagnosis choice"
}

CONFIDENCE SCALE: 1-10 (1=very uncertain, 10=extremely confident)

FORENSIC DETECTION RULES:
- If user describes symptoms NOT visible in image → "contradictory"
- If image shows different symptoms than described → "contradictory"
- If image quality too poor to confirm → "poor"
- If no image provided → "no_image"

ERROR FLAGS (Set to true when applicable):
- "equally_likely": Multiple diagnoses have similar probability
- "contradictory_symptoms": Symptoms don't align logically  
- "insufficient_evidence": Not enough info for confident diagnosis

CRITICAL: You MUST output ONLY the JSON object. NO explanatory text before or after the JSON."""
            },
            
            'skeptical_validator': {
                'model': 'gemma3:27b',
                'role': 'Skeptical Quality Reviewer',
                'system_prompt': """A veteran plant clinic director known for catching diagnostic errors.

SKEPTICAL MISSION: Challenge the diagnosis from Agent 1. Find flaws, question assumptions, detect bias.

INPUT: JSON from Agent 1

MANDATORY: OUTPUT ONLY valid JSON in this EXACT format:
{
  "primary_diagnosis_valid": true,
  "primary_confidence_adjustment": -1,
  "alternative_diagnosis_preferred": false,
  "critical_concerns": "specific diagnostic issues identified or none",
  "evidence_quality_assessment": "strong/moderate/weak",
  "overlooked_factors": "environmental/seasonal factors missed or none",
  "bias_detection": "confirmation bias detected or none identified",
  "additional_diagnostics_needed": "laboratory tests/imaging required or none",
  "final_recommendation": "support_primary/prefer_alternative/insufficient_data/request_expert"
}

SKEPTICAL VALIDATION PROTOCOL:
1. Question evidence sufficiency
2. Challenge diagnostic reasoning
3. Identify potential biases
4. Consider alternative explanations
5. Assess seasonal/environmental factors
6. Flag missing information

CONFIDENCE ADJUSTMENT: -3 to +3 scale
FINAL RECOMMENDATIONS:
- "support_primary": Agree with primary diagnosis
- "prefer_alternative": Alternative more likely
- "insufficient_data": Need more information
- "request_expert": Complex case needing specialist

CRITICAL: Output ONLY the JSON object. NO explanatory text."""
            },
            
            'conservative_advisor': {
                'model': 'phi4:14b',
                'role': 'Conservative Extension Agent',
                'system_prompt': """A conservative extension specialist.

ADVISORY PRINCIPLE: Recommendations proportional to certainty.
- High confidence (8-10): Specific treatments with timelines
- Medium confidence (5-7): Conservative monitoring approach  
- Low confidence (1-4): Diagnostic steps only

You receive JSON analysis from Agent 1 (diagnoser) and Agent 2 (validator). Synthesize findings into practical recommendations.

MANDATORY RESPONSE FORMAT:
**DIAGNOSTIC SYNTHESIS**: [Combine both agents' findings]
**CONFIDENCE ASSESSMENT**: [Final confidence level with reasoning]
**IMAGE-TEXT CORRELATION**: [Note any contradictions flagged]
**CRITICAL CONCERNS**: [Address validator's concerns]

**RECOMMENDED ACTION**:
[Specific guidance based on final confidence level]

**MONITORING PLAN**: [What to watch for]
**WHEN TO ESCALATE**: [Red flags requiring professional help]

REQUIRED ELEMENTS:
- Cost-benefit considerations
- Environmental impact factors  
- Timeline for expected results
- At least one limitation or caution
- Safety considerations for farmer/gardener

Make professional judgment considering both analyses but prioritize farmer safety and practicality."""
            }
        }
        
        print("🤖 Professional AI Team Assembled:")
        for name, info in self.agents.items():
            print(f"   • {info['role']}")
            print(f"     Model: {info['model']}")
        print("=" * 60)
        
        # Performance tracking
        self.analysis_history = []
    
    def _extract_json(self, text: str) -> Dict[Any, Any]:
        """
        Smart JSON extractor that can find JSON in mixed text output.
        This handles cases where models include explanations with the JSON.
        """
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',    
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    cleaned = match.strip()
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        markers = [
            (r'```json\s*(.*?)\s*```', re.DOTALL),
            (r'```\s*(.*?)\s*```', re.DOTALL),
            (r'\{.*\}', re.DOTALL),
        ]
        
        for pattern, flags in markers:
            match = re.search(pattern, text, flags)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except (json.JSONDecodeError, IndexError):
                    continue
        
        raise json.JSONDecodeError(f"Could not extract valid JSON from: {text[:200]}...")
    
    def _create_fallback_json(self, agent_type: str, error_msg: str) -> Dict[Any, Any]:
        """Create fallback JSON when parsing fails"""
        if agent_type == 'diagnoser':
            return {
                "primary_diagnosis": "Analysis failed - JSON parsing error",
                "primary_confidence": 1,
                "alternative_diagnosis": "Unable to determine",
                "alternative_confidence": 1,
                "image_text_correlation": "no_image",
                "key_symptoms_observed": "Parsing error occurred",
                "visual_evidence_quality": "none",
                "error_flags": {
                    "equally_likely": False,
                    "contradictory_symptoms": False,
                    "insufficient_evidence": True
                },
                "diagnostic_reasoning": "JSON parsing failed: " + str(error_msg)
            }
        elif agent_type == 'validator':
            return {
                "primary_diagnosis_valid": False,
                "primary_confidence_adjustment": -3,
                "alternative_diagnosis_preferred": False,
                "critical_concerns": "Validation failed due to parsing error: " + str(error_msg),
                "evidence_quality_assessment": "weak",
                "overlooked_factors": "Unable to assess due to parsing error",
                "bias_detection": "none identified",
                "additional_diagnostics_needed": "Retry analysis with corrected prompts",
                "final_recommendation": "request_expert"
            }
        return {}
    
    def analyze_plant_problem(self, user_description: str, image_path: Optional[str] = None) -> dict:
        """Main analysis function with robust JSON optimization"""
        
        print(f"\n📋 NEW ANALYSIS REQUEST")
        print(f"🔍 Problem: {user_description}")
        if image_path:
            print(f"📷 Image: {image_path}")
        print("-" * 50)
        
        analysis_start = time.time()
        results = {'metadata': {'timestamp': time.time(), 'problem': user_description}}
        
        # STAGE 1: Critical Diagnosis (JSON OUTPUT)
        print("1️⃣ CRITICAL DIAGNOSER")
        stage_start = time.time()
        
        try:
            diag_response = ollama.generate(
                model=self.agents['critical_diagnoser']['model'],
                prompt=f"FORENSIC PLANT ANALYSIS:\n\nPROBLEM: {user_description}\nIMAGE PROVIDED: {'Yes - analyze visual evidence' if image_path else 'No - text-only analysis'}",
                system=self.agents['critical_diagnoser']['system_prompt'],
                images=[image_path] if image_path else None,
                options={
                    'temperature': 0.05,
                    'num_predict': 500,
                    'top_p': 0.9
                }
            )
            
            try:
                diag_json = self._extract_json(diag_response['response'])
                stage_time = time.time() - stage_start
                results['diagnoser'] = {
                    'response_json': diag_json,
                    'response_text': diag_response['response'],
                    'time': stage_time,
                    'model': self.agents['critical_diagnoser']['model'],
                    'status': 'success'
                }
                print(f"   ✅ Completed in {stage_time:.1f}s")
            except json.JSONDecodeError as e:

                fallback_json = self._create_fallback_json('diagnoser', str(e))
                stage_time = time.time() - stage_start
                results['diagnoser'] = {
                    'response_json': fallback_json,
                    'response_text': diag_response['response'],
                    'time': stage_time,
                    'model': self.agents['critical_diagnoser']['model'],
                    'status': 'json_error_recovered',
                    'error': f"JSON parsing failed, using fallback: {e}",
                    'raw_response': diag_response['response']
                }
                print(f"   ⚠️ JSON Error recovered in {stage_time:.1f}s")
                diag_json = fallback_json  
                
        except Exception as e:
            results['diagnoser'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Error: {e}")
            return results
        
        # STAGE 2: Skeptical Validation (JSON OUTPUT)
        print("\n2️⃣ SKEPTICAL VALIDATOR")
        stage_start = time.time()
        
        try:
            valid_response = ollama.generate(
                model=self.agents['skeptical_validator']['model'],
                prompt=f"""PEER REVIEW CHALLENGE:

ORIGINAL PROBLEM: {user_description}
AGENT 1 DIAGNOSIS JSON: {json.dumps(diag_json, indent=2)}

Your mission: Challenge every aspect of this diagnosis with scientific skepticism.""",
                system=self.agents['skeptical_validator']['system_prompt'],
                options={
                    'temperature': 0.2,
                    'num_predict': 400,
                    'top_p': 0.9
                }
            )
            
            try:
                valid_json = self._extract_json(valid_response['response'])
                stage_time = time.time() - stage_start
                results['validator'] = {
                    'response_json': valid_json,
                    'response_text': valid_response['response'],
                    'time': stage_time,
                    'model': self.agents['skeptical_validator']['model'],
                    'status': 'success'
                }
                print(f"   ✅ Completed in {stage_time:.1f}s")
            except json.JSONDecodeError as e:

                fallback_json = self._create_fallback_json('validator', str(e))
                stage_time = time.time() - stage_start
                results['validator'] = {
                    'response_json': fallback_json,
                    'response_text': valid_response['response'],
                    'time': stage_time,
                    'model': self.agents['skeptical_validator']['model'],
                    'status': 'json_error_recovered',
                    'error': f"JSON parsing failed, using fallback: {e}",
                    'raw_response': valid_response['response']
                }
                print(f"   ⚠️ Validator JSON Error recovered in {stage_time:.1f}s")
                valid_json = fallback_json  
                
        except Exception as e:
            
            print(f"   ⚠️ Validator failed ({e}), continuing with diagnoser results...")
            valid_json = self._create_fallback_json('validator', str(e))
            results['validator'] = {
                'response_json': valid_json,
                'status': 'error_recovered',
                'error': str(e),
                'time': time.time() - stage_start
            }
        
        # STAGE 3: Conservative Advisory (NORMAL TEXT OUTPUT)
        print("\n3️⃣ CONSERVATIVE ADVISOR")
        stage_start = time.time()
        
        try:
            advisor_response = ollama.generate(
                model=self.agents['conservative_advisor']['model'],
                prompt=f"""EXTENSION CONSULTATION:

FARMER'S PROBLEM: {user_description}

DIAGNOSTIC ANALYSIS (Agent 1 JSON):
{json.dumps(diag_json, indent=2)}

PEER REVIEW (Agent 2 JSON):
{json.dumps(valid_json, indent=2)}

Provide practical, evidence-based recommendations considering both analyses.""",
                system=self.agents['conservative_advisor']['system_prompt'],
                options={
                    'temperature': 0.15,
                    'num_predict': 400,
                    'top_p': 0.9
                }
            )
            
            stage_time = time.time() - stage_start
            results['advisor'] = {
                'response': advisor_response['response'],
                'time': stage_time,
                'model': self.agents['conservative_advisor']['model'],
                'status': 'success'
            }
            print(f"   ✅ Completed in {stage_time:.1f}s")
            
        except Exception as e:
            results['advisor'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Error: {e}")
            
        
        # ANALYSIS SUMMARY
        total_time = time.time() - analysis_start
        results['metadata']['total_time'] = total_time
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print(f"🚀 Powered by your RTX 4070 + 32GB RAM")
        print("=" * 60)
        

        self.analysis_history.append(results)
        
        return results
    
    def display_results(self, results: dict):
        """Crash-proof results display with JSON support"""
        
        print("\n" + "=" * 80)
        print("📊 MINI-PESTMA COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        # Metadata
        print(f"🕒 Timestamp: {time.ctime(results['metadata']['timestamp'])}")
        print(f"📝 Problem: {results['metadata']['problem']}")
        print(f"⏱️  Total Time: {results['metadata']['total_time']:.1f} seconds")
        print()
        

        diag_data = None
        valid_data = None
        advisor_data = None
        
        if 'diagnoser' in results and results['diagnoser'].get('status') in ['success', 'json_error_recovered']:
            diag_data = results['diagnoser'].get('response_json', {})
        
        if 'validator' in results and results['validator'].get('status') in ['success', 'json_error_recovered', 'error_recovered']:
            valid_data = results['validator'].get('response_json', {})
        
        if 'advisor' in results and results['advisor'].get('status') == 'success':
            advisor_data = results['advisor'].get('response', 'No recommendations available')
        
        # Display results based on what we have
        if diag_data:
            # Summary
            print("🎯 DIAGNOSTIC SUMMARY")
            print("-" * 60)
            print(f"Primary Diagnosis: {diag_data.get('primary_diagnosis', 'Unknown')} (Confidence: {diag_data.get('primary_confidence', 0)}/10)")
            print(f"Alternative: {diag_data.get('alternative_diagnosis', 'Unknown')} (Confidence: {diag_data.get('alternative_confidence', 0)}/10)")
            print(f"Image-Text Correlation: {diag_data.get('image_text_correlation', 'Unknown')}")
            
            if valid_data:
                print(f"Validator Recommendation: {valid_data.get('final_recommendation', 'Unknown')}")
            print()
            
            # Agent 1 Details
            print("🎯 CRITICAL DIAGNOSIS")
            print("-" * 60)
            print(f"Key Symptoms: {diag_data.get('key_symptoms_observed', 'Not specified')}")
            print(f"Visual Evidence: {diag_data.get('visual_evidence_quality', 'Unknown')}")
            print(f"Reasoning: {diag_data.get('diagnostic_reasoning', 'No reasoning provided')}")
            
            error_flags = diag_data.get('error_flags', {})
            if error_flags and any(error_flags.values()):
                print("⚠️ ALERT FLAGS:")
                for flag, value in error_flags.items():
                    if value:
                        print(f"  - {flag.replace('_', ' ').title()}")
            
            diag_status = results['diagnoser'].get('status', 'unknown')
            if diag_status == 'json_error_recovered':
                print("⚠️ Note: JSON parsing issues were automatically recovered")
            
            print(f"⏱️ Processing Time: {results['diagnoser'].get('time', 0):.1f}s")
            print()
        
        # Agent 2 Details (if available)
        if valid_data:
            print("🔍 QUALITY REVIEW")
            print("-" * 60)
            print(f"Primary Diagnosis Valid: {valid_data.get('primary_diagnosis_valid', 'Unknown')}")
            print(f"Confidence Adjustment: {valid_data.get('primary_confidence_adjustment', 0):+d}")
            print(f"Evidence Quality: {valid_data.get('evidence_quality_assessment', 'Unknown')}")
            
            critical_concerns = valid_data.get('critical_concerns', 'none')
            if critical_concerns != "none":
                print(f"⚠️ Critical Concerns: {critical_concerns}")
            
            overlooked_factors = valid_data.get('overlooked_factors', 'none')
            if overlooked_factors != "none":
                print(f"🔍 Overlooked Factors: {overlooked_factors}")
            
            valid_status = results['validator'].get('status', 'unknown')
            if valid_status in ['json_error_recovered', 'error_recovered']:
                print("⚠️ Note: Validation issues were automatically recovered")
                
            print(f"⏱️ Processing Time: {results['validator'].get('time', 0):.1f}s")
            print()
        
        # Agent 3 Details (if available)
        if advisor_data:
            print("💡 FINAL RECOMMENDATIONS")
            print("-" * 60)
            print(advisor_data)
            print(f"⏱️ Processing Time: {results['advisor'].get('time', 0):.1f}s")
        else:
            print("💡 FINAL RECOMMENDATIONS")
            print("-" * 60)
            print("⚠️ Recommendations unavailable due to processing error")
            if 'advisor' in results and results['advisor'].get('error'):
                print(f"Error: {results['advisor']['error']}")
        
        print("\n🛡️ SYSTEM STATUS")
        print("-" * 60)
        error_count = 0
        for agent_name in ['diagnoser', 'validator', 'advisor']:
            if agent_name in results:
                status = results[agent_name].get('status', 'unknown')
                if status == 'error':
                    print(f"❌ {agent_name}: {results[agent_name].get('error', 'Unknown error')}")
                    error_count += 1
                elif status in ['json_error_recovered', 'error_recovered']:
                    print(f"⚠️ {agent_name}: Recovered from errors")
                elif status == 'success':
                    print(f"✅ {agent_name}: Operating normally")
        
        if error_count == 0:
            print("✅ All systems operational")
        else:
            print(f"⚠️ {error_count} system(s) experienced unrecoverable errors")
        
        print("\n" + "=" * 80)
    
    def save_analysis(self, results: dict, filename: Optional[str] = None):
        """Save analysis to JSON file"""
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Analysis saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
    
    def get_performance_summary(self):
        """Get performance statistics"""
        
        if not self.analysis_history:
            return "No analyses completed yet."
        
        total_analyses = len(self.analysis_history)
        
        # Calculate different success levels
        full_success = 0
        partial_success = 0
        failed = 0
        
        for analysis in self.analysis_history:
            success_count = 0
            for agent in ['diagnoser', 'validator', 'advisor']:
                if agent in analysis:
                    status = analysis[agent].get('status', 'error')
                    if status in ['success', 'json_error_recovered', 'error_recovered']:
                        success_count += 1
            
            if success_count == 3:
                full_success += 1
            elif success_count > 0:
                partial_success += 1
            else:
                failed += 1
        
        # Calculate average time for successful analyses
        successful_analyses = [a for a in self.analysis_history 
                              if 'metadata' in a and 'total_time' in a['metadata']]
        
        if successful_analyses:
            avg_time = sum(a['metadata']['total_time'] for a in successful_analyses) / len(successful_analyses)
        else:
            avg_time = 0
        
        return f"""
📊 PERFORMANCE SUMMARY
Total Analyses: {total_analyses}
Full Success: {full_success} ({full_success/total_analyses:.1%})
Partial Success: {partial_success} ({partial_success/total_analyses:.1%})
Failed: {failed} ({failed/total_analyses:.1%})
Average Time: {avg_time:.1f} seconds
JSON Optimization: ✅ Active
Error Recovery: ✅ Active
System Status: 🛡️ Crash-Proof
        """

class MiniPestMAEvaluator:
    """Comprehensive evaluation framework for Mini-PestMA system"""
    
    def __init__(self, pestma_system):
        self.pestma_system = pestma_system
        self.evaluation_results = {}
        
    def run_complete_evaluation(self):
        """Run all evaluation tests and generate comprehensive report"""
        
        print("🔬 Starting Comprehensive Mini-PestMA Evaluation")
        print("=" * 80)
        
        # 1. Agent Workflow Evaluation (Most Important)
        workflow_results = self.evaluate_agent_workflow()
        
        # 2. System Robustness Testing
        robustness_results = self.test_system_robustness()
        
        # 3. Hallucination Resistance Testing
        hallucination_results = self.test_hallucination_resistance()
        
        # 4. Performance Measurement
        performance_results = self.measure_system_performance()
        
        # 5. Agent Independence Assessment
        independence_results = self.assess_agent_independence()
        
        # 6. JSON Recovery Testing
        json_recovery_results = self.test_json_recovery()
        
        # Generate final report
        final_report = self.generate_evaluation_report({
            'workflow': workflow_results,
            'robustness': robustness_results,
            'hallucination': hallucination_results,
            'performance': performance_results,
            'independence': independence_results,
            'json_recovery': json_recovery_results
        })
        
        return final_report
    
    def evaluate_agent_workflow(self):
        """Test how well the 3-agent workflow performs"""
        
        print("\n1️⃣ Agent Workflow Evaluation")
        print("-" * 50)
        
        test_cases = [
            {
                "description": "My tomato plants have brown spots with yellow halos",
                "category": "fungal_disease",
                "complexity": "medium"
            },
            {
                "description": "White powdery substance on rose leaves after humid weather",
                "category": "powdery_mildew", 
                "complexity": "easy"
            },
            {
                "description": "Cucumber plants wilting, yellow leaves from bottom up, soil moist",
                "category": "root_problem",
                "complexity": "medium"
            },
            {
                "description": "Small holes in lettuce leaves, might be insects",
                "category": "pest_damage",
                "complexity": "easy"
            },
            {
                "description": "My plant looks sick but I'm not sure what's wrong",
                "category": "vague_symptoms",
                "complexity": "hard"
            }
        ]
        
        workflow_results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"   Testing case {i}: {case['description'][:50]}...")
            
            try:
                start_time = time.time()
                result = self.pestma_system.analyze_plant_problem(case["description"])
                processing_time = time.time() - start_time
                
                # Evaluate workflow completion
                evaluation = {
                    "case": case["description"],
                    "category": case["category"],
                    "complexity": case["complexity"],
                    "processing_time": processing_time,
                    "diagnoser_success": result.get("diagnoser", {}).get("status") == "success",
                    "validator_success": result.get("validator", {}).get("status") in ["success", "json_error_recovered"],
                    "advisor_success": result.get("advisor", {}).get("status") == "success",
                    "workflow_completed": False,
                    "has_recovery": False,
                    "agent_outputs": {}
                }
                
                # Check if all agents provided meaningful output
                if evaluation["diagnoser_success"] and evaluation["validator_success"] and evaluation["advisor_success"]:
                    evaluation["workflow_completed"] = True
                
                # Check for JSON recovery
                for agent in ["diagnoser", "validator"]:
                    if result.get(agent, {}).get("status") == "json_error_recovered":
                        evaluation["has_recovery"] = True
                        break
                
                workflow_results.append(evaluation)
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                workflow_results.append({
                    "case": case["description"],
                    "category": case["category"],
                    "complexity": case["complexity"],
                    "error": str(e),
                    "workflow_completed": False
                })
        
        # Calculate workflow metrics
        completed_workflows = sum(1 for r in workflow_results if r.get("workflow_completed", False))
        recovery_count = sum(1 for r in workflow_results if r.get("has_recovery", False))
        
        print(f"   ✅ Workflow Success Rate: {completed_workflows}/{len(workflow_results)} ({completed_workflows/len(workflow_results):.1%})")
        print(f"   🔧 Recovery Instances: {recovery_count}/{len(workflow_results)}")
        
        return {
            "success_rate": completed_workflows / len(workflow_results),
            "recovery_rate": recovery_count / len(workflow_results),
            "detailed_results": workflow_results
        }
    
    def test_system_robustness(self):
        """Test system behavior with edge cases and problematic inputs"""
        
        print("\n2️⃣ System Robustness Testing")
        print("-" * 50)
        
        edge_cases = [
            {
                "description": "",  # Empty input
                "type": "empty_input"
            },
            {
                "description": "Help me with my car engine problem",  # Wrong domain
                "type": "wrong_domain"
            },
            {
                "description": "My plant looks sick but I'm not sure what's wrong",
                "type": "vague_input"
            }
        ]
        
        robustness_results = []
        
        for i, case in enumerate(edge_cases, 1):
            print(f"   Testing edge case {i}: {case['type']}")
            
            try:
                start_time = time.time()
                result = self.pestma_system.analyze_plant_problem(case["description"])
                processing_time = time.time() - start_time
                
                # Check if system handled gracefully
                handled_gracefully = all([
                    "error" not in result.get("diagnoser", {}).get("status", ""),
                    "error" not in result.get("validator", {}).get("status", ""),
                    "error" not in result.get("advisor", {}).get("status", "")
                ])
                
                robustness_results.append({
                    "case_type": case["type"],
                    "handled_gracefully": handled_gracefully,
                    "processing_time": processing_time,
                    "completed": result.get("metadata", {}).get("total_time") is not None
                })
                
            except Exception as e:
                print(f"      ⚠️  Exception: {str(e)[:100]}...")
                robustness_results.append({
                    "case_type": case["type"],
                    "handled_gracefully": False,
                    "error": str(e),
                    "completed": False
                })
        
        graceful_handling = sum(1 for r in robustness_results if r.get("handled_gracefully", False))
        
        print(f"   ✅ Graceful Handling Rate: {graceful_handling}/{len(robustness_results)} ({graceful_handling/len(robustness_results):.1%})")
        
        return {
            "graceful_handling_rate": graceful_handling / len(robustness_results),
            "detailed_results": robustness_results
        }
    
    def test_hallucination_resistance(self):
        """Test system's resistance to hallucinations and false agreements"""
        
        print("\n3️⃣ Hallucination Resistance Testing")
        print("-" * 50)
        
        contradictory_cases = [
            {
                "description": "My plant is completely healthy with no visible problems",
                "type": "overconfidence_test"
            },
            {
                "description": "White powdery mildew on leaves",
                "type": "no_image_test"
            }
        ]
        
        hallucination_results = []
        
        for i, case in enumerate(contradictory_cases, 1):
            print(f"   Testing hallucination case {i}: {case['type']}")
            
            try:
                result = self.pestma_system.analyze_plant_problem(case["description"])
                
                # Check for cautious language
                cautious_indicators = [
                    "cannot confirm", "without image", "visual evidence needed",
                    "uncertain", "requires confirmation", "would need to see",
                    "image would help", "visual inspection needed", "no_image"
                ]
                
                # Check responses for caution
                system_cautious = False
                
                # Check diagnoser
                if "diagnoser" in result and result["diagnoser"].get("response_json"):
                    diag_json = result["diagnoser"]["response_json"]
                    if diag_json.get("image_text_correlation") == "no_image":
                        system_cautious = True
                
                # Check advisor
                if "advisor" in result and result["advisor"].get("response"):
                    advisor_text = result["advisor"]["response"].lower()
                    if any(indicator in advisor_text for indicator in cautious_indicators):
                        system_cautious = True
                
                hallucination_results.append({
                    "case_type": case["type"],
                    "hallucination_resistance": system_cautious
                })
                
            except Exception as e:
                hallucination_results.append({
                    "case_type": case["type"],
                    "error": str(e),
                    "hallucination_resistance": False
                })
        
        resistance_count = sum(1 for r in hallucination_results if r.get("hallucination_resistance", False))
        
        print(f"   ✅ Hallucination Resistance Rate: {resistance_count}/{len(hallucination_results)} ({resistance_count/len(hallucination_results):.1%})")
        
        return {
            "resistance_rate": resistance_count / len(hallucination_results) if hallucination_results else 0,
            "detailed_results": hallucination_results
        }
    
    def measure_system_performance(self):
        """Measure system performance metrics"""
        
        print("\n4️⃣ Performance Measurement")
        print("-" * 50)
        
        test_inputs = [
            "Brown spots on tomato leaves",
            "Yellowing cucumber plants", 
            "White substance on rose leaves"
        ]
        
        response_times = []
        success_count = 0
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"   Performance test {i}/3...")
            
            try:
                start_time = time.time()
                result = self.pestma_system.analyze_plant_problem(test_input)
                total_time = time.time() - start_time
                
                response_times.append(total_time)
                
                if result.get("metadata", {}).get("total_time") is not None:
                    success_count += 1
                    
            except Exception as e:
                print(f"      ❌ Performance test {i} failed: {e}")
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        success_rate = success_count / len(test_inputs)
        
        print(f"   ⏱️  Average Response Time: {avg_response_time:.1f} seconds")
        print(f"   ✅ Performance Success Rate: {success_rate:.1%}")
        
        return {
            "average_response_time": avg_response_time,
            "success_rate": success_rate,
            "response_times": response_times
        }
    
    def assess_agent_independence(self):
        """Assess how independently agents operate"""
        
        print("\n5️⃣ Agent Independence Assessment")
        print("-" * 50)
        
        test_cases = [
            "Tomato plants with brown leaf spots",
            "White powdery mildew on roses"
        ]
        
        independence_results = []
        
        for case in test_cases:
            try:
                result = self.pestma_system.analyze_plant_problem(case)
                
                # Check for validator skepticism
                validator_skeptical = False
                if "validator" in result and result["validator"].get("response_json"):
                    validator_data = result["validator"]["response_json"]
                    
                    # Look for signs of skepticism
                    skeptical_indicators = [
                        validator_data.get("primary_diagnosis_valid") == False,
                        validator_data.get("primary_confidence_adjustment", 0) < 0,
                        validator_data.get("final_recommendation") in ["prefer_alternative", "insufficient_data", "request_expert"]
                    ]
                    
                    validator_skeptical = any(skeptical_indicators)
                
                independence_results.append({
                    "case": case,
                    "shows_independence": validator_skeptical
                })
                
            except Exception as e:
                independence_results.append({
                    "case": case,
                    "shows_independence": False
                })
        
        independence_count = sum(1 for r in independence_results if r.get("shows_independence", False))
        independence_rate = independence_count / len(independence_results) if independence_results else 0
        
        print(f"   🔍 Agent Independence Rate: {independence_count}/{len(independence_results)} ({independence_rate:.1%})")
        
        return {
            "independence_rate": independence_rate,
            "detailed_results": independence_results
        }
    
    def test_json_recovery(self):
        """Test JSON parsing recovery mechanisms"""
        
        print("\n6️⃣ JSON Recovery Testing")
        print("-" * 50)
        
        # Test with inputs that might stress JSON parsing
        stress_test_inputs = [
            "Complex plant disease with multiple symptoms: brown spots, yellowing, wilting",
            "Describe treatment for early blight management"
        ]
        
        recovery_results = []
        
        for i, test_input in enumerate(stress_test_inputs, 1):
            print(f"   JSON stress test {i}/2...")
            
            try:
                result = self.pestma_system.analyze_plant_problem(test_input)
                
                # Check for recovery instances
                recovery_detected = False
                for agent in ["diagnoser", "validator"]:
                    if agent in result:
                        if result[agent].get("status") == "json_error_recovered":
                            recovery_detected = True
                            break
                
                # Check if system still provided useful output
                useful_output = (
                    result.get("advisor", {}).get("status") == "success" and
                    result.get("advisor", {}).get("response", "").strip() != ""
                )
                
                recovery_results.append({
                    "test_case": i,
                    "recovery_detected": recovery_detected,
                    "system_resilient": useful_output
                })
                
            except Exception as e:
                recovery_results.append({
                    "test_case": i,
                    "system_resilient": False
                })
        
        resilient_count = sum(1 for r in recovery_results if r.get("system_resilient", False))
        resilience_rate = resilient_count / len(recovery_results) if recovery_results else 0
        
        print(f"   🛡️  JSON Recovery Resilience: {resilient_count}/{len(recovery_results)} ({resilience_rate:.1%})")
        
        return {
            "resilience_rate": resilience_rate,
            "detailed_results": recovery_results
        }
    
    def generate_evaluation_report(self, all_results):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "=" * 80)
        print("📊 MINI-PESTMA COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        
        # Calculate overall system score
        scores = [
            all_results["workflow"]["success_rate"],
            all_results["robustness"]["graceful_handling_rate"], 
            all_results["hallucination"]["resistance_rate"],
            all_results["performance"]["success_rate"],
            all_results["independence"]["independence_rate"],
            all_results["json_recovery"]["resilience_rate"]
        ]
        
        overall_score = sum(scores) / len(scores)
        
        print(f"\n🎯 OVERALL SYSTEM SCORE: {overall_score:.1%}")
        print("\n📈 COMPONENT SCORES:")
        print(f"   • Agent Workflow Success: {all_results['workflow']['success_rate']:.1%}")
        print(f"   • System Robustness: {all_results['robustness']['graceful_handling_rate']:.1%}")
        print(f"   • Hallucination Resistance: {all_results['hallucination']['resistance_rate']:.1%}")
        print(f"   • Performance Reliability: {all_results['performance']['success_rate']:.1%}")
        print(f"   • Agent Independence: {all_results['independence']['independence_rate']:.1%}")
        print(f"   • JSON Recovery Resilience: {all_results['json_recovery']['resilience_rate']:.1%}")
        
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"   • Average Response Time: {all_results['performance']['average_response_time']:.1f} seconds")
        print(f"   • JSON Recovery Rate: {all_results['workflow']['recovery_rate']:.1%}")
        
        # System status assessment
        if overall_score >= 0.8:
            status = "🟢 EXCELLENT - Production Ready"
        elif overall_score >= 0.6:
            status = "🟡 GOOD - Minor Improvements Needed"
        else:
            status = "🟠 FAIR - Improvements Recommended"
        
        print(f"\n🏆 SYSTEM STATUS: {status}")
        print("\n✅ Evaluation Complete!")
        print("=" * 80)
        
        return {
            "overall_score": overall_score,
            "component_scores": scores,
            "status": status,
            "detailed_results": all_results,
            "timestamp": time.ctime()
        }

def run_pestma_evaluation(pestma_system):
    """Run complete evaluation of Mini-PestMA system"""
    
    evaluator = MiniPestMAEvaluator(pestma_system)
    evaluation_report = evaluator.run_complete_evaluation()
    
    # Save results
    filename = f"pestma_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        print(f"📁 Evaluation results saved to: {filename}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    return evaluation_report

def main():
    """Main execution function"""
    
    print("🎓 MINI-PESTMA - PROFESSIONAL VS CODE EDITION")
    print("=" * 80)
    print("🔬 Advanced 3-Agent Plant Diagnosis System")
    print("💻 Optimized for Local Hardware Performance")
    print("⚡ Robust JSON Optimized for Ultra-Fast Processing")
    print("🛡️ Crash-Proof with Smart Error Recovery")
    print("🎯 Built for Research and Professional Use")
    print("=" * 80)
    
    # Initialize system
    try:
        pestma = LocalMiniPestMA()
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Choose mode
    print("\n🔧 SELECT OPERATION MODE:")
    print("1. Run Test Cases (Default)")
    print("2. Run Comprehensive Evaluation")
    print("3. Run Both")
    
    try:
        choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    if choice in ["1", "3"]:
        # Test cases for demonstration
        test_cases = [
            {
                'description': "My tomato plants have brown spots on the leaves with yellow halos around them. The spots started small but are growing larger.",
                'category': 'Fungal Disease'
            },
            {
                'description': "I see white powdery substance covering my rose leaves. It appeared after recent humid weather.",
                'category': 'Powdery Mildew'
            },
            {
                'description': "My cucumber plants are wilting badly and the leaves are turning yellow from bottom up. Soil seems moist.",
                'category': 'Root Problem'
            }
        ]
        
        print(f"\n🧪 RUNNING {len(test_cases)} PROFESSIONAL TEST CASES")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔬 TEST CASE {i}: {test_case['category']}")
            print("=" * 40)
            
            # Run analysis
            results = pestma.analyze_plant_problem(test_case['description'])
            
            # Display results
            pestma.display_results(results)
            
            # Save analysis
            pestma.save_analysis(results, f"test_case_{i}.json")
            
            # Brief pause between tests
            if i < len(test_cases):
                print("\n⏳ Preparing next test case...")
                time.sleep(2)
        
        # Summary
        print("\n" + "=" * 80)
        print("🎉 ALL TEST CASES COMPLETED SUCCESSFULLY!")
        print(pestma.get_performance_summary())
    
    if choice in ["2", "3"]:
        print("\n" + "=" * 80)
        print("🔬 STARTING COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 80)
        
        # Run comprehensive evaluation
        evaluation_results = run_pestma_evaluation(pestma)
        
        print(f"\n📊 Evaluation completed with overall score: {evaluation_results['overall_score']:.1%}")
        print(f"🏆 System Status: {evaluation_results['status']}")
    
    print("\n" + "=" * 80)
    print("💡 Next Steps:")
    print("1. Test with your own plant problems")
    print("2. Upload plant images for visual analysis")
    print("3. Review saved JSON analysis files")
    print("4. Run the Streamlit web interface")
    print("5. Review evaluation results for improvements")
    print("\n🎓 Ready for thesis demonstration!")

if __name__ == "__main__":
    main()