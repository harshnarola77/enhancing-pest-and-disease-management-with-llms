# STREAMLIT WEB INTERFACE 
import streamlit as st
import ollama
import time
import json
from PIL import Image
import os
from datetime import datetime
import re
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Mini-PestMA Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #2E8B57;
    }
    .success-metric {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .warning-metric {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .error-recovery {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitMiniPestMA:
    """Web interface for Mini-PestMA with robust JSON parsing"""
    
    def __init__(self):
        self.agents = {
            'critical_diagnoser': {
                'model': 'mistral-small3.2:24b',
                'role': 'Critical Plant Pathologist',
                'icon': '🎯',
                'system_prompt': """You are a forensic plant pathologist with 20 years of experience.

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
                'icon': '🔍',
                'system_prompt': """You are a veteran plant clinic director known for catching diagnostic errors.

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
                'icon': '💡',
                'system_prompt': """You are a conservative extension specialist.

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
    
    def _extract_json(self, text: str) -> Dict[Any, Any]:
        """
        Smart JSON extractor that can find JSON in mixed text output.
        This handles cases where models include explanations with the JSON.
        """
        # First, try direct parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Look for JSON object patterns
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',    # More complex nesting
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON between markers
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
        
        # If all else fails, try to construct a minimal valid JSON
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
    
    def run_analysis(self, problem_description: str, image_path: str = None) -> dict:
        """Execute the 3-agent analysis with robust JSON optimization and progress tracking"""
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'problem': problem_description,
                'has_image': image_path is not None
            }
        }
        
        with st.status("🔬 Analysis in Progress...", expanded=True) as status:
            
            # Stage 1: Critical Diagnosis (JSON OUTPUT)
            st.write("1/3 - 🎯 Agent 1 (Critical Diagnoser) analyzing...")
            
            diag_start = time.time()
            try:
                diag_response = ollama.generate(
                    model=self.agents['critical_diagnoser']['model'],
                    prompt=f"""PROFESSIONAL FORENSIC PLANT ANALYSIS:

PROBLEM DESCRIPTION: {problem_description}
IMAGE PROVIDED: {'Yes - conduct visual analysis' if image_path else 'No - text-based analysis only'}

Conduct thorough forensic analysis following established protocols.""",
                    system=self.agents['critical_diagnoser']['system_prompt'],
                    images=[image_path] if image_path else None,
                    options={'temperature': 0.05, 'num_predict': 500}
                )
                
                # Robust JSON parsing
                try:
                    diag_json = self._extract_json(diag_response['response'])
                    results['diagnoser'] = {
                        'response_json': diag_json,
                        'response_text': diag_response['response'],
                        'time': time.time() - diag_start,
                        'status': 'success'
                    }
                except json.JSONDecodeError as e:
                    # Create fallback JSON
                    fallback_json = self._create_fallback_json('diagnoser', str(e))
                    results['diagnoser'] = {
                        'response_json': fallback_json,
                        'response_text': diag_response['response'],
                        'time': time.time() - diag_start,
                        'status': 'json_error_recovered',
                        'error': f"JSON parsing failed, using fallback: {e}",
                        'raw_response': diag_response['response']
                    }
                    diag_json = fallback_json  # Continue with fallback
                    st.warning("⚠️ Agent 1 JSON parsing recovered with fallback")
                    
            except Exception as e:
                results['diagnoser'] = {'status': 'error', 'error': str(e)}
                status.update(label="❌ Analysis failed at diagnosis stage!", state="error")
                return results
            
            # Stage 2: Skeptical Validation (JSON OUTPUT)
            st.write("2/3 - 🔍 Agent 2 (Skeptical Validator) reviewing...")
            
            valid_start = time.time()
            try:
                valid_response = ollama.generate(
                    model=self.agents['skeptical_validator']['model'],
                    prompt=f"""RIGOROUS PEER REVIEW REQUIRED:

ORIGINAL PROBLEM: {problem_description}
AGENT 1 DIAGNOSIS JSON: {json.dumps(diag_json, indent=2)}

Your mission: Challenge every aspect of this diagnosis with scientific skepticism.""",
                    system=self.agents['skeptical_validator']['system_prompt'],
                    options={'temperature': 0.2, 'num_predict': 400}
                )
                
                # Robust JSON parsing
                try:
                    valid_json = self._extract_json(valid_response['response'])
                    results['validator'] = {
                        'response_json': valid_json,
                        'response_text': valid_response['response'],
                        'time': time.time() - valid_start,
                        'status': 'success'
                    }
                except json.JSONDecodeError as e:
                    # Create fallback JSON but continue workflow
                    fallback_json = self._create_fallback_json('validator', str(e))
                    results['validator'] = {
                        'response_json': fallback_json,
                        'response_text': valid_response['response'],
                        'time': time.time() - valid_start,
                        'status': 'json_error_recovered',
                        'error': f"JSON parsing failed, using fallback: {e}",
                        'raw_response': valid_response['response']
                    }
                    valid_json = fallback_json  # Continue with fallback
                    st.warning("⚠️ Agent 2 JSON parsing recovered with fallback")
                    
            except Exception as e:
                # If validator fails completely, create minimal fallback and continue
                st.warning(f"⚠️ Validator failed ({e}), continuing with diagnoser results...")
                valid_json = self._create_fallback_json('validator', str(e))
                results['validator'] = {
                    'response_json': valid_json,
                    'status': 'error_recovered',
                    'error': str(e),
                    'time': time.time() - valid_start
                }
            
            # Stage 3: Conservative Advisory (NORMAL TEXT OUTPUT)
            st.write("3/3 - 💡 Agent 3 (Conservative Advisor) recommending...")
            
            advisor_start = time.time()
            try:
                advisor_response = ollama.generate(
                    model=self.agents['conservative_advisor']['model'],
                    prompt=f"""PROFESSIONAL EXTENSION CONSULTATION:

FARMER'S CONCERN: {problem_description}

DIAGNOSTIC ANALYSIS (Agent 1 JSON):
{json.dumps(diag_json, indent=2)}

PEER REVIEW (Agent 2 JSON):
{json.dumps(valid_json, indent=2)}

Provide evidence-based, cost-effective recommendations considering both analyses.""",
                    system=self.agents['conservative_advisor']['system_prompt'],
                    options={'temperature': 0.15, 'num_predict': 400}
                )
                
                results['advisor'] = {
                    'response': advisor_response['response'],
                    'time': time.time() - advisor_start,
                    'status': 'success'
                }
                
            except Exception as e:
                results['advisor'] = {'status': 'error', 'error': str(e)}
                st.error(f"❌ Advisory stage failed: {e}")
                # Don't return here - we can still show partial results
            
            # Calculate total time
            total_time = 0
            if 'diagnoser' in results and 'time' in results['diagnoser']:
                total_time += results['diagnoser']['time']
            if 'validator' in results and 'time' in results['validator']:
                total_time += results['validator']['time']
            if 'advisor' in results and 'time' in results['advisor']:
                total_time += results['advisor']['time']
            
            results['metadata']['total_time'] = total_time
            
            status.update(label="✅ Analysis Complete!", state="complete")
        
        return results

# Initialize system
if 'pestma_system' not in st.session_state:
    st.session_state.pestma_system = StreamlitMiniPestMA()
    st.session_state.analysis_count = 0
    st.session_state.analysis_history = []

# Main interface
st.markdown('<h1 class="main-header">🌱 Mini-PestMA</h1>', unsafe_allow_html=True)
st.markdown("**🛡️ Multi-Agent Plant Health Analysis System**")

# Professional sidebar
with st.sidebar:
    st.header("🏥 Agentic AI Team")
    
    for agent_key, agent_info in st.session_state.pestma_system.agents.items():
        st.markdown(f"""
        <div class="agent-card">
            <h4>{agent_info['icon']}</h4>
            <p><strong>{agent_info['role']}</strong></p>
            <p><code>{agent_info['model']}</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyses", st.session_state.analysis_count)
    with col2:
        if st.session_state.analysis_history:
            # Calculate average time safely
            valid_times = [a['metadata']['total_time'] for a in st.session_state.analysis_history if 'total_time' in a['metadata']]
            avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
            st.metric("Avg Time", f"{avg_time:.1f}s")
        else:
            st.metric("Avg Time", "N/A")
    
    if st.button("🔄 Reset Session"):
        st.session_state.analysis_count = 0
        st.session_state.analysis_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("📝 Plant Analysis")
    
    # Analysis form
    with st.form("professional_analysis", clear_on_submit=False):
        problem_description = st.text_area(
            "Detailed Problem Description:",
            placeholder="Provide comprehensive details: symptoms, affected plant parts, timeline, environmental conditions, previous treatments...",
            height=120
        )
        
        uploaded_image = st.file_uploader(
            "High-Resolution Plant Image (Optional):",
            type=['png', 'jpg', 'jpeg'],
            help="Upload clear, well-lit images for accurate visual analysis"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_button = st.form_submit_button("🔬 Run Analysis", type="primary")
        with col_b:
            priority = st.selectbox("Analysis Priority:", ["Standard", "High Priority", "Research"])
    
    # Execute analysis
    if analyze_button:
        if not problem_description.strip():
            st.error("⚠️ Please provide a detailed problem description for accurate analysis.")
        else:
            # Handle image upload
            image_path = None
            if uploaded_image:
                image = Image.open(uploaded_image)
                # Create temp directory if it doesn't exist
                os.makedirs("temp", exist_ok=True)
                image_path = f"temp/{uploaded_image.name}"
                image.save(image_path)
                
                with col2:
                    st.header("📷 Visual Evidence")
                    st.image(image, caption="Uploaded plant image for analysis")
            
            # Run the professional analysis
            results = st.session_state.pestma_system.run_analysis(problem_description, image_path)
            
            # Update session state
            st.session_state.analysis_count += 1
            st.session_state.analysis_history.append(results)
            
            # Safe data extraction
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
                st.success("✅ Analysis completed successfully!")
                
                # Executive Summary (Most Important)
                st.header("📋 Executive Summary & Recommendations")
                if advisor_data:
                    st.markdown(f"""
                    <div class="success-metric">
                    <h4>💡 Final Agentic Recommendations</h4>
                    {advisor_data}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-recovery">
                    <h4>⚠️ Recommendations Unavailable</h4>
                    Advisory agent encountered an error. Analysis continues with available data.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key Diagnostic Insights
                st.header("🎯 Key Diagnostic Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Primary Diagnosis")
                    confidence_delta = ""
                    if valid_data and 'primary_confidence_adjustment' in valid_data:
                        adj = valid_data['primary_confidence_adjustment']
                        confidence_delta = f"Validator: {adj:+d}" if adj != 0 else "Validated"
                    
                    st.metric(
                        label=diag_data.get('primary_diagnosis', 'Unknown'),
                        value=f"{diag_data.get('primary_confidence', 0)}/10",
                        delta=confidence_delta
                    )
                    
                    # Image correlation status
                    correlation = diag_data.get('image_text_correlation', 'unknown')
                    if correlation == "contradictory":
                        st.error(f"⚠️ Image-Text Correlation: {correlation.upper()}")
                    elif correlation == "consistent":
                        st.success(f"✅ Image-Text Correlation: {correlation}")
                    else:
                        st.info(f"📷 Image-Text Correlation: {correlation}")
                
                with col2:
                    st.subheader("Alternative Diagnosis")
                    alt_delta = "Secondary"
                    if valid_data and valid_data.get('alternative_diagnosis_preferred'):
                        alt_delta = "Alternative"
                    
                    st.metric(
                        label=diag_data.get('alternative_diagnosis', 'Unknown'),
                        value=f"{diag_data.get('alternative_confidence', 0)}/10",
                        delta=alt_delta
                    )
                    
                    # Validator recommendation
                    if valid_data and 'final_recommendation' in valid_data:
                        recommendation = valid_data['final_recommendation'].replace('_', ' ').title()
                        if valid_data['final_recommendation'] in ['prefer_alternative', 'insufficient_data', 'request_expert']:
                            st.warning(f"🔍 Validator: {recommendation}")
                        else:
                            st.success(f"✅ Validator: {recommendation}")
                    else:
                        st.info("🔍 Validator: Status unknown")
                
                # Error Flags and Concerns
                error_flags = diag_data.get('error_flags', {})
                has_concerns = valid_data and valid_data.get('critical_concerns', 'none') != 'none'
                
                if any(error_flags.values()) or has_concerns:
                    st.header("⚠️ Critical Alerts")
                    
                    # Agent 1 error flags
                    if error_flags.get('equally_likely'):
                        st.warning("🔄 Multiple diagnoses equally likely")
                    if error_flags.get('contradictory_symptoms'):
                        st.warning("⚠️ Contradictory symptoms detected")
                    if error_flags.get('insufficient_evidence'):
                        st.warning("📊 Insufficient evidence for confident diagnosis")
                    
                    # Agent 2 concerns
                    if has_concerns:
                        st.error(f"🚨 Validator Concerns: {valid_data['critical_concerns']}")
                
                # System Status
                st.header("🛡️ System Status")
                recovery_count = 0
                
                for agent_name in ['diagnoser', 'validator', 'advisor']:
                    if agent_name in results:
                        status = results[agent_name].get('status', 'unknown')
                        if status in ['json_error_recovered', 'error_recovered']:
                            recovery_count += 1
                
                if recovery_count > 0:
                    st.info(f"🔧 System recovered from {recovery_count} error(s) automatically")
                
                # Performance metrics
                total_time = results['metadata'].get('total_time', 0)
                st.success(f"⚡ Analysis completed in {total_time:.1f} seconds using your local hardware")
                
                # Detailed breakdown
                with st.expander("🔬 Detailed Analysis", expanded=False):
                    
                    # Agent 1 JSON breakdown
                    st.subheader("🎯 Critical Diagnosis")
                    
                    # Show recovery status
                    if results['diagnoser'].get('status') == 'json_error_recovered':
                        st.warning("⚠️ Note: JSON parsing issues were automatically recovered")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "Primary Diagnosis": diag_data.get('primary_diagnosis', 'Unknown'),
                            "Confidence": diag_data.get('primary_confidence', 0),
                            "Alternative": diag_data.get('alternative_diagnosis', 'Unknown'),
                            "Alt Confidence": diag_data.get('alternative_confidence', 0)
                        })
                    
                    with col2:
                        st.json({
                            "Image Correlation": diag_data.get('image_text_correlation', 'unknown'),
                            "Evidence Quality": diag_data.get('visual_evidence_quality', 'unknown'),
                            "Key Symptoms": diag_data.get('key_symptoms_observed', 'Not specified'),
                            "Error Flags": diag_data.get('error_flags', {})
                        })
                    
                    st.text_area("Diagnostic Reasoning:", diag_data.get('diagnostic_reasoning', 'No reasoning provided'), height=80, disabled=True)
                    st.caption(f"⏱️ Processing time: {results['diagnoser'].get('time', 0):.1f}s")
                    
                    # Agent 2 JSON breakdown
                    if valid_data:
                        st.subheader("🔍 Quality Review")
                        
                        # Show recovery status
                        if results['validator'].get('status') in ['json_error_recovered', 'error_recovered']:
                            st.warning("⚠️ Note: Validation issues were automatically recovered")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.json({
                                "Primary Valid": valid_data.get('primary_diagnosis_valid', 'Unknown'),
                                "Confidence Adj": valid_data.get('primary_confidence_adjustment', 0),
                                "Prefer Alternative": valid_data.get('alternative_diagnosis_preferred', False),
                                "Final Recommendation": valid_data.get('final_recommendation', 'unknown')
                            })
                        
                        with col2:
                            st.json({
                                "Evidence Quality": valid_data.get('evidence_quality_assessment', 'unknown'),
                                "Bias Detection": valid_data.get('bias_detection', 'unknown'),
                                "Additional Tests": valid_data.get('additional_diagnostics_needed', 'unknown'),
                                "Overlooked Factors": valid_data.get('overlooked_factors', 'unknown')
                            })
                        
                        if valid_data.get('critical_concerns', 'none') != 'none':
                            st.error(f"Critical Concerns: {valid_data['critical_concerns']}")
                        
                        st.caption(f"⏱️ Processing time: {results['validator'].get('time', 0):.1f}s")
                    else:
                        st.subheader("🔍 Quality Review")
                        st.error("⚠️ Validation data unavailable")
                
                # Save analysis option
                if st.button("💾 Save Analysis Report"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"analysis_report_{timestamp}.json"
                    try:
                        with open(filename, 'w') as f:
                            json.dump(results, f, indent=2)
                        st.success(f"Analysis saved as {filename}")
                    except Exception as e:
                        st.error(f"Failed to save analysis: {e}")
                
                # Clean up temp image
                if image_path and os.path.exists(image_path):
                   os.remove(image_path)
            
            else:
                st.error("❌ Analysis failed. Please check system status and try again.")
                
                # Enhanced error reporting
                with st.expander("Error Details"):
                    for agent_name, agent_result in results.items():
                        if agent_name != 'metadata' and agent_result.get('status') in ['error', 'json_error']:
                            if agent_result.get('status') == 'json_error':
                                st.error(f"🔧 {agent_name}: JSON parsing failed")
                                st.code(agent_result.get('raw_response', 'No response'), language='text')
                                st.info("💡 This might be resolved by adjusting the model temperature or trying again.")
                            else:
                                st.error(f"❌ {agent_name}: {agent_result.get('error', 'Unknown error')}")

# Analysis history
if st.session_state.analysis_history:
    st.header("📊 Analysis History")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(st.session_state.analysis_history))
    
    with col2:
        valid_times = [a['metadata']['total_time'] for a in st.session_state.analysis_history if 'total_time' in a['metadata']]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        st.metric("Average Time", f"{avg_time:.1f}s")
    
    with col3:
        # Critical analysis rate
        critical_count = 0
        for a in st.session_state.analysis_history:
            if 'validator' in a and a['validator'].get('status') in ['success', 'json_error_recovered']:
                validator_data = a['validator'].get('response_json', {})
                if validator_data.get('final_recommendation') in ['prefer_alternative', 'insufficient_data', 'request_expert']:
                    critical_count += 1
        
        critical_rate = critical_count / len(st.session_state.analysis_history) if st.session_state.analysis_history else 0
        st.metric("Critical Analysis Rate", f"{critical_rate:.1%}")
    
    with col4:
        image_count = sum(1 for a in st.session_state.analysis_history if a['metadata'].get('has_image', False))
        st.metric("With Images", f"{image_count}/{len(st.session_state.analysis_history)}")
    
    # Recovery statistics
    st.subheader("🛡️ System Resilience")
    recovery_stats = {"Full Success": 0, "Partial Recovery": 0, "Failed": 0}
    
    for analysis in st.session_state.analysis_history:
        success_count = 0
        recovery_count = 0
        
        for agent in ['diagnoser', 'validator', 'advisor']:
            if agent in analysis:
                status = analysis[agent].get('status', 'error')
                if status == 'success':
                    success_count += 1
                elif status in ['json_error_recovered', 'error_recovered']:
                    success_count += 1
                    recovery_count += 1
        
        if success_count == 3 and recovery_count == 0:
            recovery_stats["Full Success"] += 1
        elif success_count > 0:
            recovery_stats["Partial Recovery"] += 1
        else:
            recovery_stats["Failed"] += 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Full Success", recovery_stats["Full Success"], 
                 f"{recovery_stats['Full Success']/len(st.session_state.analysis_history):.1%}")
    with col2:
        st.metric("Recovered", recovery_stats["Partial Recovery"], 
                 f"{recovery_stats['Partial Recovery']/len(st.session_state.analysis_history):.1%}")
    with col3:
        st.metric("Failed", recovery_stats["Failed"], 
                 f"{recovery_stats['Failed']/len(st.session_state.analysis_history):.1%}")
    
    # Recent analysis details
    with st.expander("Recent Analysis Details", expanded=False):
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
            st.subheader(f"Analysis {len(st.session_state.analysis_history) - i}")
            st.text(f"Problem: {analysis['metadata']['problem'][:100]}...")
            st.text(f"Time: {analysis['metadata'].get('total_time', 0):.1f}s")
            
            # Agent status summary
            col1, col2, col3 = st.columns(3)
            
            agents = ['diagnoser', 'validator', 'advisor']
            agent_names = ['🎯 Diagnoser', '🔍 Validator', '💡 Advisor']
            
            for j, (agent, name) in enumerate(zip(agents, agent_names)):
                with [col1, col2, col3][j]:
                    if agent in analysis:
                        status = analysis[agent].get('status', 'unknown')
                        if status == 'success':
                            st.success(f"{name}: ✅")
                        elif status in ['json_error_recovered', 'error_recovered']:
                            st.warning(f"{name}: ⚠️ Recovered")
                        else:
                            st.error(f"{name}: ❌ Error")
                    else:
                        st.info(f"{name}: ❓ Missing")
            
            st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🌱 Mini-PestMA</h4>
    <p><strong>🛡️ Multi-Agent Plant Health Analysis System</strong></p>
    <p>Built in VS Code • Powered by Local AI • Research Grade Performance</p>
    <p>🔧 Features: Smart JSON Recovery • Error Resilience • Performance Optimization</p>
    <p>Running on: RTX 4070 + 32GB RAM • LLMs Used: mistral-small3.2:24b, gemma3:27b, phi4:14b</p>
</div>
""", unsafe_allow_html=True)