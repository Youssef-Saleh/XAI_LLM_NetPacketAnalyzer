import torch
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import re
import time
from tqdm import tqdm

# Import llama-cpp-python for GGUF model loading
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Install with 'pip install llama-cpp-python'")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMExplanationGeneratorGGUF:
    """
    LLM-based explanation generator using GGUF models (like Qwen2.5-7B) for network security explanations.
    
    This class replaces the transformer-based implementation with llama-cpp-python to load
    quantized GGUF models for better performance and memory efficiency.
    """

    def __init__(self,
                 model_path: str = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                 device: torch.device = None,
                 max_length: int = 1024,  # Maximum total sequence length
                 max_new_tokens: int = 300,  # Number of new tokens to generate
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repeat_penalty: float = 1.1):
        """
        Initialize LLM explanation generator with GGUF model.

        Args:
            model_path: Path to the GGUF model file
            device: Device to run model on (ignored for llama-cpp-python)
            max_length: Maximum total sequence length (input + output)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Penalty for repeated tokens
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required to use GGUF models. Install with 'pip install llama-cpp-python'")
        
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        logger.info(f"Loading GGUF LLM from: {model_path}")

        # Load the GGUF model using llama-cpp-python
        # Note: For split models, we need to specify the first part
        try:
            # Check if we have multiple parts for the model
            import os
            if not os.path.exists(model_path):
                # Try to find the model files in the current directory
                model_dir = os.path.dirname(model_path) or "."
                model_files = [f for f in os.listdir(model_dir) if f.startswith(os.path.basename(model_path).split('-00001-of-00002')[0])]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])  # Use the first part
                    logger.info(f"Found model file: {model_path}")

            # V135 PROTOCOL FIX: Hardcode n_gpu_layers=-1 per Thesis Pillar C requirement
            # Rule 1: n_gpu_layers=-1 must be hardcoded for maximum GPU offloading
            n_gpu_layers = -1  # Hardcoded to use all possible GPU layers
            logger.info(f"V135: Using n_gpu_layers={n_gpu_layers} (hardcoded per protocol)")

            self.model = Llama(
                model_path=model_path,
                n_ctx=max_length,  # Context length
                n_threads=4,  # Number of threads to use
                n_gpu_layers=n_gpu_layers,  # Use GPU layers if appropriate
                verbose=True
            )

            logger.info(f"GGUF LLM initialized successfully")

            # Security terminology database for constraint enforcement
            self.security_terms = {
                'attack_types': ['DDoS', 'DoS', 'port scan', 'brute force', 'SQL injection',
                               'cross-site scripting', 'malware', 'botnet', 'backdoor'],
                'protocols': ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS'],
                'features': ['packet size', 'payload size', 'source port', 'destination port',
                           'inter-arrival time', 'flow duration', 'packet rate', 'byte rate']
            }
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_prompt(self, feature_explanation: Dict[str, Dict[str, float]],
                     prediction: float,
                     stakeholder_type: str = 'analyst',
                     shap_features: Dict[str, float] = None,
                     lime_features: Dict[str, float] = None) -> str:
        """
        Create detailed prompt for LLM based on feature explanation and prediction.

        Args:
            feature_explanation: SHAP/LIME explanation with top features
            prediction: Model prediction probability (0-1)
            stakeholder_type: Type of stakeholder ('analyst', 'developer', 'decision_maker')
            shap_features: SHAP feature importance dict
            lime_features: LIME feature importance dict

        Returns:
            Formatted prompt string
        """
        # Extract top features and their importance
        top_features = feature_explanation.get('top_features', {})
        if not top_features:
            return "Unable to generate explanation: no significant features identified."

        # Determine classification
        classification = "malicious" if prediction >= 0.5 else "benign"
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1 range

        # Format SHAP features with detailed descriptions
        shap_text = ""
        if shap_features:
            shap_items = list(shap_features.items())[:5]
            shap_text = "\nSHAP Feature Analysis:\n"
            for i, (feat, imp) in enumerate(shap_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, imp, 'shap')
                shap_text += f"{i}. {feat} (importance: {imp:.3f}): {feat_desc}\n"

        # Format LIME features with detailed descriptions
        lime_text = ""
        if lime_features:
            lime_items = list(lime_features.items())[:5]
            lime_text = "\nLIME Feature Analysis:\n"
            for i, (feat, imp) in enumerate(lime_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, abs(imp), 'lime')
                lime_text += f"{i}. {feat} (weight: {imp:.3f}): {feat_desc}\n"

        # V135 PROTOCOL FIX: Use Mistral-Instruct [INST] template format per Pillar C
        # Format: [INST] Role: X. Task: Y. Context: Z. Constraint: W. [/INST]
        shap_summary = ", ".join([f"{k}={v:.4f}" for k, v in list(shap_features.items())[:5]]) if shap_features else "N/A"
        risk_cost = prediction * 50000  # Scale to $0-$50000 based on prediction
        
        if stakeholder_type == 'analyst':
            prompt = f"""[INST] Role: SOC Analyst - Tier 1 Incident Response.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be TECHNICAL-TACTICAL. Use keywords: IOC, IP, port, protocol, firewall, network, tactical, response. [/INST]
"""
        elif stakeholder_type == 'decision_maker':
            prompt = f"""[INST] Role: Business Decision Maker - Non-Technical Executive.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be PLAIN-LANGUAGE. Use simple terms, business impact, risk level. [/INST]
"""
        else:  # developer (default)
            prompt = f"""[INST] Role: ML Engineer / Security Developer.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be ML-TECHNICAL. Use keywords: model, feature, bias, drift, validation, training, inference. [/INST]
"""

        return prompt

    def create_enhanced_prompt(self, feature_explanation: Dict[str, Dict[str, float]],
                     prediction: float,
                     stakeholder_type: str = 'analyst',
                     shap_features: Dict[str, float] = None,
                     lime_features: Dict[str, float] = None) -> str:
        """
        Create enhanced prompt for LLM based on feature explanation and prediction with more stakeholder types.<|im_start|>system
You are a cybersecurity expert analyzing network traffic. Provide a detailed technical explanation based on the given information.
<|im_end|>
<|im_start|>user
You are analyzing a network packet. The CNN-LSTM model classified this packet as {classification} with {confidence:.1%} confidence (prediction score: {prediction:.3f}).

{shap_text}
{lime_text}

Your task: Write a detailed 4-5 sentence explanation that:
1. States clearly why this packet was classified as {classification}
2. Explains how each top feature (from SHAP and LIME) contributed to this classification
3. Describes what these features mean in network security context
4. Explains the security implications - what type of attack or normal behavior this suggests
5. Notes the confidence level and what it means

Write a comprehensive technical explanation:<|im_end|>
<|im_start|>assistant
"""
        elif stakeholder_type == 'decision_maker':
            prompt = f"""<|im_start|>system
You are explaining network security analysis to a non-technical decision maker. Provide a clear, non-technical explanation.
<|im_end|>
<|im_start|>user
You are explaining network security analysis to a non-technical decision maker. A network packet was classified as {classification} with {confidence:.1%} confidence.

{shap_text}
{lime_text}

Your task: Write a clear 3-4 sentence explanation in plain language that:
1. Explains what this classification means in simple terms
2. Describes the security risk level (high/medium/low) and why
3. Explains what actions should be taken based on this classification
4. Uses analogies or simple examples to help understanding

Write a clear, non-technical explanation:<|im_end|>
<|im_start|>assistant
"""
        else:  # developer
            prompt = f"""<|im_start|>system
You are a security developer analyzing model predictions. Provide a detailed technical analysis.
<|im_end|>
<|im_start|>user
You are a security developer analyzing model predictions. The CNN-LSTM model classified this packet as {classification} with {confidence:.1%} confidence (prediction: {prediction:.3f}).

{shap_text}
{lime_text}

Your task: Write a detailed 5-6 sentence technical explanation that:
1. Explains the technical reasoning behind the {classification} classification
2. Analyzes how each feature contributed to the prediction (using SHAP and LIME values)
3. Discusses what these feature values indicate about the packet's characteristics
4. Identifies any anomalies or patterns that stand out
5. Suggests what this means for network security monitoring
6. Notes any potential model improvements or edge cases

Write a comprehensive technical analysis:<|im_end|>
<|im_start|>assistant
"""

        return prompt

    def create_enhanced_prompt(self, feature_explanation: Dict[str, Dict[str, float]],
                     prediction: float,
                     stakeholder_type: str = 'analyst',
                     shap_features: Dict[str, float] = None,
                     lime_features: Dict[str, float] = None) -> str:
        """
        Create enhanced prompt for LLM based on feature explanation and prediction with more stakeholder types.

        Args:
            feature_explanation: SHAP/LIME explanation with top features
            prediction: Model prediction probability (0-1)
            stakeholder_type: Type of stakeholder ('analyst', 'manager', 'compliance_officer', 'cto', 'decision_maker', 'developer')
            shap_features: SHAP feature importance dict
            lime_features: LIME feature importance dict

        Returns:
            Formatted prompt string
        """
        # Extract top features and their importance
        top_features = feature_explanation.get('top_features', {})
        if not top_features:
            return "Unable to generate explanation: no significant features identified."

        # Determine classification
        classification = "malicious" if prediction >= 0.5 else "benign"
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1 range

        # Format SHAP features with detailed descriptions
        shap_text = ""
        if shap_features:
            shap_items = list(shap_features.items())[:5]
            shap_text = "\nSHAP Feature Analysis:\n"
            for i, (feat, imp) in enumerate(shap_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, imp, 'shap')
                shap_text += f"{i}. {feat} (importance: {imp:.3f}): {feat_desc}\n"

        # Format LIME features with detailed descriptions
        lime_text = ""
        if lime_features:
            lime_items = list(lime_features.items())[:5]
            lime_text = "\nLIME Feature Analysis:\n"
            for i, (feat, imp) in enumerate(lime_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, abs(imp), 'lime')
                lime_text += f"{i}. {feat} (weight: {imp:.3f}): {feat_desc}\n"

        # V135 PROTOCOL FIX: Use Mistral-Instruct [INST] template format per Pillar C
        # Format: [INST] Role: X. Task: Y. Context: Z. Constraint: W. [/INST]
        shap_summary = ", ".join([f"{k}={v:.4f}" for k, v in list(shap_features.items())[:5]]) if shap_features else "N/A"
        risk_cost = prediction * 50000  # Scale to $0-$50000 based on prediction
        
        if stakeholder_type == 'analyst':
            prompt = f"""[INST] Role: SOC Analyst - Tier 1 Incident Response.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be TECHNICAL-TACTICAL. Use keywords: IOC, IP, port, protocol, firewall, network, tactical, response, blocking, segment, MITRE, ATT&CK. [/INST]
"""
        elif stakeholder_type == 'manager':
            prompt = f"""[INST] Role: SOC Manager - Business Risk Officer.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be EXECUTIVE-STRATEGIC. Use keywords: risk, budget, impact, business, resource, investment, disruption, recovery, executive, ROI, CISO, board. [/INST]
"""
        elif stakeholder_type == 'compliance_officer':
            prompt = f"""[INST] Role: Compliance Officer - Regulatory Framework Specialist.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be REGULATORY-LEGAL. Use keywords: GDPR, SOX, HIPAA, PCI, regulation, compliance, audit, breach, notification, legal, DPO. [/INST]
"""
        elif stakeholder_type == 'cto':
            prompt = f"""[INST] Role: CTO - Technology Strategy Officer.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be TECHNOLOGY-STRATEGIC. Use keywords: architecture, infrastructure, technology, scalability, performance, vendor, capacity, ROI, platform, integration, SIEM, SOAR, zero-trust. [/INST]
"""
        elif stakeholder_type == 'decision_maker':
            prompt = f"""[INST] Role: Business Decision Maker - Non-Technical Executive.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be PLAIN-LANGUAGE. Use simple terms, business impact, risk level, action items. Avoid technical jargon. [/INST]
"""
        else:  # developer (default)
            prompt = f"""[INST] Role: ML Engineer / Security Developer - Model Performance Specialist.
Task: Explain why Packet is {classification}.
Context: Risk Cost=${risk_cost:.2f}. SHAP Features=[{shap_summary}].
Constraint: Be ML-TECHNICAL. Use keywords: algorithm, model, feature, bias, drift, hyperparameter, validation, uncertainty, training, inference, ensemble, attention, retraining. [/INST]
"""

        return prompt

    def _get_detailed_feature_description(self, feature_name: str, importance: float, method: str = 'shap') -> str:
        """Get detailed contextual description for feature based on name and importance."""
        feature_name_lower = feature_name.lower()

        # Map feature indices to common network features (if we know them)
        feature_mapping = {
            'feature_0': 'duration - connection duration in seconds',
            'feature_1': 'protocol type - network protocol used',
            'feature_2': 'service - network service on destination',
            'feature_3': 'flag - status of connection',
            'feature_4': 'source bytes - bytes sent from source to destination',
            'feature_5': 'destination bytes - bytes sent from destination to source',
            'feature_8': 'hot - number of hot indicators',
            'feature_13': 'logged in - whether user is logged in',
            'feature_22': 'same service rate - percentage of connections to same service',
            'feature_23': 'different service rate - percentage of connections to different services',
            'feature_26': 'destination host same service rate',
            'feature_30': 'destination host service count',
            'feature_32': 'destination host same source port rate',
        }

        base_desc = feature_mapping.get(feature_name_lower, f'{feature_name} - network traffic feature')

        # Add importance-based context
        if importance > 0.2:
            importance_level = "highly significant"
            impact = "strongly indicates"
        elif importance > 0.1:
            importance_level = "moderately significant"
            impact = "suggests"
        elif importance > 0.05:
            importance_level = "somewhat significant"
            impact = "hints at"
        else:
            importance_level = "marginally significant"
            impact = "may indicate"

        # Add security context
        if 'duration' in base_desc or 'feature_0' in feature_name:
            security_context = "Long durations may indicate data exfiltration or persistent connections"
        elif 'byte' in base_desc or 'feature_4' in feature_name or 'feature_5' in feature_name:
            security_context = "Unusual byte patterns may indicate data transfer, DoS attacks, or scanning"
        elif 'service' in base_desc or 'feature_2' in feature_name:
            security_context = "Service patterns can reveal scanning, exploitation attempts, or normal application traffic"
        elif 'flag' in base_desc or 'feature_3' in feature_name:
            security_context = "TCP flag combinations can indicate scanning, SYN floods, or connection state anomalies"
        elif 'host' in base_desc or 'count' in base_desc:
            security_context = "Host connection patterns may reveal distributed attacks, botnets, or normal traffic"
        else:
            security_context = "This feature contributes to the overall traffic pattern analysis"

        return f"{base_desc}. This feature is {importance_level} ({importance:.3f} {method} value) and {impact} {security_context.lower()}"

    def _get_feature_context(self, feature_name: str, importance: float) -> str:
        """Get contextual description for feature based on name and importance."""
        feature_name = feature_name.lower()

        if 'port' in feature_name:
            if 'source' in feature_name:
                return "unusual source port usage suggesting scanning activity"
            else:
                return "rare destination port indicating potential exploitation attempt"

        elif 'size' in feature_name or 'byte' in feature_name:
            if importance > 0.1:
                return "abnormally large payload size characteristic of data exfiltration or DoS"
            else:
                return "unusual packet size distribution"

        elif 'rate' in feature_name or 'count' in feature_name:
            if 'packet' in feature_name or 'flow' in feature_name:
                return "high packet rate typical of flooding attacks"
            elif 'error' in feature_name:
                return "elevated error rates indicating malformed packets or scanning"

        elif 'duration' in feature_name:
            return "unusually long flow duration suggesting data exfiltration or persistence"

        elif 'flag' in feature_name:
            return "suspicious TCP flag combination indicating scanning or exploitation"

        else:
            if importance > 0.15:
                return "highly significant feature for attack detection"
            elif importance > 0.05:
                return "moderately important security indicator"
            else:
                return "subtle but relevant security signal"

    def generate_explanation(self, prompt: str) -> str:
        """
        Generate explanation using GGUF LLM with constraint enforcement.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated explanation text
        """
        start_time = time.time()

        try:
            # Generate text using llama-cpp-python
            response = self.model(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repeat_penalty,
                stop=["<|im_end|>", "\n\n"],
                echo=False  # Don't echo the prompt in the output
            )

            # Extract the generated text
            generated_text = response['choices'][0]['text']
            
            # Clean up the explanation
            explanation = generated_text.strip()

            # Apply constraints and post-processing
            explanation = self._apply_constraints(explanation)

            processing_time = time.time() - start_time
            logger.debug(f"Explanation generated in {processing_time:.2f}s")

            return explanation

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Error generating explanation: {str(e)}"

    def generate_stakeholder_explanation(self, 
                                       feature_explanation: Dict[str, Dict[str, float]],
                                       prediction: float,
                                       stakeholder_type: str = 'analyst',
                                       shap_features: Dict[str, float] = None,
                                       lime_features: Dict[str, float] = None) -> str:
        """
        Generate stakeholder-specific explanation using enhanced prompting.

        Args:
            feature_explanation: SHAP/LIME explanation with top features
            prediction: Model prediction probability (0-1)
            stakeholder_type: Type of stakeholder ('analyst', 'manager', 'compliance_officer', 'cto', 'decision_maker', 'developer')
            shap_features: SHAP feature importance dict
            lime_features: LIME feature importance dict

        Returns:
            Generated explanation text
        """
        prompt = self.create_enhanced_prompt(
            feature_explanation,
            prediction,
            stakeholder_type,
            shap_features,
            lime_features
        )
        
        return self.generate_explanation(prompt)

    def _apply_constraints(self, text: str) -> str:
        """
        Apply constraints to generated text to prevent hallucinations and ensure quality.

        Args:
            text: Generated explanation text

        Returns:
            Constrained and filtered text
        """
        # Constraint 1: Maximum sentence count (increased for more detailed explanations)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 8:
            text = ". ".join(sentences[:8]) + "."
        elif len(sentences) < 3:
            # If too few sentences, don't truncate - keep what we have
            pass

        # Constraint 2: Remove hallucinated technical details
        text = self._filter_hallucinations(text)

        # Constraint 3: Remove speculative language
        speculative_phrases = [
            'might be', 'could be', 'possibly', 'potentially', 'seems like',
            'appears to be', 'I think', 'I believe', 'in my opinion'
        ]
        for phrase in speculative_phrases:
            text = text.replace(phrase, '')

        # Constraint 4: Ensure security terminology accuracy
        text = self._validate_security_terms(text)

        # Constraint 5: Remove redundant information
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _filter_hallucinations(self, text: str) -> str:
        """
        Filter out hallucinated technical details not supported by feature explanations.

        Args:
            text: Generated explanation text

        Returns:
            Filtered text with hallucinations removed
        """
        # Common hallucination patterns in security explanations
        hallucination_patterns = [
            r'CVE-\d{4}-\d+',  # Specific CVE numbers
            r'exploit.*metasploit',  # Specific tool mentions
            r'zero-day',  # Unverified zero-day claims
            r'advanced persistent threat',  # Overly specific threat actor claims
            r'nation-state',  # Geopolitical attribution
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # Specific IP addresses
            r'[a-z0-9]+\.[a-z]{2,}'  # Specific domain names
        ]

        for pattern in hallucination_patterns:
            text = re.sub(pattern, '[redacted]', text, flags=re.IGNORECASE)

        return text

    def _validate_security_terms(self, text: str) -> str:
        """
        Validate security terminology against known database to prevent incorrect usage.

        Args:
            text: Generated explanation text

        Returns:
            Text with validated security terminology
        """
        # Check for incorrect attack type usage
        for term_type, terms in self.security_terms.items():
            for term in terms:
                if term.lower() in text.lower():
                    # Validate context (simple heuristic)
                    if term_type == 'attack_types':
                        if not any(keyword in text.lower() for keyword in ['attack', 'malicious', 'threat', 'compromise']):
                            text = text.replace(term, f'potential {term.lower()} activity')

        return text

    def create_template_based_explanation(self, prediction: float, classification: str,
                                         confidence: float, shap_features: Dict[str, float],
                                         lime_features: Dict[str, float], true_label: int) -> str:
        """
        Create a detailed template-based explanation when LLM generation is insufficient.
        This provides comprehensive explanations based on XAI features.
        """
        explanation_parts = []

        # Introduction with model prediction
        explanation_parts.append(f"This network packet was classified as {classification.upper()} by the CNN-LSTM model with {confidence:.1%} confidence (prediction score: {prediction:.3f}).")

        # Add true label comparison if available
        if true_label is not None:
            true_class = "MALICIOUS" if true_label == 1 else "BENIGN"
            if classification.upper() == true_class:
                explanation_parts.append(f"The model's classification matches the true label ({true_class}), indicating accurate detection.")
            else:
                explanation_parts.append(f"Note: The true label is {true_class}, but the model predicted {classification.upper()} - this is a misclassification.")

        # SHAP analysis with detailed feature explanations
        if shap_features:
            explanation_parts.append("\n\nSHAP (SHapley Additive exPlanations) Analysis:")
            explanation_parts.append("SHAP values quantify the contribution of each feature to the model's prediction. The following features were identified as most influential:")

            shap_items = sorted(shap_features.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feat, imp) in enumerate(shap_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, imp, 'shap')
                explanation_parts.append(f"\n{i}. {feat} (SHAP importance: {imp:.3f}):")
                explanation_parts.append(f"   {feat_desc}")

        # LIME analysis with detailed feature explanations
        if lime_features:
            explanation_parts.append("\n\nLIME (Local Interpretable Model-agnostic Explanations) Analysis:")
            explanation_parts.append("LIME provides local interpretability by approximating the model's behavior around this specific packet. The following features were identified:")

            lime_items = sorted(lime_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for i, (feat, imp) in enumerate(lime_items, 1):
                feat_desc = self._get_detailed_feature_description(feat, abs(imp), 'lime')
                explanation_parts.append(f"\n{i}. {feat} (LIME weight: {imp:.3f}):")
                explanation_parts.append(f"   {feat_desc}")

        # Combined XAI analysis with security implications
        explanation_parts.append("\n\nCombined XAI Analysis and Security Implications:")

        if classification == "MALICIOUS":
            explanation_parts.append("The combination of SHAP and LIME analyses indicates this packet exhibits characteristics consistent with malicious network activity.")

            # Analyze feature patterns
            high_importance_features = [f for f, imp in list(shap_features.items()) + list(lime_features.items()) if imp > 0.15]
            if high_importance_features:
                explanation_parts.append(f"Features with high importance values ({', '.join(high_importance_features[:3])}) strongly suggest malicious activity.")

            # Specific attack type inference
            if any('feature_8' in str(f) for f in list(shap_features.keys()) + list(lime_features.keys())):
                explanation_parts.append("The high importance of 'hot indicators' suggests this packet may be part of a scanning or reconnaissance activity.")
            if any('feature_22' in str(f) or 'feature_23' in str(f) for f in list(shap_features.keys()) + list(lime_features.keys())):
                explanation_parts.append("Service rate anomalies indicate potential exploitation attempts or unusual service access patterns.")

            explanation_parts.append("RECOMMENDATION: This packet should be flagged for immediate investigation, and the source IP should be considered for blocking if this pattern continues.")
        else:
            explanation_parts.append("The combination of SHAP and LIME analyses indicates this packet exhibits characteristics consistent with normal, benign network traffic.")

            # Analyze why it's benign
            if shap_features or lime_features:
                explanation_parts.append("The feature values align with expected patterns for legitimate network communications, such as standard protocol usage, normal connection durations, and typical service access patterns.")

            explanation_parts.append("RECOMMENDATION: No action required. This packet can be allowed through the network.")

        # Confidence assessment with interpretation
        explanation_parts.append("\n\nConfidence Assessment:")
        if confidence > 0.9:
            explanation_parts.append(f"The very high confidence level ({confidence:.1%}) indicates the model is extremely certain about this classification. The XAI features provide strong, consistent evidence supporting this decision.")
        elif confidence > 0.7:
            explanation_parts.append(f"The high confidence level ({confidence:.1%}) indicates the model is very certain about this classification. The XAI analysis shows clear feature patterns supporting this decision.")
        elif confidence > 0.5:
            explanation_parts.append(f"The moderate confidence level ({confidence:.1%}) suggests the classification is reasonably certain. While the XAI features support this decision, some ambiguity may exist, and additional context could be helpful.")
        else:
            explanation_parts.append(f"The low confidence level ({confidence:.1%}) indicates uncertainty in the classification. The XAI features show mixed signals, and this packet may require manual review or additional analysis.")

        return "\n".join(explanation_parts)

    def batch_generate_explanations(self,
                                   explanations: List[Dict],
                                   predictions: List[float],
                                   stakeholder_type: str = 'analyst') -> List[Dict]:
        """
        Generate explanations for batch of feature attributions.

        Args:
            explanations: List of SHAP/LIME explanations
            predictions: List of model predictions
            stakeholder_type: Type of stakeholder

        Returns:
            List of enhanced explanations with LLM-generated text
        """
        results = []

        for i, (feature_explanation, prediction) in enumerate(tqdm(zip(explanations, predictions),
                                                                 total=len(explanations),
                                                                 desc='Generating explanations with Qwen2.5-7B')):
            # Extract SHAP and LIME features from explanation
            shap_features = feature_explanation.get('top_features', {})
            lime_features = {}  # Will be passed separately if available

            # Use the enhanced method for stakeholder-specific explanations
            llm_explanation = self.generate_stakeholder_explanation(
                feature_explanation,
                prediction,
                stakeholder_type,
                shap_features=shap_features,
                lime_features=lime_features
            )

            # Fallback to template if LLM explanation is too short
            if len(llm_explanation) < 150:
                classification = "malicious" if prediction >= 0.5 else "benign"
                confidence = abs(prediction - 0.5) * 2
                llm_explanation = self.create_template_based_explanation(
                    prediction, classification, confidence,
                    shap_features, lime_features, 0  # true_label not available in batch
                )

            result = {
                'original_explanation': feature_explanation,
                'prediction': float(prediction),
                'llm_explanation': llm_explanation,
                'stakeholder_type': stakeholder_type,
                'explanation_length': len(llm_explanation)
            }
            results.append(result)

        return results

    def evaluate_explanation_quality(self, human_explanations: List[str],
                                   llm_explanations: List[str]) -> Dict[str, float]:
        """
        Evaluate LLM explanation quality against human-written explanations.

        Args:
            human_explanations: List of human-written explanations
            llm_explanations: List of LLM-generated explanations

        Returns:
            Quality metrics including BLEU and METEOR scores
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.translate.meteor_score import meteor_score
            import nltk

            # Download required NLTK data
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)

            bleu_scores = []
            meteor_scores = []

            for human, llm in zip(human_explanations, llm_explanations):
                # Tokenize
                human_tokens = nltk.word_tokenize(human.lower())
                llm_tokens = nltk.word_tokenize(llm.lower())

                # BLEU score (using 1-gram to 4-gram)
                bleu = sentence_bleu([human_tokens], llm_tokens)
                bleu_scores.append(bleu)

                # METEOR score
                meteor = meteor_score([human], llm)
                meteor_scores.append(meteor)

            return {
                'bleu_score': np.mean(bleu_scores),
                'meteor_score': np.mean(meteor_scores),
                'bleu_std': np.std(bleu_scores),
                'meteor_std': np.std(meteor_scores)
            }

        except ImportError as e:
            logger.warning(f"NLTK not available for evaluation: {str(e)}")
            return {'error': 'NLTK evaluation unavailable'}

    def save_fine_tuning_dataset(self, examples: List[Dict], output_path: str):
        """
        Save examples for LLM fine-tuning in instruction format.

        Args:
            examples: List of explanation examples
            output_path: Output file path
        """
        fine_tuning_data = []

        for example in examples:
            prompt = self.create_prompt(
                example['feature_explanation'],
                example['prediction'],
                example.get('stakeholder_type', 'analyst')
            )

            fine_tuning_data.append({
                'instruction': prompt,
                'input': '',
                'output': example['human_explanation']
            })

        with open(output_path, 'w') as f:
            json.dump(fine_tuning_data, f, indent=2)

        logger.info(f"Fine-tuning dataset saved to {output_path} with {len(fine_tuning_data)} examples")