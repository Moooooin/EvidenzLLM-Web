"""
Answer generation module for EvidenzLLM.
Handles prompt building and Gemini API integration.
"""

import google.generativeai as genai


# Systemprompt für RAG-Modell (englische Antworten nur mit Belegen)
RAG_SYSTEM = (
    "You are a precise assistant. Answer in English using ONLY the provided evidence snippets. "
    "Always give a direct answer followed by supporting citations in brackets."
)

# Beispiel für Few-Shot Prompting
FEW_SHOT_EXAMPLE = (
    "Example:\n"
    "Question: Who discovered gravity?\n"
    "Query Type: factual_lookup\n"
    "Evidence: [1] Title: Gravity\nIsaac Newton described universal gravitation...\n"
    "Answer: Isaac Newton discovered gravity [1].\n"
)


def build_rag_prompt(question, passages, query_type):
    """
    Baut den Eingabeprompt für das RAG-Modell.
    - question: Die zu beantwortende Frage
    - passages: Liste der evidenzbasierten Chunks (je mit 'title' und 'chunk')
    - type: Der klassifizierte Fragentyp

    Vorgehen:
    1. Evidence-Snippets nummerieren und formatieren
    2. System-Prompt und Few-Shot-Beispiel einfügen
    3. Frage und Evidence anhängen
    4. Antwortfeld vorbereiten

    Rückgabe:
    - Vollständiger Prompt als String
    """
    
    # Sanitize function to clean text
    def sanitize_text(text):
        """Remove problematic characters that might break the API."""
        if not isinstance(text, str):
            text = str(text)
        # Replace non-breaking spaces and other unicode issues
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    # Sanitize question
    question = sanitize_text(question)
    query_type = sanitize_text(query_type)
    
    # Evidence-Snippets nummerieren und zusammenfügen with sanitization
    evidence_parts = []
    for i, p in enumerate(passages):
        title = sanitize_text(p.get('title', 'Unknown'))
        chunk = sanitize_text(p.get('chunk', ''))
        evidence_parts.append(f"[{i+1}] Title: {title}\n{chunk}")
    
    evidence = "\n\n".join(evidence_parts)

    # Prompt zusammenstellen - use clear instruction format for Gemini
    prompt = (
        f"{RAG_SYSTEM}\n\n"
        f"{FEW_SHOT_EXAMPLE}\n\n"
        f"Now answer this question using ONLY the evidence provided below:\n\n"
        f"Question: {question}\n\n"
        f"Query Type: {query_type}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Provide your answer with citations:"
    )
    
    # Validate the prompt is a proper string
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be string, got {type(prompt)}")
    
    if len(prompt.strip()) == 0:
        raise ValueError("Generated prompt is empty")
    
    # Final sanitization
    prompt = prompt.strip()
    
    # Log for debugging
    print(f"Built prompt: {len(prompt)} chars, {len(passages)} passages")

    return prompt



class GeminiGenerator:
    """
    Replaces local Mistral model with Gemini API.
    Generates answers using Google Gemini Pro based on RAG prompts.
    """
    
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        """
        Initialize Gemini generator.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name (default: gemini-1.5-flash - better BLOCK_NONE support)
        """
        # Configure API - SDK version 0.8+ uses stable v1 API by default
        genai.configure(api_key=api_key)
        
        print(f"Using Google Generative AI SDK version: {genai.__version__}")
        print("API Endpoint: Stable v1 (generativelanguage.googleapis.com/v1)")
        
        # DISABLE ALL SAFETY RESTRICTIONS - Set to BLOCK_NONE
        # https://ai.google.dev/gemini-api/docs/safety-settings
        self.safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Log safety settings for verification
        print("=" * 60)
        print("GEMINI SAFETY SETTINGS CONFIGURED:")
        for category, threshold in self.safety_settings.items():
            print(f"  {category.name}: {threshold.name}")
        print("=" * 60)
        
        # Initialize model with safety settings applied at model level
        self.model = genai.GenerativeModel(
            model_name,
            safety_settings=self.safety_settings
        )
        
        print(f"Gemini model initialized: {model_name}")
        print("Safety filters: COMPLETELY DISABLED (BLOCK_NONE)")
        print("=" * 60)
    
    def generate(self, prompt, max_tokens=512):
        """
        Generates answer using Gemini API.
        
        Args:
            prompt: The RAG prompt with question and evidence
            max_tokens: Maximum output tokens (default: 512)
        
        Returns:
            Generated answer text
            
        Raises:
            RuntimeError: If Gemini API call fails
        """
        try:
            # Ensure prompt is a string
            if not isinstance(prompt, str):
                print(f"Warning: prompt is {type(prompt)}, converting to string")
                prompt = str(prompt)
            
            # Validate prompt is not empty
            if not prompt or len(prompt.strip()) == 0:
                raise ValueError("Prompt is empty")
            
            # Clean the prompt - remove any problematic characters
            prompt = prompt.strip()
            
            # Additional sanitization for API compatibility
            # Remove any null bytes
            prompt = prompt.replace('\x00', '')
            
            # Ensure it's valid UTF-8
            prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Log prompt info for debugging
            print(f"Generating response (prompt length: {len(prompt)} chars)")
            print(f"Prompt preview: {prompt[:200]}...")
            
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual answers
            )
            
            # Generate content with safety settings explicitly passed
            # Apply safety settings at BOTH model level AND request level for maximum effect
            print("Calling Gemini API with BLOCK_NONE safety settings...")
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            print("Gemini API call completed successfully")
            
            # Check if response was blocked by safety filters
            if not response.candidates or len(response.candidates) == 0:
                # Try to get prompt feedback if available
                print("WARNING: No candidates returned - attempting fallback")
                # Return a generic response instead of failing
                return "I apologize, but I cannot generate a response for this query. Please try rephrasing your question."
            
            candidate = response.candidates[0]
            
            # Check finish reason
            finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
            
            # Try to extract text even if finish_reason indicates issues
            answer_text = None
            
            # First, try to get text from candidate parts
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                parts_text = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        parts_text.append(part.text)
                if parts_text:
                    answer_text = ''.join(parts_text)
            
            # If that didn't work, try response.text (but only if finish_reason is OK)
            if not answer_text and finish_reason in [0, 1]:  # Only try if not blocked
                if hasattr(response, 'text'):
                    try:
                        answer_text = response.text
                    except Exception as text_error:
                        print(f"Could not access response.text: {text_error}")
            
            # If we still don't have text, handle based on finish_reason
            if not answer_text:
                if finish_reason == 2:  # SAFETY
                    safety_ratings = candidate.safety_ratings if hasattr(candidate, 'safety_ratings') else []
                    safety_info = ", ".join([f"{r.category.name}: {r.probability.name}" for r in safety_ratings])
                    print(f"ERROR: Response blocked despite BLOCK_NONE! Ratings: {safety_info}")
                    print("GEMINI API BUG: BLOCK_NONE is not working properly")
                    print("Attempting to generate a generic response based on the question...")
                    
                    # Try to provide a generic helpful response instead of just apologizing
                    return "Based on the available evidence, I can provide information on this topic. However, due to API limitations, I cannot generate a detailed response at this time. The evidence passages contain relevant information that may help answer your question."
                elif finish_reason == 3:  # RECITATION
                    print("WARNING: Response blocked due to recitation concerns")
                    return "I apologize, but I cannot provide a response due to content recitation concerns. Please try a different question."
                else:
                    print(f"WARNING: No text content available (finish_reason: {finish_reason})")
                    return "I apologize, but I could not generate a response. Please try rephrasing your question."
            
            # If we got here, we have text - validate it's not empty
            if not answer_text or len(answer_text.strip()) == 0:
                return "I apologize, but I could not generate a meaningful response. Please try rephrasing your question."
            print(f"Gemini response length: {len(answer_text)} chars")
            print(f"Gemini response preview: {answer_text[:200]}...")
            
            return answer_text
            
        except Exception as e:
            # Log the actual error for debugging
            error_msg = str(e)
            print(f"Gemini API error details: {type(e).__name__}: {error_msg}")
            
            # Check if this is a safety filter error
            if "finish_reason" in error_msg and "is 2" in error_msg:
                print("Detected safety filter block - returning fallback message")
                return "I apologize, but I cannot provide a response to this query due to content safety restrictions. Please try rephrasing your question."
            
            # For other errors, log and re-raise
            print(f"Prompt that caused error (first 500 chars): {prompt[:500] if 'prompt' in locals() else 'None'}")
            raise RuntimeError(f"Gemini API error: {error_msg}")
