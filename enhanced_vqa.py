import torch
import gc
import os
from transformers import BlipProcessor, BlipForQuestionAnswering, T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPlantDiseaseVQA:
    def __init__(self, checkpoint_path, processor_path="Salesforce/blip-vqa-base", 
                 enhancer_model_name="google/flan-t5-base", use_cuda=None):
        """
        Initialize the enhanced plant disease VQA system.
        
        Args:
            checkpoint_path: Path to the fine-tuned BLIP model checkpoint
            processor_path: Path or name of the BLIP processor
            enhancer_model_name: Name of the T5 model to use for answer enhancement
            use_cuda: Whether to use CUDA. If None, will detect automatically.
        """
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Determine device
        if use_cuda is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize BLIP model and processor
        try:
            self.processor = BlipProcessor.from_pretrained(processor_path)
            self.model = BlipForQuestionAnswering.from_pretrained(processor_path)
            
            # Load custom checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error initializing BLIP model: {str(e)}")
            raise
        
        # Initialize Flan-T5 model
        try:
            self.enhancer_model_name = enhancer_model_name
            self.enhancer_tokenizer = T5Tokenizer.from_pretrained(enhancer_model_name)
            self.enhancer_model = T5ForConditionalGeneration.from_pretrained(enhancer_model_name).to(self.device)
            self.enhancer_model.eval()  # Set to evaluation mode
            logger.info(f"Loaded enhancer model: {enhancer_model_name}")
        except Exception as e:
            logger.error(f"Error initializing T5 model: {str(e)}")
            raise
    
    @lru_cache(maxsize=32)  # Cache the last 32 enhanced answers
    def enhance_answer_with_huggingface(self, question, initial_answer):
        """Enhance the initial answer using Flan-T5."""
        # Create a prompt for the enhancer model
        prompt = f"""Respond directly with a detailed plant disease analysis. Give specific information about symptoms, causes, and implications.

Context: {initial_answer}
Question: {question}

Answer:"""

        try:
            # Tokenize input
            inputs = self.enhancer_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            # Generate enhanced answer
            with torch.no_grad():
                outputs = self.enhancer_model.generate(
                    **inputs,
                    max_length=300,  # Reduced from 500 for faster inference
                    min_length=50,   # Reduced from 100 for faster inference
                    num_beams=3,     # Reduced from 5 for faster inference
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the response
            response = self.enhancer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and format the response
            enhanced_answer = self._clean_and_format_response(response)
            
            return enhanced_answer
        
        except Exception as e:
            logger.error(f"Error enhancing answer: {str(e)}")
            return self.format_fallback_response(initial_answer)
        finally:
            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _clean_and_format_response(self, text):
        """Clean and format the model's response."""
        # Remove common prefixes
        prefixes_to_remove = [
            "Initial Diagnosis:",
            "Comprehensive Analysis:",
            "Expert Analysis:",
            "Answer:",
            "Context:"
        ]

        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Clean up the text
        text = ' '.join(text.split())
        text = text.replace('_', ' ')

        # Ensure proper capitalization and punctuation
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences if s]
        text = '. '.join(sentences)

        # Add final period if missing
        if not text.endswith('.'):
            text += '.'

        return text

    def format_fallback_response(self, initial_answer):
        """Format the initial answer when enhancement fails."""
        return f"""Disease Detection Result: {initial_answer}
                  Note: Detailed information unavailable. Please consult with a local agricultural expert
                  for specific treatment recommendations."""

    def ask_question(self, image, question):
        """
        Ask a question about the plant disease in the image.
        
        Args:
            image: PIL Image or path to image file
            question: Question about the plant disease
            
        Returns:
            Enhanced answer to the question
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    return f"Error: Image file not found at {image}"
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                return "Error: Invalid image format"
            
            # Process image and question
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,  # Reduced from 5 for faster inference
                    early_stopping=True
                )
            
            # Decode answer
            initial_answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Enhance answer
            enhanced_answer = self.enhance_answer_with_huggingface(question, initial_answer)
            
            return enhanced_answer
            
        except torch.cuda.OutOfMemoryError:
            # Handle GPU memory errors
            torch.cuda.empty_cache()
            gc.collect()
            logger.error("GPU out of memory. Try with a smaller image or on CPU.")
            return "Error: Out of memory. Try with a smaller image or on CPU."
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            # Clean up resources
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            gc.collect()

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            # Clear GPU memory
            if hasattr(self, 'device') and self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Clear large objects
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'enhancer_model'):
                del self.enhancer_model
                
            gc.collect()
        except:
            pass