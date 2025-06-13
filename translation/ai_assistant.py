import openai
from .translator import TranslationManager
from dotenv import load_dotenv

load_dotenv()

class AIAssistant(TranslationManager):
    def answer_question(self, question, context, current_role, timestamp, language):
        try:
            # Prepare prompt with context
            prompt = f"""Context of the conversation:
                    {context}

                    Based on the above context, please answer this question:
                    {question}

                    Please provide a concise and relevant answer based on the context provided.
                    Pay attention, the context and question are from an interview, so sometimes the context is not related to the question.
                    The answer should be in {language}"""
            
            # Get answer from LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": current_role},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Format and display the answer
            formatted_answer = f"[{timestamp}] Q: {question}\nA: {answer}\n"
            return formatted_answer
            
        except Exception as e:
            error_msg = f"[{timestamp}] Error answering question: {str(e)}\n"
            return error_msg

    def is_question(self, text):
        # Detect if text contains a question
        question_markers = {'?', '？'}  # English and Chinese question marks
        question_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom',
            '什么', '何时', '哪里', '谁', '为什么', '怎么', '怎样', '哪个', '谁的'
        }
        
        # Check for question marks
        if any(marker in text for marker in question_markers):
            return True
        
        # Check for question words at the start
        words = text.lower().split()
        if words and words[0] in question_words:
            return True

        return False
