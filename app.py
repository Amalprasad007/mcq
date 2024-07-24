import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import sent_tokenize
import random
import streamlit as st
import pickle

nltk.download('punkt')

def extract_text_from_pdf_with_pymupdf(pdf_file):
    text = ""
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def segment_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def chunk_text(sentences, chunk_size=5):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_models():
    with open('question_gen_tokenizer.pkl', 'rb') as f:
        question_gen_tokenizer = pickle.load(f)
    
    with open('question_gen_model.pkl', 'rb') as f:
        question_gen_model = pickle.load(f)

    with open('qa_pipeline.pkl', 'rb') as f:
        qa_pipeline = pickle.load(f)

    return question_gen_tokenizer, question_gen_model, qa_pipeline

def is_valid_question(question):
    return question.endswith('?') and len(question.split()) > 3  # Ensure it's a question and has more than 3 words

def is_valid_answer(answer):
    return len(answer.split()) > 1  # Ensure the answer is more than 1 word

def main():
    st.title("MCQ Generator from PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf_with_pymupdf(uploaded_file)
        cleaned_text = clean_text(pdf_text)
        sentences = segment_sentences(cleaned_text)
        context_chunks = chunk_text(sentences)
        
        question_gen_tokenizer, question_gen_model, qa_pipeline = load_models()
        
        def generate_questions(context, max_length=50, num_return_sequences=5):
            input_text = "context: " + context + " </s>"
            inputs = question_gen_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
            outputs = question_gen_model.generate(
                inputs, max_length=max_length, num_return_sequences=num_return_sequences,
                do_sample=True, top_k=50, top_p=0.95
            )
            questions = [question_gen_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            return questions
        
        def generate_qa(context, question):
            result = qa_pipeline(question=question, context=context)
            return result['answer']
        
        def create_mcq(context, question, answer):
            words = list(set(context.split()))
            random.shuffle(words)
            options = [answer]
            
            for word in words:
                if word != answer and len(options) < 4:
                    options.append(word)
            
            if len(options) < 4:
                options.extend(['option1', 'option2', 'option3'])  # Fallback options if not enough unique words
            
            options = options[:4]  # Ensure only 4 options
            random.shuffle(options)
            return question, options, answer

        # Generate MCQs
        num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=100, value=5)
        mcq_questions = []

        # Iterate over chunks until the desired number of questions is generated
        for chunk in context_chunks:
            questions = generate_questions(chunk, num_return_sequences=min(num_questions, 5))
            for question in questions:
                if is_valid_question(question):
                    answer = generate_qa(chunk, question)
                    if is_valid_answer(answer):
                        mcq_question, options, correct_answer = create_mcq(chunk, question, answer)
                        mcq_questions.append((mcq_question, options, correct_answer))
                if len(mcq_questions) >= num_questions:
                    break
            if len(mcq_questions) >= num_questions:
                break

        # Display MCQs
        for mcq in mcq_questions:
            st.write(f"Question: {mcq[0]}")
            for i, option in enumerate(mcq[1], 1):
                st.write(f"Option {i}: {option}")
            st.write(f"Correct Answer: {mcq[2]}")
            st.write("\n")

if __name__ == '__main__':
    main()
