import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load the models and tokenizer (use the appropriate model names)
question_gen_tokenizer = T5Tokenizer.from_pretrained('valhalla/t5-small-qg-hl')
question_gen_model = T5ForConditionalGeneration.from_pretrained('valhalla/t5-small-qg-hl')
tokenizer = AutoTokenizer.from_pretrained("Dingyun-Huang/roberta-large-squad1")
model = AutoModelForQuestionAnswering.from_pretrained("Dingyun-Huang/roberta-large-squad1")
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Save the models and tokenizer as pickle files
with open('question_gen_tokenizer.pkl', 'wb') as f:
    pickle.dump(question_gen_tokenizer, f)

with open('question_gen_model.pkl', 'wb') as f:
    pickle.dump(question_gen_model, f)

with open('qa_pipeline.pkl', 'wb') as f:
    pickle.dump(qa_pipeline, f)
