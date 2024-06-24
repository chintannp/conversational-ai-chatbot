from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from PyPDF2 import PdfReader
import os 

app = Flask(__name__)

cache_dir = "/app/cache"  # Or wherever you want to store the cache within your app

# Set the cache directory for transformers
os.environ['HF_HOME'] = './cache'


# Load pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)

with open('data/qa.txt','r') as qa_file:
    txt = qa_file.read()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Extract text from the PDF (provide the correct path to your PDF file)
context_text = extract_text_from_pdf('data/your_document.pdf')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    inputs = tokenizer.encode_plus(user_input, txt+"\n"+context_text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print(f"User Input: {user_input}")
    print(f"Tokenized Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        print(f"Start Logits: {start_logits}")
        print(f"End Logits: {end_logits}")

        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()

    if start_index <= end_index:
        answer_tokens = input_ids[0][start_index:end_index+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    else:
        answer = ""

    print(f"Answer Tokens: {answer_tokens}")
    print(f"Decoded Answer: {answer}")

    if not answer.strip():
        answer = "I couldn't find an answer to your question."

    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)


