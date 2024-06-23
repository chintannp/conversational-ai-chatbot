import pandas as pd
from datasets import Dataset
from evaluate import load as load_metric
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments, EvalPrediction
import torch
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the CSV file
qa_df = pd.read_csv('data/course_qa.csv')
train_df, val_df = train_test_split(qa_df, test_size=0.2, random_state=42)  # Split into train and validation
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load the model and tokenizer
model_name = 'distilbert-base-uncased'
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
model.to(device)  # Move model to GPU if available
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Tokenize the dataset
def preprocess(data):
    questions = data['question']
    answers = data['answer']
    inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(answers, padding=True, truncation=True, return_tensors="pt")
    
    # Find start and end positions of the answers in the inputs
    start_positions = []
    end_positions = []
    for i in range(len(questions)):
        input_ids = inputs['input_ids'][i].tolist()
        answer_tokens = targets['input_ids'][i][1:-1].tolist()  # Remove CLS and SEP tokens
        start_idx = None
        end_idx = None
        for idx in range(len(input_ids) - len(answer_tokens) + 1):
            if input_ids[idx:idx + len(answer_tokens)] == answer_tokens:
                start_idx = idx
                end_idx = idx + len(answer_tokens) - 1
                break
        start_positions.append(start_idx if start_idx is not None else 0)
        end_positions.append(end_idx if end_idx is not None else 0)

    inputs["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
    inputs["end_positions"] = torch.tensor(end_positions, dtype=torch.long)

    return inputs

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

# Ensure the dataset tensors are moved to the GPU
def move_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

train_dataset = train_dataset.map(move_to_device, batched=True)
val_dataset = val_dataset.map(move_to_device, batched=True)

# Load metric for evaluation
metric = load_metric("squad_v2", trust_remote_code=True)

# Custom compute metrics function
def compute_metrics(p: EvalPrediction):
    if p.inputs is None:  # Handle NoneType error
        return {}

    start_logits = torch.tensor(p.predictions[0])
    end_logits = torch.tensor(p.predictions[1])


    # Extract start and end positions from label_ids
    start_positions = torch.tensor(p.label_ids['start_positions'])
    end_positions = torch.tensor(p.label_ids['end_positions'])

    # Convert predictions to string
    def decode_predictions(start_logits, end_logits, input_ids):
        start_indexes = torch.argmax(start_logits, dim=1).tolist()
        end_indexes = torch.argmax(end_logits, dim=1).tolist()
        answers = []
        for start, end, input_id in zip(start_indexes, end_indexes, input_ids):
            if start >= len(input_id) or end >= len(input_id):
                answer = ""
            else:
                answer = tokenizer.decode(input_id[start:end+1], skip_special_tokens=True)
            answers.append(answer)
        return answers

    predictions = decode_predictions(start_logits, end_logits, p.inputs['input_ids'])

    # Create formatted output for metric evaluation
    formatted_predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)]
    formatted_references = [{"id": str(i), "answers": {"text": [ans], "answer_start": [0]}} for i, ans in enumerate(start_positions.tolist())]

    return metric.compute(predictions=formatted_predictions, references=formatted_references)

# Training arguments
training_args = TrainingArguments(
    output_dir='./model/fine_tuned_model',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Disable mixed precision training to simplify debugging
)

# Trainer
class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)  # Keep inputs
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                return (loss, (start_logits, end_logits), inputs)  # Keep inputs

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./model/fine_tuned_model')
tokenizer.save_pretrained('./model/fine_tuned_model')
