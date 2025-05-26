import json
import re
import os
from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
import numpy as np
import nltk
import evaluate
import shutil
import math
import wandb
import gdown


# --- 0. Define Paths (Colab or Local) ---
print("--- Setting up Project Paths ---")
IS_COLAB = False
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    PROJECT_BASE_PATH_ON_DRIVE = "/content/drive/MyDrive/ColabNotebooks/MedicalChatbotProject"
    PROJECT_BASE_PATH = PROJECT_BASE_PATH_ON_DRIVE
    IS_COLAB = True
    print(f"Running in Google Colab. Google Drive mounted.")
    print(f"Project base path on Drive: {PROJECT_BASE_PATH}")
    if not os.path.exists(PROJECT_BASE_PATH):
        os.makedirs(PROJECT_BASE_PATH)
        print(f"Created project base directory on Drive: {PROJECT_BASE_PATH}")
except ImportError:
    PROJECT_BASE_PATH = "."
    PROJECT_BASE_PATH = os.path.abspath(PROJECT_BASE_PATH)
    print(f"Not running in Google Colab. Assuming local execution.")
    print(f"Project base path (local): {os.path.abspath(PROJECT_BASE_PATH)}")

PROCESSED_DATA_DIR = os.path.join(PROJECT_BASE_PATH, "processed_data")
MODEL_CHECKPOINT_HF_ID = "google/flan-t5-base"
MODEL_OUTPUT_NAME = f"{MODEL_CHECKPOINT_HF_ID.split('/')[-1]}-medical-chatbot-finetuned"

CHECKPOINT_DIR = os.path.join(PROJECT_BASE_PATH, "training_checkpoints", MODEL_OUTPUT_NAME)
MODEL_SAVE_DIR = os.path.join(PROJECT_BASE_PATH, "trained_models", MODEL_OUTPUT_NAME)

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Created directory for processed data: {PROCESSED_DATA_DIR}")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Created directory for training checkpoints: {CHECKPOINT_DIR}")

final_model_parent_dir = os.path.dirname(MODEL_SAVE_DIR)
if not os.path.exists(final_model_parent_dir):
    os.makedirs(final_model_parent_dir, exist_ok=True)
    print(f"Created parent directory for final saved models: {final_model_parent_dir}")

print(f"\n--- Path Configuration Summary ---")
print(f"Processed data will be read from: {PROCESSED_DATA_DIR}")
print(f"Training checkpoints will be saved to: {CHECKPOINT_DIR}")
print(f"Final model will be saved to: {MODEL_SAVE_DIR}")


print("\n--- Downloading NLTK Punkt Tokenizer ---")
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer already available.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("'punkt' downloaded.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK punkt check/download: {e}")
try:
    nltk.data.find('tokenizers/punkt_tab.zip')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# --- Configuration ---
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
PREFIX = "Query: "
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 5
WEIGHT_DECAY = 0.01
FP16 = False
CHECKPOINTS_PER_EPOCH = 3
DATA_FRACTION = 1.0
RANDOM_SEED_SUBSAMPLING = 42
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# --- 1. Load Dataset Manually ---

# First we have to download the dataset.
drive_file_ids = {
    "train.jsonl":      "1h2wOddptmAHK8RUgISTtJygM7j9-4CSf", 
    "validation.jsonl": "14laT1uL4QCWWtXqQYLvNdmDAU9ww4IVA",
    "test.jsonl":       "1NpaRf08kZIUJknqnKTQ0kw9rmEiHiA1L",
}

for filename, file_id in drive_file_ids.items():
    destination_path = os.path.join(PROCESSED_DATA_DIR, filename)

    if not os.path.exists(destination_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename} from {url} ...")
        gdown.download(url, output=destination_path, quiet=False)
    else:
        print(f"{filename} already exists at: {destination_path}")

print("\n--- Manually Loading Data and Creating Dataset Objects ---")
def load_jsonl_to_list(file_path):
    data = []
    print(f"Attempting to load: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    if "Patient" in item and "Doctor" in item and \
                       item["Patient"] is not None and item["Doctor"] is not None:
                        data.append(item)
                    else:
                        print(f"Skipping line {i+1} due to missing or None 'Patient' or 'Doctor' field: {line.strip()}")
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line {i+1} in {file_path}: {line.strip()} - Error: {e}")
        print(f"Successfully loaded and validated {len(data)} items from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load or read file {file_path}: {e}")
        return None

train_file_path = os.path.join(PROCESSED_DATA_DIR, "train.jsonl")
validation_file_path = os.path.join(PROCESSED_DATA_DIR, "validation.jsonl")

train_data_list_full = load_jsonl_to_list(train_file_path)
validation_data_list_full = load_jsonl_to_list(validation_file_path)

if train_data_list_full is None or validation_data_list_full is None:
    raise ValueError("Failed to load one or more dataset files. Check paths and previous error messages. For local runs, ensure data is in the 'processed_data' directory.")
if not train_data_list_full:
    raise ValueError("Training data is empty after loading. Check train.jsonl file content and path.")

train_dataset_full = Dataset.from_list(train_data_list_full)
validation_dataset_full = Dataset.from_list(validation_data_list_full)

print(f"\n--- Subsampling Data to {DATA_FRACTION*100:.0f}% ---")
if DATA_FRACTION < 1.0:
    num_train_samples_to_select = int(len(train_dataset_full) * DATA_FRACTION)
    train_dataset_subset = train_dataset_full.shuffle(seed=RANDOM_SEED_SUBSAMPLING).select(range(num_train_samples_to_select))
    print(f"  Original train dataset size: {len(train_dataset_full)}")
    print(f"  Subsampled train dataset size: {len(train_dataset_subset)}")

    num_validation_samples_to_select = max(1, int(len(validation_dataset_full) * DATA_FRACTION))
    if len(validation_dataset_full) > 0 :
        validation_dataset_subset = validation_dataset_full.shuffle(seed=RANDOM_SEED_SUBSAMPLING).select(range(num_validation_samples_to_select))
        print(f"  Original validation dataset size: {len(validation_dataset_full)}")
        print(f"  Subsampled validation dataset size: {len(validation_dataset_subset)}")
    else:
        print("  Original validation dataset is empty. Subsampled validation dataset will also be empty.")
        validation_dataset_subset = validation_dataset_full
else:
    train_dataset_subset = train_dataset_full
    validation_dataset_subset = validation_dataset_full
    print("  DATA_FRACTION is 1.0 or greater, using full dataset.")

raw_datasets = DatasetDict({
    'train': train_dataset_subset,
    'validation': validation_dataset_subset
})

print(f"\nSuccessfully created DatasetDict with (potentially subsampled) data.")
print(f"Final train dataset size for training: {len(raw_datasets['train'])}")
print(f"Final validation dataset size for training: {len(raw_datasets['validation'])}")
if len(raw_datasets['train']) == 0:
    raise ValueError("Training dataset is empty after subsampling. Check DATA_FRACTION and original dataset size.")

print("\n--- Calculating Steps for Periodic Saving/Evaluation/Logging ---")
num_train_samples = len(raw_datasets['train'])
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count(); num_gpus = max(1, num_gpus)
else:
    num_gpus = 1
print(f"  Number of GPUs to be used by Trainer (estimated): {num_gpus}")

samples_processed_per_optimizer_step = PER_DEVICE_TRAIN_BATCH_SIZE * num_gpus
if samples_processed_per_optimizer_step == 0:
    raise ValueError("samples_processed_per_optimizer_step is zero.")

if num_train_samples == 0: optimizer_steps_per_epoch = 0
else: optimizer_steps_per_epoch = math.ceil(num_train_samples / samples_processed_per_optimizer_step)

if optimizer_steps_per_epoch == 0:
    EVAL_SAVE_LOGGING_STEPS = 1
elif CHECKPOINTS_PER_EPOCH <= 0:
    EVAL_SAVE_LOGGING_STEPS = optimizer_steps_per_epoch if optimizer_steps_per_epoch > 0 else 1
elif optimizer_steps_per_epoch < CHECKPOINTS_PER_EPOCH:
    EVAL_SAVE_LOGGING_STEPS = 1
else:
    EVAL_SAVE_LOGGING_STEPS = optimizer_steps_per_epoch // CHECKPOINTS_PER_EPOCH
EVAL_SAVE_LOGGING_STEPS = max(1, EVAL_SAVE_LOGGING_STEPS) # Ensure it's at least 1

print(f"  Number of training samples (after subsampling): {num_train_samples}")
print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch}")
print(f"  Target checkpoints per epoch: {CHECKPOINTS_PER_EPOCH}")
print(f"  => EVAL_SAVE_LOGGING_STEPS set to: {EVAL_SAVE_LOGGING_STEPS}")

# --- 2. Initialize Tokenizer ---
print("\n--- Initializing T5 Tokenizer ---")
tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT_HF_ID)

# --- 3. Preprocess Data ---
print("\n--- Preprocessing Data for T5 ---")
def preprocess_function_t5(examples):
    inputs_text = [PREFIX + str(patient_query) if patient_query is not None else PREFIX for patient_query in examples["Patient"]]
    targets_text = [str(doctor_answer) if doctor_answer is not None else "" for doctor_answer in examples["Doctor"]]
    model_inputs = tokenizer(inputs_text, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets_text, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = [[token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label_ids] for label_ids in labels]
    return model_inputs

if len(raw_datasets['train']) > 0:
    tokenized_train = raw_datasets['train'].map(preprocess_function_t5, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Tokenizing T5 train dataset")
else:
    from datasets import Features, Value, Sequence
    t5_features = Features({'input_ids': Sequence(Value(dtype='int32')), 'attention_mask': Sequence(Value(dtype='int8')), 'labels': Sequence(Value(dtype='int64'))})
    tokenized_train = Dataset.from_dict({k: [] for k in t5_features.keys()}, features=t5_features)
    print("Training dataset is empty, created an empty tokenized train dataset.")

if len(raw_datasets['validation']) > 0:
    tokenized_validation = raw_datasets['validation'].map(preprocess_function_t5, batched=True, remove_columns=raw_datasets["validation"].column_names, desc="Tokenizing T5 validation dataset")
else:
    features_to_use = tokenized_train.features if len(tokenized_train.features) > 0 else Features({'input_ids': Sequence(Value(dtype='int32')), 'attention_mask': Sequence(Value(dtype='int8')), 'labels': Sequence(Value(dtype='int64'))})
    tokenized_validation = Dataset.from_dict({k: [] for k in features_to_use.keys()}, features=features_to_use)
    print("Validation dataset is empty, created an empty tokenized validation dataset.")

tokenized_datasets = DatasetDict({'train': tokenized_train, 'validation': tokenized_validation})
print(f"Tokenized dataset features: {tokenized_datasets['train'].features if len(tokenized_datasets['train']) > 0 else 'Train dataset is empty'}")

if len(tokenized_datasets['train']) > 0:
    has_any_valid_label_token = any(any(l != -100 for l in ex['labels']) for ex in tokenized_datasets['train'])
    if not has_any_valid_label_token: print("CRITICAL: NO tokenized training example has any valid label token!")
    else: print("At least one training example has valid label tokens.")
else:
    print("Tokenized training dataset is empty.")

# --- 4. Load Model ---
print("\n--- Loading Pre-trained T5 Model ---")
model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_HF_ID)

# --- 5. Training Arguments ---
print("\n--- Defining Training Arguments for T5 ---")

# --- Weights & Biases Configuration (User Prompt) ---
USE_WANDB_USER_CHOICE = False
while True:
    user_choice = input("Do you want to use Weights & Biases for logging? (yes/no): ").strip().lower()
    if user_choice in ['yes', 'y']:
        USE_WANDB_USER_CHOICE = True
        break
    elif user_choice in ['no', 'n']:
        USE_WANDB_USER_CHOICE = False
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

WANDB_PROJECT_NAME = "medical-chatbot-t5-finetuning"
WANDB_RUN_NAME = f"{MODEL_OUTPUT_NAME}-epochs_{NUM_TRAIN_EPOCHS}-lr_{LEARNING_RATE}-bs_{PER_DEVICE_TRAIN_BATCH_SIZE*num_gpus}-data_{DATA_FRACTION*100:.0f}pct_seed{RANDOM_SEED_SUBSAMPLING}"
effective_report_to = []
wandb_active_for_run = False

if USE_WANDB_USER_CHOICE:
    if os.getenv("WANDB_DISABLED") == "true":
        print("Weights & Biases is globally disabled via WANDB_DISABLED environment variable. User choice for W&B will be ignored.")
    else:
        try:
            wandb.login() # Attempt login. Might be interactive if not configured.
            os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
            print(f"Weights & Biases enabled by user. Project: {WANDB_PROJECT_NAME}, Run: {WANDB_RUN_NAME}")
            effective_report_to.append("wandb")
            wandb_active_for_run = True
        except Exception as e:
            print(f"W&B login/configuration failed: {e}. Disabling W&B for this run.")
            os.environ["WANDB_DISABLED"] = "true" # Ensure disabled if setup fails
else:
    print("Weights & Biases logging disabled by user choice.")
    os.environ["WANDB_DISABLED"] = "true" # Explicitly disable if user chooses no


save_limit = (CHECKPOINTS_PER_EPOCH * NUM_TRAIN_EPOCHS) + 2
save_limit = max(2, save_limit)

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="steps" if len(tokenized_datasets['validation']) > 0 else "no",
    eval_steps=EVAL_SAVE_LOGGING_STEPS if len(tokenized_datasets['validation']) > 0 and EVAL_SAVE_LOGGING_STEPS > 0 else None,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=save_limit,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    predict_with_generate=True if len(tokenized_datasets['validation']) > 0 else False,
    fp16=FP16,
    logging_strategy="steps",
    logging_steps=EVAL_SAVE_LOGGING_STEPS if EVAL_SAVE_LOGGING_STEPS > 0 else 10,
    save_strategy="steps",
    save_steps=EVAL_SAVE_LOGGING_STEPS if EVAL_SAVE_LOGGING_STEPS > 0 else 500,
    load_best_model_at_end=True if len(tokenized_datasets['validation']) > 0 else False,
    metric_for_best_model="eval_loss" if len(tokenized_datasets['validation']) > 0 else None,
    report_to=effective_report_to if effective_report_to else None,
    run_name=WANDB_RUN_NAME if wandb_active_for_run else None,
)
print(f"Training arguments: FP16 set to {training_args.fp16}")
if len(tokenized_datasets['validation']) == 0:
    print("Validation dataset is empty. Evaluation, load_best_model_at_end are disabled.")


# --- 6. Data Collator ---
print("\n--- Initializing Data Collator for Seq2Seq ---")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# --- 7. Evaluation Metrics (ROUGE for T5) ---
print("\n--- Setting up Evaluation Metrics (ROUGE) ---")
rouge_metric = evaluate.load("rouge")

def compute_metrics_t5(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    if torch.is_tensor(preds): preds_np = preds.cpu().numpy()
    else: preds_np = np.array(preds)

    vocab_size = tokenizer.vocab_size
    cleaned_preds = np.where((preds_np < 0) | (preds_np >= vocab_size), tokenizer.pad_token_id, preds_np).astype(np.int32)
    decoded_preds = tokenizer.batch_decode(cleaned_preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(p_seq != tokenizer.pad_token_id) for p_seq in cleaned_preds]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v.item() if isinstance(v, np.generic) else v, 4) for k, v in result.items()}

# --- 8. Initialize Trainer ---
print("\n--- Initializing Seq2SeqTrainer for T5 ---")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"] if len(tokenized_datasets['train']) > 0 else None,
    eval_dataset=tokenized_datasets["validation"] if len(tokenized_datasets['validation']) > 0 else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_t5 if training_args.predict_with_generate else None,
)

# --- 9. Start Fine-tuning ---
print("\n--- Starting T5 Fine-tuning ---")
try:
    if len(tokenized_datasets['train']) == 0:
        print("Training dataset is empty. Skipping training.")
    else:
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and any("checkpoint" in d for d in os.listdir(training_args.output_dir)):
             print(f"Checkpoints found in {training_args.output_dir}. Trainer will attempt to resume if applicable.")
             last_checkpoint = True

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        print("\n--- Fine-tuning Complete ---")

        metrics = train_result.metrics
        if 'epoch' not in metrics and hasattr(trainer.state, 'epoch'):
            metrics['epoch'] = round(trainer.state.epoch, 2)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print(f"\n--- Saving final best T5 model to: {MODEL_SAVE_DIR} ---")
        if os.path.exists(MODEL_SAVE_DIR):
            print(f"Warning: Destination path {MODEL_SAVE_DIR} already exists. Overwriting.")
            shutil.rmtree(MODEL_SAVE_DIR)
        trainer.save_model(MODEL_SAVE_DIR)
        print(f"Final best T5 model and tokenizer saved to {MODEL_SAVE_DIR}")

        print("\n--- Example T5 Inference (using the final saved model) ---")
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device {('cuda:0' if device == 0 else 'cpu')} for inference pipeline.")
        
        chatbot = pipeline("text2text-generation", model=MODEL_SAVE_DIR, tokenizer=MODEL_SAVE_DIR, device=device)
        test_patient_query = "I have a persistent cough and a slight fever for 3 days. What should I do?"
        formatted_query_for_t5 = PREFIX + test_patient_query

        response = chatbot(formatted_query_for_t5, max_length=MAX_TARGET_LENGTH, num_beams=5, early_stopping=True)
        print(f"Patient Query: {test_patient_query}")
        print(f"Chatbot Response: {response[0]['generated_text']}")

except Exception as e:
    print(f"\nAn error occurred during T5 training or saving: {e}")
    import traceback
    traceback.print_exc()
finally:
    if wandb_active_for_run and wandb.run is not None:
        print("Finishing W&B run.")
        wandb.finish()
    print("\n--- T5 Training Script Finished ---")