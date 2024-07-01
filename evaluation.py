import torch
from evaluate import load
from dataset import ShowDataset, load_episodes
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


CHECKPOINT = 'gpt2'
MODEL_DIR = './finetuned-gpt2'
TEST_DIR = './data/test'
OUTPUT_DIR = './evaluation'

BATCH_SIZE = 2


def get_eval_trainer(model, tokenizer, dataset, batch_size, output_dir) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        logging_dir='./logs',
        logging_steps=200,
        use_cpu=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )


def compute_metrics(eval_pred, tokenizer):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    # Decode the token IDs to strings
    decoded_preds = [tokenizer.decode([t for t in pred if t is not None], skip_special_tokens=True) for pred in preds]
    decoded_labels = [tokenizer.decode([t for t in label if t is not None], skip_special_tokens=True) for label in labels]

    rouge = load('rouge')
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL'],
    }


def inputModelDir() -> str:
    dirpath = input(f'Model directory or checkpoint ({MODEL_DIR}): ').strip()
    return dirpath if dirpath else MODEL_DIR


def main():
    model_dir = inputModelDir()

    print(f'GPU available: {torch.cuda.is_available()}')
    print('Loading model and tokenizer...')
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    print('Fetching testing dataset...')
    test_dataset = ShowDataset(load_episodes(TEST_DIR), tokenizer)


    print('Evaluating...')
    trainer = get_eval_trainer(model, tokenizer, test_dataset, BATCH_SIZE, OUTPUT_DIR)
    eval_results = trainer.evaluate()


    print('Evaluation results:')
    print(eval_results)


if __name__ == '__main__':
    main()
