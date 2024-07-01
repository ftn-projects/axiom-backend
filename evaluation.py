import torch
from evaluate import load
from dataset import ShowDataset, load_episodes
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


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
        compute_metrics=compute_metrics
    )


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    rouge = load('rouge')

    result = rouge.compute(predictions=preds, references=labels)

    return {
        'rouge1': result['rouge1'].mid.fmeasure,
        'rouge2': result['rouge2'].mid.fmeasure,
        'rougeL': result['rougeL'].mid.fmeasure,
    }


def main():
    print(f'GPU available: {torch.cuda.is_available()}')
    print('Loading model and tokenizer...')
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
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
