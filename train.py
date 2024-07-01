import torch
from dataset import ShowDataset, load_episodes
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


CHECKPOINT = 'gpt2'
TRAIN_DIR = './data/train'
OUTPUT_DIR = './finetuned-gpt2'

EPOCHS = 3
BATCH_SIZE = 4


def get_trainer(model, tokenizer, dataset, epochs, batch_size, output_dir) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        # use_cpu=False,
        # gradient_accumulation_steps=8,
        # fp16=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )


def main():
    print(f'GPU available: {torch.cuda.is_available()}')
    print('Loading model and tokenizer...')
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT)
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token


    print('Fetching training dataset...')
    training_dataset = ShowDataset(load_episodes(TRAIN_DIR), tokenizer)


    print('Training...')
    trainer = get_trainer(model, tokenizer, training_dataset, EPOCHS, BATCH_SIZE, OUTPUT_DIR)
    trainer.train()


    print('Saving model...')
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    main()
    