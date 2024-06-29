import os, torch
from dataset import ShowDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


CHECKPOINT = 'gpt2'
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
OUTPUT_DIR = './finetuned-gpt2'

EPOCHS = 3
BATCH_SIZE = 4


def get_trainer(model, tokenizer, dataset, epochs, batch_size, output_dir) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        no_cuda=True
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


def load_episodes(input_dir) -> list[str]:
    episodes = []

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                episodes.append(filename + '\n\n' + file.read())

    return episodes


def main():
    print(f'GPU available: {torch.cuda.is_available()}')
    print('Loading model and tokenizer...')
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT)
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT)


    print('Creating training dataset...')
    training_dataset = ShowDataset(load_episodes(TRAIN_DIR), tokenizer)


    trainer = get_trainer(model, tokenizer, training_dataset, EPOCHS, BATCH_SIZE, OUTPUT_DIR)


    print('Training...')
    trainer.train()

    print('Saving model...')
    trainer.save_model(OUTPUT_DIR)


    # print('Creating evaluation dataset...')
    # test_dataset = ShowDataset(load_episodes(TEST_DIR), tokenizer)

    # print('Model evaluation...')
    # metrics = trainer.evaluate(eval_dataset=test_dataset)

    # print(metrics)


main()
