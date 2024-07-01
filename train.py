import torch
from dataset import ShowDataset, load_episodes
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from accelerate import Accelerator, find_executable_batch_size


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


def training_function(starting_batch_size, epochs, checkpoint, train_dir, output_dir):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inner_training_function(batch_size):
        nonlocal accelerator
        accelerator.free_memory()  # Free all lingering references

        model = GPT2LMHeadModel.from_pretrained(checkpoint)
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        training_dataset = ShowDataset(load_episodes(train_dir), tokenizer)

        trainer = get_trainer(model, tokenizer, training_dataset, epochs, batch_size, output_dir)
        trainer.train()
        trainer.save_model(output_dir)

        del model
        del tokenizer
        torch.cuda.empty_cache()


    inner_training_function()


def main():
    print(f'GPU available: {torch.cuda.is_available()}')

    print('Training...')
    training_function(BATCH_SIZE, EPOCHS, CHECKPOINT, TRAIN_DIR, OUTPUT_DIR)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
