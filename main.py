from os import path
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_DIR = './finetuned-gpt2'
OUTPUT_DIR = './data/out'


def inputModelDir() -> str:
    dirpath = input(f'Model directory ({MODEL_DIR}): ').strip()
    return dirpath if dirpath else MODEL_DIR


def inputEpisodeSize() -> int:
    while True:
        size = input('Episode size (512): ').strip().lower()
        if not size:
            return 512
        
        try:
            size = int(size)
            if size <= 0:
                raise ValueError()
            return size
        except: pass

        print('Size must be a positive number.')


def inputPrompt() -> str:
    while True:
        prompt = input('Prompt: ').strip()
        if prompt: return prompt
        print('Prompt must be non empty.')


def save_episode(model, prompt, generated_text) -> None:
    timestamp = round(time.time())
    filepath = f'{OUTPUT_DIR}/{model}_{timestamp}.txt'

    with open(filepath, 'w') as f:
        f.write(f'Prompt: {prompt}\n\n' + generated_text)


def main():
    model_dir = inputModelDir()
    ep_size = inputEpisodeSize()
    prompt = inputPrompt()


    print('Loading model and tokenizer...')
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        num_return_sequences=1,
        max_length=ep_size, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True
    )[0]
    generated_text = tokenizer.decode(output, skip_special_tokens=True)

    print('\n\n\n')
    print(generated_text)
    save_episode(model_dir.split('\\')[-1], prompt, generated_text)


if __name__ == '__main__':
    main()
