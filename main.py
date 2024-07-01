from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_DIR = './finetuned-gpt2'
EP_SIZES = {
    'small': 512,
    'medium': 1024,
    'large': 2048
}


def inputModelDir() -> str:
    dirpath = input(f'Model directory ({MODEL_DIR}): ').strip()
    return dirpath if dirpath else MODEL_DIR


def inputEpisodeSize() -> int:
    while True:
        size = input('Episode size (medium): ').strip().lower()
        if not size:
            return EP_SIZES['medium']
        
        if size in EP_SIZES:
            return EP_SIZES[size]

        try:
            size = int(size)
            if size <= 0:
                raise ValueError()
            return size
        except: pass

        print('Size must be a positive number or [small, medium, big].')


def inputPrompt() -> str:
    while True:
        prompt = input('Prompt: ').strip()
        if prompt: return prompt
        print('Prompt must be non empty.')


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


if __name__ == '__main__':
    main()
