import warnings
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_DIR = './finetuned-gpt2'


print('Loading model and tokenizer...')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


generated = input('Prompt: ')

print('\n\n\n')

while True:
    # Generate text based on current prompt
    if len(generated) > 1024:
        generated = generated[-1024:]

    input_ids = tokenizer.encode(generated, return_tensors='pt')

    output = model.generate(
        input_ids, 
        num_return_sequences=1, 
        max_new_tokens=20, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True
    )[0]
    generated_text = tokenizer.decode(output, skip_special_tokens=True)

    # Print the difference between prompt and generated text
    diff = generated_text.replace(generated, '', 1)
    print(diff, end='')

    # Update the prompt for next iteration
    generated += generated_text
