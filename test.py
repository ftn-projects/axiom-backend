from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_DIR = './finetuned-gpt2'


print('Loading model and tokenizer...')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


prompt = 'Mordecai and Rigby are'

input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(input_ids, num_return_sequences=1, max_length=1024, pad_token_id=tokenizer.eos_token_id, do_sample=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
