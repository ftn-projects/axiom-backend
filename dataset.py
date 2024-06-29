import os
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class ShowDataset(Dataset):
    def __init__(self, episodes: list[str], tokenizer: GPT2Tokenizer, max_length=512) -> None:
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __len__(self) -> int:
        return len(self.episodes)


    def __getitem__(self, index) -> dict:
        episode = self.episodes[index]
        encoding = self.tokenizer(
            episode,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }


def load_episodes(input_dir) -> list[str]:
    episodes = []

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                episodes.append(file.read().strip())

    return episodes
