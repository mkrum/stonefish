from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from stonefish.rep import create_tokenizer_rep


class CommonGen(Dataset):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    BertBaseCased = create_tokenizer_rep("BertBaseCased", tokenizer)

    def __init__(self, split):
        dataset = load_dataset("common_gen")
        self.dataset = dataset[split]

    def __getitem__(self, idx):
        data = self.dataset[idx]
        concept = data["concepts"]
        target = data["target"]

        concept = self.BertBaseCased.from_str_list(concept).to_tensor()
        target = self.BertBaseCased.from_str(target).to_tensor()
        return concept, target

    def __len__(self):
        return len(self.dataset)


class DeEn(Dataset):

    english_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", max_length=256)
    german_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-german-cased", max_length=256
    )

    EnglishBertBaseCased = create_tokenizer_rep(
        "EnglishBertBaseCased", english_tokenizer
    )
    GermanBertBaseCased = create_tokenizer_rep("GermanBertBaseCased", german_tokenizer)

    def __init__(self, split, max_len=256):
        dataset = load_dataset("wmt18", "de-en")
        self.dataset = dataset[split]
        self.max_len = max_len

    def __getitem__(self, idx):
        data = self.dataset[idx]
        de = data["translation"]["de"]
        en = data["translation"]["en"]

        de_tensor = self.GermanBertBaseCased.from_str(de).to_tensor()
        en_tensor = self.EnglishBertBaseCased.from_str(en).to_tensor()
        return de_tensor[: self.max_len], en_tensor[: self.max_len]

    def __len__(self):
        return len(self.dataset)
