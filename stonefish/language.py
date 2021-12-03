from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from stonefish.rep import create_tokenizer_rep


class CommonGen(Dataset):
    def __init__(self, tokenizer, split):
        dataset = load_dataset("common_gen")
        self.dataset = dataset[split]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data = self.dataset[idx]
        concept = " ".join(data["concepts"])
        target = data["target"]

        concept = self.tokenizer.encode(concept, return_tensors="pt")[0]
        target = self.tokenizer.encode(target, return_tensors="pt")[0]
        return concept, target

    def __len__(self):
        return len(self.dataset)


class CommonGenEval(Dataset):
    def __init__(self, tokenizer):
        dataset = load_dataset("common_gen")
        self.tokenizer = tokenizer

        dataset = dataset["validation"]

        self.dataset = {}

        for d in dataset:
            id_ = d["concept_set_idx"]

            if id_ not in self.dataset.keys():
                self.dataset[id_] = (d["concepts"], [])

            self.dataset[id_][1].append(d["target"])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data[0], data[1]

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def collate_fn(cls, batch):
        starters, targets = zip(*batch)
        starters = pad_sequence(starters, batch_first=True, padding_value=-1)
        return starters, targets


class DeEn(Dataset):

    # english_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", max_length=256)
    # german_tokenizer = AutoTokenizer.from_pretrained(
    #    "bert-base-german-cased", max_length=256
    # )

    # EnglishBertBaseCased = create_tokenizer_rep(
    #    "EnglishBertBaseCased", english_tokenizer
    # )
    # GermanBertBaseCased = create_tokenizer_rep("GermanBertBaseCased", german_tokenizer)

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


class SingleCommonGen(Dataset):
    def __init__(self, token_rep, split):
        super().__init__()
        self.token_rep = token_rep
        dataset = load_dataset("common_gen")
        self.dataset = dataset[split]

    def __getitem__(self, idx):
        data = self.dataset[idx]
        concept = data["concepts"]
        target = data["target"]
        return self.token_rep.from_str(
            "<|endoftext|>" + ", ".join(concept) + ": " + target + "<|endoftext|>"
        ).to_tensor()

    def __len__(self):
        return len(self.dataset)
