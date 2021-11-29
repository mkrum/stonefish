from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from stonefish.rep import create_tokenizer_rep


class CommonGen(Dataset):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", local_files_only=True)
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
