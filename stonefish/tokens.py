from chessenv.rep import CBoard
from stonefish.rep import MoveToken
from transformers import PreTrainedTokenizer

from contextlib import contextmanager


class BoardMoveSeq2SeqTokenizer(PreTrainedTokenizer):
    def __init__(self, max_len=1024):
        super().__init__(max_len=max_len, pad_token="<pad>")
        self._board_tokenizer = BoardTokenizer()
        self._move_tokenizer = MoveTokenizer()
        self.current_tokenizer = self._board_tokenizer

    def save_vocabulary(self, *args, **kwargs):
        pass
        return ()

    def load_vocabulary(self, *args, **kwargs):
        pass
        return ()

    @contextmanager
    def as_target_tokenizer(self):
        self.current_tokenizer = self._move_tokenizer
        yield
        self.current_tokenizer = self._board_tokenizer

    def _tokenize(self, sequence):
        return self.current_tokenizer._tokenize(sequence)

    def _convert_token_to_id(self, token):
        return self.current_tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, idx):
        return self.current_tokenizer._convert_id_to_token(idx)

    def get_vocab(self):
        return self.current_tokenizer.SYMBOLS

    @property
    def vocab_size(self):
        return len(self.current_tokenizer.SYMBOLS)


class BoardTokenizer(PreTrainedTokenizer):
    rep = CBoard
    SYMBOLS = [
        "e",
        "P",
        "N",
        "B",
        "R",
        "Q",
        "K",
        "p",
        "n",
        "b",
        "r",
        "q",
        "k",
        "ep",
        "w",
        "b",
        "cK",
        "!cK",
        "cQ",
        "!cQ",
        "ck",
        "!ck",
        "cq",
        "!cq",
        "<pad>",
    ]

    def __init__(self, max_len=1024):
        super().__init__(max_len=max_len, pad_token="<pad>")

    def _tokenize(self, sequence):
        return [
            self.SYMBOLS[i] for i in self.rep.from_fen(sequence).to_array().tolist()
        ]

    def _convert_token_to_id(self, token):
        return self.SYMBOLS.index(token)

    def _convert_id_to_token(self, idx):
        return self.SYMBOLS[idx]

    def get_vocab(self):
        return self.SYMBOLS

    @property
    def vocab_size(self):
        return len(self.SYMBOLS)


class MoveTokenizer(PreTrainedTokenizer):

    rep = MoveToken
    _bos_token = "<start>"

    def __init__(self, max_len=1024):
        super().__init__(max_len=max_len, pad_token="<pad>")

    def _tokenize(self, sequence):
        return ["<start>", sequence[:2], sequence[2:]]

    def _convert_token_to_id(self, token):
        return self.rep.from_str(token).to_int()

    def _convert_id_to_token(self, idx):
        return self.rep.from_int(idx).to_str()

    def get_vocab(self):
        return self.rep._tokens

    @property
    def vocab_size(self):
        return len(self.rep._tokens)
