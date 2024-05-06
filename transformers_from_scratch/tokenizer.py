from abc import ABC, abstractmethod

class Tokenizer(ABC):

    def __init__(self) -> None:
        self._vocab_size: int | None = None
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise Exception("`vocab_size` has not been set!")

        return self._vocab_size
    
    @abstractmethod
    def train(self, training_text: str) -> None:
        pass
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass
    

class CharacterTokenizer(Tokenizer):

    def train(self, training_text: str) -> None:
        unique_tokens = sorted(list(set(training_text)))
        self._vocab_size = len(unique_tokens)

        for idx, token in enumerate(unique_tokens):
            self._char_to_idx[token] = idx
            self._idx_to_char[idx] = token

    def encode(self, text: str) -> list[int]:
        return [
            self._char_to_idx[token]
            for token in text
        ]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            [
                self._idx_to_char[idx]
                for idx in tokens
            ]
        )
