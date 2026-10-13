import logging

import numpy as np
import sentencepiece

import vam.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        disable_proprio_for_value: bool = False,
        omit_state_from_prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            if omit_state_from_prompt:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            if disable_proprio_for_value:
                value_prompt = f"Task: {cleaned_text};\nValue: "
                value_tokens = self._tokenizer.encode(value_prompt, add_bos=True)
            else:
                value_prompt = None
                value_tokens = None
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
            value_tokens = tokens.copy()
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        value_mask = None
        if value_tokens is not None:
            value_tokens_len = len(value_tokens)
            if value_tokens_len < self._max_len:
                padding = [False] * (self._max_len - value_tokens_len)
                value_mask = [True] * value_tokens_len + padding
                value_tokens = value_tokens + padding
            else:
                value_tokens = value_tokens[: self._max_len]
                value_mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask), np.asarray(value_tokens), np.asarray(value_mask)

    def decode(self, tokens: np.ndarray | list[int]) -> str:
        """Decode token IDs back to text, skipping padding (0) and EOS (1)."""
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        valid = [int(t) for t in tokens if t not in {0, 1}]
        return self._tokenizer.decode(valid)
