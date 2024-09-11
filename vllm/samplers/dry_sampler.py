import torch
from vllm.sampling_params import LogitsProcessor

class DRYLogitsProcessor(LogitsProcessor):
    def __init__(self, multiplier: float, base: float, allowed_length: int, sequence_breakers: set[int], _range: int):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self._range > 0:
            token_ids = token_ids[-self._range:]

        last_token = token_ids[-1]

        if last_token in self.sequence_breakers:
            return logits

        match_lengths = {}

        for i in range(len(token_ids) - 2, -1, -1):
            if token_ids[i] == last_token:
                next_token = token_ids[i + 1]

                if next_token in self.sequence_breakers:
                    continue

                match_length = 1
                while i - match_length >= 0 and token_ids[i - match_length] == token_ids[-match_length - 1]:
                    if token_ids[i - match_length] in self.sequence_breakers:
                        break
                    match_length += 1

                match_lengths[next_token] = max(match_lengths.get(next_token, 0), match_length)

        for token, match_length in match_lengths.items():
            if match_length >= self.allowed_length:
                penalty = self.multiplier * self.base ** (match_length - self.allowed_length)
                logits[token] -= penalty

        return logits