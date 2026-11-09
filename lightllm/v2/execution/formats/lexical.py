from dataclasses import dataclass


@dataclass(frozen=True)
class LexicalStateSpec:
    ngram_context: int
    conv_context: int
    width: int
    eos_token_id: int
    dtype: object

    def validate(self):
        if min(self.ngram_context, self.conv_context, self.width) <= 0:
            raise ValueError("lexical state dimensions must be positive")
        if self.eos_token_id < 0:
            raise ValueError("lexical state requires a nonnegative EOS token ID")
