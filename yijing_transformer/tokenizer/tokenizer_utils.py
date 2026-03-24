"""Загрузка SentencePiece токенизатора."""

import os
import sentencepiece as spm


def load_tokenizer(model_path=None):
    if model_path is None:
        # Ищем относительно расположения модуля, а не CWD
        _dir = os.path.dirname(os.path.abspath(__file__))
        _project = os.path.dirname(os.path.dirname(_dir))
        candidates = [
            os.path.join(_project, "vocab", "e8_morpheme_.model"),
            os.path.join(_dir, "..", "..", "vocab", "e8_morpheme_.model"),
            "vocab/e8_morpheme_.model",
            "../vocab/e8_morpheme_.model",
        ]
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            raise FileNotFoundError(
                "Tokenizer model not found. "
                "Place e8_morpheme_.model in vocab/ or pass path explicitly."
            )
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
