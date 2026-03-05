"""
Запуск обучения YiJing-Transformer.

Использование:
    python scripts/train_model.py
    python scripts/train_model.py --checkpoint_dir checkpoints --no-resume
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    train(checkpoint_dir=args.checkpoint_dir, resume=not args.no_resume)
