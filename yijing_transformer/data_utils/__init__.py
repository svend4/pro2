from .text_dataset import TextDataset, ShuffledBatchIterator

try:
    from .streaming_dataset import get_batch_streaming, create_train_val_iterators
except ImportError:
    pass  # datasets library not installed
