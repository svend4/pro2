"""
v17 утилиты: RAG scaffold, Activation Statistics.

RAG (Retrieval-Augmented Generation): scaffold для retrieval +
generation pipeline. Включает простой in-memory retriever
на cosine similarity и интеграцию с моделью.

Activation Statistics: трекер для мониторинга здоровья модели.
Отслеживает нормы активаций, мёртвые нейроны, gradient flow.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ==================== RAG Scaffold ====================

class SimpleRetriever:
    """
    Простой in-memory retriever на cosine similarity.

    Хранит документы как пары (text, embedding).
    При запросе возвращает top-k наиболее похожих.

    Args:
        embedding_dim: размерность эмбеддингов
    """
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.documents = []  # list of str
        self.embeddings = None  # (N, D) tensor

    def add_documents(self, docs, embeddings):
        """
        Добавляет документы с их эмбеддингами.

        Args:
            docs: list[str]
            embeddings: (N, D) tensor
        """
        self.documents.extend(docs)
        if self.embeddings is None:
            self.embeddings = embeddings.detach()
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings.detach()])

    def retrieve(self, query_embedding, top_k=3):
        """
        Находит top-k похожих документов.

        Args:
            query_embedding: (D,) или (1, D)
            top_k: число документов

        Returns:
            list[dict]: [{text, score, index}, ...]
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q = query_embedding.flatten()
        q = F.normalize(q.unsqueeze(0), dim=-1)
        db = F.normalize(self.embeddings, dim=-1)

        scores = (q @ db.T).squeeze(0)  # (N,)
        k = min(top_k, len(self.documents))
        top_vals, top_idx = scores.topk(k)

        results = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            results.append({
                'text': self.documents[idx],
                'score': val,
                'index': idx,
            })

        return results

    def __len__(self):
        return len(self.documents)


class RAGPipeline:
    """
    RAG pipeline: retrieve → augment context → generate.

    Использование:
        rag = RAGPipeline(model, retriever, tokenizer)
        rag.index_documents(["doc1", "doc2", ...])
        answer = rag.generate("query text", max_tokens=50)
    """
    def __init__(self, model, retriever, tokenizer, embed_fn=None):
        """
        Args:
            model: YiJingGPT
            retriever: SimpleRetriever
            tokenizer: tokenizer с encode/decode
            embed_fn: функция text → embedding (None = avg model embeddings)
        """
        self.model = model
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.embed_fn = embed_fn or self._default_embed

    def _default_embed(self, text):
        """Простой embed: среднее token embeddings."""
        ids = self.tokenizer.encode(text)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.tok_emb(ids_t)
        return emb.mean(dim=1).squeeze(0)  # (D,)

    def index_documents(self, documents):
        """Индексирует документы для retrieval."""
        embeddings = []
        for doc in documents:
            emb = self.embed_fn(doc)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings)
        self.retriever.add_documents(documents, embeddings)

    def retrieve_context(self, query, top_k=3):
        """Находит релевантный контекст для запроса."""
        q_emb = self.embed_fn(query)
        results = self.retriever.retrieve(q_emb, top_k=top_k)
        context = " ".join(r['text'] for r in results)
        return context, results

    def augmented_ids(self, query, top_k=3):
        """Создаёт augmented input: context + query."""
        context, retrieved = self.retrieve_context(query, top_k)
        augmented_text = context + " " + query
        ids = self.tokenizer.encode(augmented_text)
        return ids, retrieved


# ==================== Activation Statistics ====================

class ActivationTracker:
    """
    Трекер активаций для мониторинга здоровья модели.

    Отслеживает:
    - Нормы активаций per layer
    - Мёртвые нейроны (всегда нулевые)
    - Gradient norms per layer
    - Статистику attention weights

    Использование:
        tracker = ActivationTracker(model)
        tracker.start()
        model(x, y)  # forward + backward
        stats = tracker.get_stats()
        tracker.stop()
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activation_norms = defaultdict(list)
        self.gradient_norms = defaultdict(list)
        self.dead_neuron_counts = {}
        self._running = False

    def start(self):
        """Регистрирует hooks для мониторинга."""
        self.stop()
        self._running = True

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Forward hook
                handle = module.register_forward_hook(
                    self._make_fwd_hook(name)
                )
                self.hooks.append(handle)

                # Backward hook
                handle = module.register_full_backward_hook(
                    self._make_bwd_hook(name)
                )
                self.hooks.append(handle)

    def stop(self):
        """Удаляет hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._running = False

    def _make_fwd_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activation_norms[name].append(output.norm().item())
                # Dead neurons: output == 0 для всех входов
                if output.dim() >= 2:
                    dead = (output.abs().sum(dim=tuple(range(output.dim() - 1))) < 1e-8)
                    self.dead_neuron_counts[name] = dead.sum().item()
        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradient_norms[name].append(
                    grad_output[0].norm().item()
                )
        return hook

    def get_stats(self):
        """Возвращает собранную статистику."""
        stats = {}

        # Activation norms
        if self.activation_norms:
            norms = {k: sum(v) / len(v) for k, v in self.activation_norms.items() if v}
            stats['activation_norms'] = norms
            stats['avg_activation_norm'] = sum(norms.values()) / len(norms) if norms else 0

        # Gradient norms
        if self.gradient_norms:
            gnorms = {k: sum(v) / len(v) for k, v in self.gradient_norms.items() if v}
            stats['gradient_norms'] = gnorms
            stats['avg_gradient_norm'] = sum(gnorms.values()) / len(gnorms) if gnorms else 0

        # Dead neurons
        if self.dead_neuron_counts:
            stats['dead_neurons'] = dict(self.dead_neuron_counts)
            stats['total_dead'] = sum(self.dead_neuron_counts.values())

        return stats

    def reset(self):
        """Сбрасывает накопленную статистику."""
        self.activation_norms.clear()
        self.gradient_norms.clear()
        self.dead_neuron_counts.clear()
