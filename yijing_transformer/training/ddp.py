"""
Распределённое обучение YiJing-Transformer с поддержкой DDP (Distributed Data Parallel).

Модуль предоставляет:
- DDPConfig: конфигурация распределённого обучения (backend, world_size, ранги)
- DDPWrapper: обёртка для модели — инициализация process group, wrap в DDP,
  DistributedSampler, all-reduce для логирования loss
- setup_ddp_from_env(): автоматическое определение DDP из переменных окружения
  (для запуска через torchrun)
- ddp_train_step(): один шаг обучения с DDP-совместимой синхронизацией градиентов

Использование:
    # Запуск на 4 GPU через torchrun
    torchrun --nproc_per_node=4 training/train.py --ddp

    # Запуск на 2 нодах (по 4 GPU)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \\
        --rdzv_endpoint=master:29500 training/train.py --ddp
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler
from torch.amp import autocast, GradScaler


@dataclass
class DDPConfig:
    """Конфигурация распределённого обучения.

    Атрибуты:
        backend: бэкенд коммуникации ('nccl' для GPU, 'gloo' для CPU)
        world_size: общее число процессов (GPU)
        local_rank: локальный ранг текущего процесса на ноде
        rank: глобальный ранг текущего процесса
        master_addr: адрес мастер-ноды для рандеву
        master_port: порт мастер-ноды для рандеву
    """
    backend: str = 'nccl'
    world_size: int = 1
    local_rank: int = 0
    rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'


class DDPWrapper:
    """Обёртка для распределённого обучения модели через DDP.

    Управляет жизненным циклом process group, оборачивает модель
    в DistributedDataParallel, предоставляет DistributedSampler
    и утилиты для синхронизации метрик между процессами.

    Пример:
        ddp_cfg = DDPConfig(backend='nccl', world_size=4, local_rank=rank)
        wrapper = DDPWrapper(model, ddp_cfg)
        wrapper.setup()
        model = wrapper.wrap_model()
        sampler = wrapper.get_sampler(dataset)
        # ... обучение ...
        wrapper.cleanup()
    """

    def __init__(self, model: torch.nn.Module, ddp_config: DDPConfig):
        self.model = model
        self.config = ddp_config
        self._wrapped_model: Optional[DDP] = None

    def setup(self) -> None:
        """Инициализирует process group для распределённого обучения.

        Устанавливает переменные окружения MASTER_ADDR и MASTER_PORT,
        затем вызывает dist.init_process_group. После инициализации
        привязывает текущий процесс к соответствующему GPU (для CUDA).
        """
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port

        dist.init_process_group(
            backend=self.config.backend,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )

        if torch.cuda.is_available() and self.config.backend == 'nccl':
            torch.cuda.set_device(self.config.local_rank)

    def wrap_model(self) -> DDP:
        """Оборачивает модель в DistributedDataParallel.

        Для CUDA-бэкенда модель переносится на GPU, соответствующий
        local_rank, и оборачивается с device_ids=[local_rank].
        Для CPU (gloo) оборачивается без device_ids.

        Возвращает:
            Модель, обёрнутая в DistributedDataParallel.
        """
        if self.config.backend == 'nccl' and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            self.model = self.model.to(device)
            self._wrapped_model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False,
            )
        else:
            self._wrapped_model = DDP(
                self.model,
                find_unused_parameters=False,
            )
        return self._wrapped_model

    def get_sampler(self, dataset: Dataset) -> DistributedSampler:
        """Создаёт DistributedSampler для разбиения данных между процессами.

        Каждый процесс получает уникальный срез данных без пересечений.
        Sampler нужно вызывать sampler.set_epoch(epoch) каждую эпоху
        для корректного перемешивания.

        Аргументы:
            dataset: PyTorch Dataset для распределённого семплирования.

        Возвращает:
            DistributedSampler, привязанный к текущему world_size и rank.
        """
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
        )

    def cleanup(self) -> None:
        """Уничтожает process group и освобождает ресурсы.

        Вызывается после завершения обучения. Безопасен для
        повторного вызова — проверяет, инициализирована ли группа.
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main_process(self) -> bool:
        """Возвращает True, если текущий процесс — главный (rank 0).

        Используется для условного выполнения операций, которые
        должны происходить только один раз: сохранение чекпоинтов,
        логирование, вывод прогресса.
        """
        return self.config.local_rank == 0

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """All-reduce loss для усреднённого логирования по всем процессам.

        Выполняет all-reduce (SUM) и делит на world_size.
        Результат одинаков на всех процессах.

        Аргументы:
            loss: скалярный тензор с loss текущего процесса.

        Возвращает:
            Усреднённый loss по всем процессам (detached clone).
        """
        reduced = loss.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.config.world_size
        return reduced


def setup_ddp_from_env() -> Optional[DDPConfig]:
    """Автоматическое определение DDP-конфигурации из переменных окружения.

    При запуске через torchrun переменные RANK, WORLD_SIZE и LOCAL_RANK
    устанавливаются автоматически. Функция читает их и создаёт DDPConfig.

    Возвращает:
        DDPConfig, если переменные окружения установлены; None — если нет
        (значит, запуск не через torchrun, DDP не нужен).

    Пример:
        ddp_config = setup_ddp_from_env()
        if ddp_config is not None:
            wrapper = DDPWrapper(model, ddp_config)
            wrapper.setup()
    """
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')
    local_rank = os.environ.get('LOCAL_RANK')

    if rank is None or world_size is None or local_rank is None:
        return None

    # Определяем backend: nccl для GPU, gloo для CPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    return DDPConfig(
        backend=backend,
        world_size=int(world_size),
        local_rank=int(local_rank),
        rank=int(rank),
        master_addr=os.environ.get('MASTER_ADDR', 'localhost'),
        master_port=os.environ.get('MASTER_PORT', '12355'),
    )


def ddp_train_step(
    model: torch.nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    domain_ids: Optional[torch.Tensor] = None,
) -> dict:
    """Один шаг обучения с DDP-совместимой синхронизацией градиентов.

    Выполняет forward, backward, gradient clipping и optimizer step.
    При gradient accumulation синхронизация градиентов происходит
    только на последнем микро-шаге (через no_sync контекст DDP).

    Аргументы:
        model: модель (обёрнутая в DDP или обычная).
        batch: кортеж (input_ids, target_ids) — входные данные.
        optimizer: оптимизатор.
        scaler: GradScaler для mixed precision (None = без AMP).
        grad_accum_steps: число шагов накопления градиентов.
        max_grad_norm: максимальная норма градиентов для clipping.
        use_amp: использовать ли autocast (mixed precision).
        amp_dtype: тип данных для autocast (float16 или bfloat16).
        domain_ids: идентификаторы доменов (для DomainMoE, может быть None).

    Возвращает:
        Словарь с метриками шага:
        - 'loss': значение loss (float)
        - 'grad_norm': норма градиентов после clipping (float)
    """
    xb, yb = batch

    # Определяем устройство для autocast
    device_type = xb.device.type if xb.device.type in ('cuda', 'cpu') else 'cuda'

    # Контекст для отложенной синхронизации градиентов при gradient accumulation
    # DDP синхронизирует градиенты только на последнем микро-шаге
    is_ddp = isinstance(model, DDP)

    total_loss = 0.0

    for micro_step in range(grad_accum_steps):
        # На всех микро-шагах, кроме последнего, отключаем синхронизацию
        sync_context = (
            model.no_sync() if is_ddp and micro_step < grad_accum_steps - 1
            else _nullcontext()
        )

        with sync_context:
            with autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                logits, loss, _ = model(xb, yb, domain_ids=domain_ids)
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        total_loss += loss.item() * grad_accum_steps

    # Gradient clipping и optimizer step
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    return {
        'loss': total_loss / grad_accum_steps,
        'grad_norm': grad_norm_val,
    }


class _nullcontext:
    """Минимальный контекстный менеджер-заглушка (для Python <3.7 совместимости)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
