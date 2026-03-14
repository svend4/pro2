"""
BaseAdapter — протокол для всех адаптеров Nautilus Portal.
Каждый адаптер реализует два метода: fetch() и describe().
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class PortalEntry:
    """Универсальная запись — один концепт/документ из любого репо."""
    id: str
    title: str
    source: str
    format_type: str       # "document" | "concept" | "rule"
    content: str
    metadata: dict = field(default_factory=dict)
    links: list = field(default_factory=list)


class BaseAdapter(ABC):
    """
    Протокол адаптера. Реализуй два метода — и репо станет частью портала.

    Пример минимальной реализации:

        class MyAdapter(BaseAdapter):
            name = "my-repo"

            def fetch(self, query: str) -> list[PortalEntry]:
                return [PortalEntry(id="my:1", title="...", ...)]

            def describe(self) -> dict:
                return {"format": "my-format", "native_unit": "..."}
    """

    name: str = "unnamed"

    @abstractmethod
    def fetch(self, query: str) -> list[PortalEntry]:
        """
        Поиск концепта в репозитории.
        Возвращает список PortalEntry — найденные концепты/документы/правила.
        Не должен бросать исключения — при ошибке вернуть пустой список.
        """
        ...

    @abstractmethod
    def describe(self) -> dict:
        """
        Описание формата репозитория.
        Должен возвращать хотя бы: format, native_unit.
        """
        ...
