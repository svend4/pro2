"""
InfoSystemsAdapter — адаптер для домена информационных систем.

Домен infosystems охватывает:
- Архитектуры баз знаний (графовые, реляционные, документные)
- Информационные модели предметных областей
- Схемы данных и онтологии
- Паттерны интеграции данных

В pro2 этот домен проявляется через KnowledgeGraph в bidir_train.py
и доменную маршрутизацию в DomainMoE.
"""

from .base import BaseAdapter, PortalEntry


class InfoSystemsAdapter(BaseAdapter):
    name = "infosystems"

    def fetch(self, query: str) -> list[PortalEntry]:
        q = query.lower()
        entries = self._all_entries()
        if q in ("all", ""):
            return entries
        return [e for e in entries if q in e.title.lower() or q in e.content.lower()]

    def _all_entries(self) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="infosystems:knowledge_graph",
                title="KnowledgeGraph — графовая база знаний",
                source="svend4/pro2",
                format_type="schema",
                content=(
                    "Граф концептов с PageRank-центральностью. "
                    "Узлы = Concept(name, hex_idx, pagerank, depth, novelty). "
                    "Рёбра = семантические связи с весами. "
                    "Используется в bidir_train.py для двунаправленного обучения: "
                    "forward (граф→обучение) ↔ backward (генерация→граф)."
                ),
                metadata={
                    "file": "bidir_train.py",
                    "class": "KnowledgeGraph",
                    "nodes": "Concept",
                    "edges": "semantic_link",
                },
                links=["pro2:bidir", "data7:theory:transformation"],
            ),
            PortalEntry(
                id="infosystems:domain_moe",
                title="DomainMoE — доменная маршрутизация экспертов",
                source="svend4/pro2",
                format_type="architecture",
                content=(
                    "Mixture of Experts с доменной специализацией. "
                    "6 доменов: ai_agents, infosystems, knowledge, algorithms, data2, meta. "
                    "Каждый эксперт специализируется на своём домене. "
                    "Domain supervision loss направляет токены к правильному эксперту."
                ),
                metadata={
                    "file": "geometry/ffn.py",
                    "class": "DomainMoE",
                    "n_domains": 6,
                    "domains": ["ai_agents", "infosystems", "knowledge",
                                "algorithms", "data2", "meta"],
                },
                links=["pro2:moe", "pro2:domain_routing"],
            ),
            PortalEntry(
                id="infosystems:q6_ontology",
                title="Q6-онтология: 64 гексаграммы как категориальная система",
                source="svend4/pro2",
                format_type="ontology",
                content=(
                    "64 гексаграммы {-1,+1}^6 = универсальная онтология. "
                    "Каждая вершина гиперкуба = архетипическая категория. "
                    "Хэмминговое расстояние = семантическое расстояние. "
                    "BianGuaTransform = навигация (переход между категориями). "
                    "FactoredQuantizer = проекция текста на ближайшую категорию."
                ),
                metadata={
                    "vertices": 64,
                    "dimensions": 6,
                    "metric": "Hamming",
                },
                links=["pro2:q6", "meta:hexagram", "pro2:biangua"],
            ),
            PortalEntry(
                id="infosystems:corpus_structure",
                title="Структура учебного корпуса (6+1 доменов)",
                source="svend4/pro2",
                format_type="schema",
                content=(
                    "Корпус организован по 6 семантическим доменам:\n"
                    "- ai_agents: определения ИИ-агентов\n"
                    "- infosystems: информационные системы и архитектуры\n"
                    "- knowledge: базы знаний и онтологии\n"
                    "- algorithms: алгоритмы и структуры данных\n"
                    "- data2: данные Крюкова (ETD, 310+ томов)\n"
                    "- meta: мета-документы (CA-правила, гексаграммы)\n"
                    "+ synthetic: синтетические данные для быстрых тестов"
                ),
                metadata={"n_domains": 7, "format": "utf-8 text"},
                links=["data2:etd", "meta:ca_rules", "data7:theory:transformation"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "format": "infosystems",
            "native_unit": "Граф знаний + онтология + доменная модель",
            "domain": "Архитектура информационных систем и баз знаний",
            "key_components": [
                "KnowledgeGraph (bidir_train.py)",
                "DomainMoE (ffn.py)",
                "Q6 ontology (64 hexagrams)",
            ],
        }
