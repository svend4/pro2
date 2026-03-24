"""
AIAgentsAdapter — адаптер для домена ИИ-агентов.

Домен ai_agents охватывает:
- Определения и паттерны ИИ-агентов
- Мета-обучение и адаптивные стратегии
- Модели взаимодействия агент↔среда
- Автоматическая генерация и оценка гипотез

В pro2 этот домен проявляется через:
- self_train.py (3-стадийное самообучение)
- bidir_train.py (двунаправленный агент: forward + backward loop)
- AdvancedGenerator (5 стратегий генерации)
- AdaptiveLearningOptimizer (мета-обучение)
"""

from .base import BaseAdapter, PortalEntry


class AIAgentsAdapter(BaseAdapter):
    name = "ai_agents"

    def fetch(self, query: str) -> list[PortalEntry]:
        q = query.lower()
        entries = self._all_entries()
        if q in ("all", "agent", "agents", ""):
            return entries
        return [e for e in entries if q in e.title.lower() or q in e.content.lower()]

    def _all_entries(self) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="ai_agents:self_train",
                title="3-стадийное самообучение (self_train.py)",
                source="svend4/pro2",
                format_type="agent_pattern",
                content=(
                    "Агент самообучения в 3 стадии:\n"
                    "Stage 0 (Q6 topology): фиксированный квантизатор, обучение base model.\n"
                    "Stage 1 (RAG-buffer): модель генерирует, буфер фильтрует, переобучение.\n"
                    "Stage 2 (filtered wild): обучение на отфильтрованных внешних данных.\n"
                    "Каждая стадия использует curriculum с прогрессивным усложнением."
                ),
                metadata={
                    "file": "self_train.py",
                    "stages": 3,
                    "agent_type": "self-improving",
                },
                links=["pro2:self_train", "data7:theory:transformation"],
            ),
            PortalEntry(
                id="ai_agents:bidir_agent",
                title="Двунаправленный агент (bidir_train.py)",
                source="svend4/pro2",
                format_type="agent_pattern",
                content=(
                    "Forward loop: KnowledgeGraph → PageRank → отбор концептов → train batch.\n"
                    "Backward loop: generate → filter → evaluate → update graph.\n"
                    "Цикл: граф направляет обучение, обучение обогащает граф.\n"
                    "Реализует 'недостающую петлю' из data7/knowledge_transformer.py."
                ),
                metadata={
                    "file": "bidir_train.py",
                    "loops": ["forward", "backward"],
                    "agent_type": "bidirectional",
                },
                links=["pro2:bidir", "data7:missing_loop", "infosystems:knowledge_graph"],
            ),
            PortalEntry(
                id="ai_agents:generator",
                title="AdvancedGenerator — 5 стратегий генерации",
                source="svend4/pro2",
                format_type="agent_pattern",
                content=(
                    "5 стратегий генерации текста:\n"
                    "1. greedy — argmax (быстро, детерминированно)\n"
                    "2. nucleus — top-p sampling (разнообразие)\n"
                    "3. beam — beam search (оптимальность)\n"
                    "4. speculative — draft + verify (скорость)\n"
                    "5. dynamic_temp — адаптивная температура (по энтропии)\n"
                    "Каждая стратегия = поведенческий режим агента."
                ),
                metadata={
                    "file": "inference/bridge_inference.py",
                    "class": "AdvancedGenerator",
                    "strategies": 5,
                },
                links=["pro2:generate", "pro2:speculative"],
            ),
            PortalEntry(
                id="ai_agents:hmoe_curriculum",
                title="HMoE Curriculum — 5-фазное обучение экспертов",
                source="svend4/pro2",
                format_type="agent_pattern",
                content=(
                    "Curriculum обучения иерархических экспертов:\n"
                    "Phase 1: CONCRETE experts only (GEO, HYDRO).\n"
                    "Phase 2: + DYNAMIC (AERO, PYRO).\n"
                    "Phase 3: + ABSTRACT (COSMO, NOOS).\n"
                    "Phase 4: Unfreeze router, joint training.\n"
                    "Phase 5: Fine-tune all.\n"
                    "Прогрессивное раскрытие сложности = мета-обучение."
                ),
                metadata={
                    "file": "train_hmoe_curriculum.py",
                    "phases": 5,
                    "agent_type": "curriculum",
                },
                links=["pro2:hmoe", "pro2:domain_routing"],
            ),
            PortalEntry(
                id="ai_agents:nautilus_hierarchy",
                title="Nautilus — 7-камерная иерархия модулей",
                source="svend4/pro2",
                format_type="agent_pattern",
                content=(
                    "NautilusHierarchy: 7 камер от микро до макро.\n"
                    "Прогрессивная активация (NautilusScheduler): камеры включаются "
                    "последовательно по мере обучения.\n"
                    "Entropy-triggered SYNTH: при высокой неопределённости "
                    "включается синтезирующий модуль.\n"
                    "Архитектура = агент с модульным вниманием."
                ),
                metadata={
                    "file": "geometry/nautilus.py",
                    "chambers": 7,
                    "agent_type": "hierarchical",
                },
                links=["pro2:nautilus", "pro2:six_sources"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "format": "ai_agents",
            "native_unit": "Агентный паттерн (стратегия + цикл + curriculum)",
            "domain": "ИИ-агенты, мета-обучение, самосовершенствование",
            "key_agents": [
                "self_train (3-stage self-improving)",
                "bidir (bidirectional knowledge loop)",
                "AdvancedGenerator (5 strategies)",
                "HMoE curriculum (5 phases)",
                "Nautilus (7-chamber hierarchy)",
            ],
        }
