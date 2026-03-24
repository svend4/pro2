# ⬡ Nautilus Portal

**Единая точка входа для экосистемы svend4**

> Не слияние — совместимость.
> Как Office Suite читает .docx, .pdf, .xlsx не сливая их в один формат —
> Nautilus читает все репозитории экосистемы, находит связи, строит общий вид.

---

## Репозитории экосистемы

| Репо | Формат | Что хранит | Угол зрения |
|------|--------|-----------|-------------|
| [svend4/info1](../svend4/info1) | `.info1` | 74 документа с α-уровнями | Методологический |
| [svend4/pro2](../svend4/pro2) | `.pro2` | Q6-концепты, граф знаний | Семантический |
| [svend4/meta](../svend4/meta) | `.meta` | 256 CA-правил, гексаграммы | Символьный |

---

## Быстрый старт

```bash
git clone https://github.com/svend4/nautilus
cd nautilus
python portal.py --query "кристалл"
python portal.py --serve    # открыть http://localhost:8000
```

---

## Как это работает

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  info1   │  │   pro2   │  │   meta   │
│ α-уровни │  │ Q6-граф  │  │ CA-прав. │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
            ⬡ Nautilus Portal
            (адаптеры + консенсус)
                    │
         ┌──────────┴──────────┐
         │                     │
    текстовый вывод        HTML / веб
```

**Консенсус** — концепт признаётся "согласованным" если найден во всех трёх репо (100% coverage). Частичный консенсус показывает в каких репо концепт есть, в каких отсутствует.

---

## Подключить новый репозиторий

**Минимально (5 минут)** — добавить в корень репо:

```json
// nautilus.json
{
  "name": "my-repo",
  "format": "my-format",
  "native_unit": "...",
  "bridges": {
    "pro2": "...",
    "info1": "..."
  }
}
```

**Полностью** — написать адаптер:

```python
# adapters/my_repo.py
from adapters.base import BaseAdapter, PortalEntry

class MyRepoAdapter(BaseAdapter):
    name = "my-repo"

    def fetch(self, query: str) -> list[PortalEntry]:
        ...

    def describe(self) -> dict:
        ...
```

Уровни совместимости: **0** (обнаруживаемый) → **1** (читаемый) → **2** (связанный) → **3** (интерактивный).

---

## Файловая структура

```
nautilus/
├── README.md          ← эта страница
├── portal.py          ← движок портала
├── nautilus.json      ← реестр репозиториев
├── adapters/
│   ├── base.py        ← BaseAdapter (протокол)
│   ├── info1.py       ← адаптер info1
│   ├── pro2.py        ← адаптер pro2
│   └── meta.py        ← адаптер meta
├── passports/
│   ├── info1.md
│   ├── pro2.md
│   └── meta.md
└── index.html         ← GitHub Pages (статический портал)
```

---

*Nautilus Portal Protocol v1.0 · [svend4/pro2/PORTAL-PROTOCOL.md](../pro2/PORTAL-PROTOCOL.md)*
