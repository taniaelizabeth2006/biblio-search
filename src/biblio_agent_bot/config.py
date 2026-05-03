from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from biblio_agent_bot.models import TopicConfig


def load_config(path: Path) -> TopicConfig:
    load_dotenv()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return TopicConfig.model_validate(data)
