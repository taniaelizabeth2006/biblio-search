from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from biblio_agent_bot.agents import format_citation
from biblio_agent_bot.models import ReviewState


def write_outputs(review: ReviewState, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(review.synthesis_markdown or "", encoding="utf-8")
    (output_dir / "audit.json").write_text(
        review.model_dump_json(indent=2),
        encoding="utf-8",
    )
    records = []
    decisions = {decision.stable_id: decision for decision in review.decisions}
    embedding_ranks = {rank.stable_id: rank for rank in review.embedding_ranks}
    for article in review.deduped_records:
        decision = decisions.get(article.stable_id)
        embedding_rank = embedding_ranks.get(article.stable_id)
        records.append(
            {
                "embedding_rank": embedding_rank.rank if embedding_rank else None,
                "embedding_score": embedding_rank.score if embedding_rank else None,
                "embedding_text_source": embedding_rank.text_source if embedding_rank else None,
                "included": decision.include if decision else None,
                "score": decision.relevance_score if decision else None,
                "reasons": " | ".join(decision.reasons) if decision else None,
                "key_contribution": decision.key_contribution if decision else None,
                "limitations": decision.limitations if decision else None,
                "source": article.source,
                "title": article.title,
                "authors": "; ".join(article.authors),
                "year": article.year,
                "journal": article.journal,
                "doi": article.doi,
                "pmid": article.pmid,
                "url": str(article.url) if article.url else None,
                "citation": format_citation(article),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "screening_matrix.csv", index=False)
    (output_dir / "search_plan.json").write_text(
        review.plan.model_dump_json(indent=2) if review.plan else "{}",
        encoding="utf-8",
    )
    (output_dir / "raw_records.json").write_text(
        json.dumps([article.model_dump(mode="json") for article in review.raw_records], indent=2),
        encoding="utf-8",
    )
