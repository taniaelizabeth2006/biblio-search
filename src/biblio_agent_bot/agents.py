from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel

from biblio_agent_bot.embeddings import embedding_provider_from_config, rank_articles_by_embedding
from biblio_agent_bot.json_tools import parse_model
from biblio_agent_bot.llm import LLMProvider, NullProvider
from biblio_agent_bot.models import (
    Article,
    ReviewState,
    SearchConcept,
    ScreeningDecision,
    SearchPlan,
    TopicConfig,
)
from biblio_agent_bot.sources import source_client


class ScreeningBatch(BaseModel):
    decisions: list[ScreeningDecision]


class AgentState(TypedDict, total=False):
    review: ReviewState


class BiblioAgentBot:
    def __init__(self, llm: LLMProvider, progress: Callable[[str], None] | None = None) -> None:
        self.llm = llm
        self.progress = progress or (lambda _message: None)

    def run(self, config: TopicConfig) -> ReviewState:
        state = ReviewState(config=config)
        try:
            from langgraph.graph import END, StateGraph
        except Exception as exc:
            self._progress("LangGraph unavailable; falling back to sequential runner.")
            state.audit.append(f"LangGraph unavailable or failed ({exc}); using sequential runner.")
            for node in [
                self._node_plan,
                self._node_search,
                self._node_dedupe,
                self._node_embedding_filter,
                self._node_screen,
                self._node_synthesize,
            ]:
                state = node({"review": state})["review"]
            return state
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._node_plan)
        graph.add_node("search", self._node_search)
        graph.add_node("dedupe", self._node_dedupe)
        graph.add_node("embedding_filter", self._node_embedding_filter)
        graph.add_node("screen", self._node_screen)
        graph.add_node("synthesize", self._node_synthesize)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "dedupe")
        graph.add_edge("dedupe", "embedding_filter")
        graph.add_edge("embedding_filter", "screen")
        graph.add_edge("screen", "synthesize")
        graph.add_edge("synthesize", END)
        compiled = graph.compile()
        result = compiled.invoke({"review": state})
        return result["review"]

    def _node_plan(self, state: AgentState) -> AgentState:
        review = state["review"]
        self._progress(f"[1/6] Building PECO/search plan with {self.llm.name}...")
        review.plan = self._build_search_plan(review.config)
        self._progress(f"[1/6] Search plan ready. PubMed query: {_shorten(review.plan.pubmed_query)}")
        review.audit.append("PECO/Search-plan agent completed.")
        return {"review": review}

    def _node_search(self, state: AgentState) -> AgentState:
        review = state["review"]
        assert review.plan is not None
        self._progress(
            f"[2/6] Searching sources ({', '.join(review.config.sources)}), "
            f"up to {review.config.max_records_per_source} records each..."
        )
        query_by_source = {
            "pubmed": review.plan.pubmed_query,
            "europepmc": review.plan.europepmc_query,
            "crossref": review.plan.crossref_query,
        }
        for source in review.config.sources:
            query = query_by_source[source]
            self._progress(f"[2/6] {source}: searching. Query: {_shorten(query)}")
            try:
                client = source_client(source)
                articles = client.search(query, max_records=review.config.max_records_per_source)
            except Exception as exc:
                self._progress(f"[2/6] {source}: failed, continuing. Error: {_shorten(str(exc), 180)}")
                review.audit.append(f"{source}: failed for query {query}. Error: {exc}")
                continue
            review.raw_records.extend(articles)
            hit_count = getattr(client, "last_hit_count", None)
            summary = _retrieval_summary(len(articles), hit_count)
            self._progress(f"[2/6] {source}: {summary}.")
            review.audit.append(f"{source}: {summary} for query: {query}")
        return {"review": review}

    def _node_dedupe(self, state: AgentState) -> AgentState:
        review = state["review"]
        self._progress(f"[3/6] Deduplicating {len(review.raw_records)} raw records...")
        review.deduped_records = dedupe_articles(review.raw_records)
        self._progress(f"[3/6] Deduplication complete: {len(review.deduped_records)} unique records.")
        review.audit.append(
            f"Dedupe agent: {len(review.raw_records)} raw -> {len(review.deduped_records)} unique."
        )
        return {"review": review}

    def _node_embedding_filter(self, state: AgentState) -> AgentState:
        review = state["review"]
        if review.config.embedding_provider == "none":
            self._progress("[4/6] Embedding prefilter disabled; all unique records go to screening.")
            review.audit.append("Embedding filter skipped.")
            return {"review": review}

        self._progress(
            f"[4/6] Ranking {len(review.deduped_records)} records with "
            f"{review.config.embedding_provider}:{review.config.embedding_model}..."
        )
        provider = embedding_provider_from_config(
            review.config,
            cache_dir=Path(".biblio-agent-cache") / "embeddings",
            progress=self._progress,
        )
        ranks, selected = rank_articles_by_embedding(review.deduped_records, review.config, provider)
        review.embedding_ranks = ranks
        review.embedding_selected_records = selected
        review.audit.append(
            f"Embedding filter ranked {len(ranks)} records and selected {len(selected)} "
            f"for LLM screening using top_k={review.config.embedding_top_k}, "
            f"min_score={review.config.embedding_min_score}."
        )
        if ranks:
            best = ranks[0]
            self._progress(
                f"[4/6] Embedding ranking complete: selected {len(selected)}/{len(ranks)}. "
                f"Best score={best.score:.4f}."
            )
        else:
            self._progress("[4/6] Embedding ranking complete: no records had title/abstract text.")
        return {"review": review}

    def _node_screen(self, state: AgentState) -> AgentState:
        review = state["review"]
        records_to_screen = (
            review.embedding_selected_records
            if review.config.embedding_provider != "none"
            else review.deduped_records
        )
        self._progress(
            f"[5/6] Screening {len(records_to_screen)} records with {self.llm.name}..."
        )
        if isinstance(self.llm, NullProvider):
            decisions = [heuristic_screen(article, review.config) for article in records_to_screen]
        else:
            decisions = self._llm_screen(records_to_screen, review.config)
        review.decisions = sorted(decisions, key=lambda item: item.relevance_score, reverse=True)
        include_ids = {decision.stable_id for decision in review.decisions if decision.include}
        review.selected_records = [
            article for article in records_to_screen if article.stable_id in include_ids
        ]
        self._progress(f"[5/6] Screening complete: {len(review.selected_records)} records selected.")
        review.audit.append(
            f"Screening agent selected {len(review.selected_records)} articles for synthesis."
        )
        return {"review": review}

    def _node_synthesize(self, state: AgentState) -> AgentState:
        review = state["review"]
        self._progress(f"[6/6] Writing synthesis with {self.llm.name}...")
        if isinstance(self.llm, NullProvider):
            review.synthesis_markdown = heuristic_synthesis(review)
        else:
            review.synthesis_markdown = self._llm_synthesize(review)
        self._progress("[6/6] Synthesis complete.")
        review.audit.append("Synthesis agent completed.")
        return {"review": review}

    def _build_search_plan(self, config: TopicConfig) -> SearchPlan:
        if isinstance(self.llm, NullProvider):
            return default_search_plan(config)
        self._progress(
            f"[1/6] Calling {self.llm.name} for structured search-plan JSON."
        )
        prompt = f"""
You are a health-sciences librarian and clinical epidemiologist.
Create a reproducible bibliographic search plan for this resident research topic.
Do not invent sources or claims. Return only JSON.

Topic: {config.title}
Clinical context: {config.clinical_context}
Question: {config.question}
Population: {config.population}
Exposure: {config.exposure}
Comparator: {config.comparator}
Outcome: {config.outcome}
Years: {config.year_from}-{config.year_to}
Languages: {", ".join(config.languages)}
Include terms: {", ".join(config.include_terms)}
Exclude terms: {", ".join(config.exclude_terms)}

The PubMed query should use MeSH where useful, plus title/abstract terms.
The Europe PMC query should use Europe PMC search syntax, with FIRST_PDATE and
LANG filters when useful, but no URL parameters such as sort or pageSize.
The Crossref query should be either a simple bibliographic keyword string or
query.bibliographic=<terms>; filters: from-pub-date:YYYY-MM-DD,
until-pub-date:YYYY-MM-DD, type:journal-article.
"""
        return parse_model(self.llm.complete(prompt, schema=SearchPlan), SearchPlan)

    def _llm_screen(self, articles: list[Article], config: TopicConfig) -> list[ScreeningDecision]:
        decisions: list[ScreeningDecision] = []
        chunks = chunked(articles, size=10)
        for index, chunk in enumerate(chunks, start=1):
            self._progress(
                f"[5/6] Calling {self.llm.name} for screening batch {index}/{len(chunks)} "
                f"({len(chunk)} records)."
            )
            payload = [article_for_prompt(article) for article in chunk]
            prompt = f"""
You are screening bibliographic records for a pediatric resident's research project.
Use only the article metadata and abstracts provided below. Do not add external facts.
Return a JSON object with a "decisions" array matching the schema.

Question: {config.question}
Inclusion criteria:
{json.dumps(config.inclusion_criteria, ensure_ascii=False, indent=2)}
Exclusion criteria:
{json.dumps(config.exclusion_criteria, ensure_ascii=False, indent=2)}

Records:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
            batch = parse_model(self.llm.complete(prompt, schema=ScreeningBatch), ScreeningBatch)
            decisions.extend(batch.decisions)
        known_ids = {article.stable_id for article in articles}
        return [decision for decision in decisions if decision.stable_id in known_ids]

    def _llm_synthesize(self, review: ReviewState) -> str:
        top_articles = top_selected_articles(review, limit=8)
        self._progress(
            f"[6/6] Calling {self.llm.name} for final Spanish synthesis "
            f"from {len(top_articles)} selected records."
        )
        prompt = f"""
You are helping a medical resident write the Methods homework for a scientific
research methodology module. Write in Spanish, concise academic style.

Use only the records below and the audit trail. Include:
1. Tema-problema delimitado.
2. Bases consultadas.
3. Palabras clave and boolean strategy.
4. Filtros, embedding prefilter if used, and selection criteria.
5. Brief relevance comments for at least 3 selected articles.
6. A transparent paragraph explaining the agentic AI workflow.

Do not invent PMIDs, DOIs, article counts, or findings beyond the metadata.

Research config:
{review.config.model_dump_json(indent=2)}

Search plan:
{review.plan.model_dump_json(indent=2) if review.plan else ""}

Selected records:
{json.dumps([article_for_prompt(a) for a in top_articles], ensure_ascii=False, indent=2)}

Embedding ranks:
{json.dumps([r.model_dump() for r in review.embedding_ranks[:12]], ensure_ascii=False, indent=2)}

Screening decisions:
{json.dumps([d.model_dump() for d in review.decisions[:12]], ensure_ascii=False, indent=2)}

Audit:
{json.dumps(review.audit, ensure_ascii=False, indent=2)}
"""
        return self.llm.complete(prompt)

    def _progress(self, message: str) -> None:
        self.progress(message)


def default_search_plan(config: TopicConfig) -> SearchPlan:
    years = f'("{config.year_from}"[Date - Publication] : "{config.year_to}"[Date - Publication])'
    screen_terms = [
        '"Screen Time"[Mesh]',
        '"screen time"[Title/Abstract]',
        '"digital media"[Title/Abstract]',
        '"mobile device"[Title/Abstract]',
        'smartphone[Title/Abstract]',
        'tablet[Title/Abstract]',
        'television[Title/Abstract]',
    ]
    language_terms = [
        '"Language Development"[Mesh]',
        '"Language Development Disorders"[Mesh]',
        '"expressive language"[Title/Abstract]',
        '"speech delay"[Title/Abstract]',
        '"language delay"[Title/Abstract]',
        'communication[Title/Abstract]',
    ]
    age_terms = [
        'infant[Title/Abstract]',
        'toddler[Title/Abstract]',
        'preschool[Title/Abstract]',
        '"young children"[Title/Abstract]',
        '"18 months"[Title/Abstract]',
        '"36 months"[Title/Abstract]',
    ]
    pubmed_query = (
        f"({' OR '.join(screen_terms)}) AND ({' OR '.join(language_terms)}) "
        f"AND ({' OR '.join(age_terms)}) AND {years}"
    )
    simple = (
        "(screen time OR digital media OR mobile device OR smartphone OR tablet OR television) "
        "AND (language development OR expressive language OR speech delay OR language delay) "
        "AND (infant OR toddler OR preschool OR young children)"
    )
    return SearchPlan(
        peco_question=config.question,
        concepts=[
            SearchConcept(
                name="population",
                terms=[config.population, "toddler", "young children", "18-36 months"],
            ),
            SearchConcept(
                name="exposure",
                terms=["screen time", "digital media", "mobile device", "television"],
            ),
            SearchConcept(
                name="outcome",
                terms=["expressive language", "speech delay", "language development"],
            ),
        ],
        pubmed_query=pubmed_query,
        europepmc_query=(
            f"{simple} AND "
            f"FIRST_PDATE:[{config.year_from}-01-01 TO {config.year_to}-12-31]"
        ),
        crossref_query=(
            f"query.bibliographic={simple}; filters: "
            f"from-pub-date:{config.year_from}-01-01, "
            f"until-pub-date:{config.year_to}-12-31, type:journal-article"
        ),
        filters=[
            f"Publication year {config.year_from}-{config.year_to}",
            "Humans; early childhood prioritized",
            "English, Spanish, Portuguese",
        ],
        rationale="Default PECO-based strategy generated without an LLM provider.",
    )


def dedupe_articles(articles: list[Article]) -> list[Article]:
    unique: OrderedDict[str, Article] = OrderedDict()
    for article in articles:
        key = article.stable_id
        if key in unique:
            unique[key] = merge_article(unique[key], article)
        else:
            duplicate_key = fuzzy_duplicate_key(article, unique)
            if duplicate_key:
                unique[duplicate_key] = merge_article(unique[duplicate_key], article)
            else:
                unique[key] = article
    return list(unique.values())


def fuzzy_duplicate_key(article: Article, existing: OrderedDict[str, Article]) -> str | None:
    try:
        from rapidfuzz import fuzz
    except Exception:
        return None
    title = article.title.lower()
    if not title:
        return None
    for key, other in existing.items():
        if fuzz.token_set_ratio(title, other.title.lower()) >= 96:
            return key
    return None


def merge_article(primary: Article, secondary: Article) -> Article:
    data = primary.model_dump()
    for field in ["doi", "pmid", "pmcid", "abstract", "journal", "publication_type", "url"]:
        if not data.get(field) and getattr(secondary, field):
            data[field] = getattr(secondary, field)
    if len(secondary.authors) > len(primary.authors):
        data["authors"] = secondary.authors
    data["source"] = ";".join(sorted(set(primary.source.split(";") + secondary.source.split(";"))))
    data["raw"] = {"merged": [primary.raw, secondary.raw]}
    return Article.model_validate(data)


def heuristic_screen(article: Article, config: TopicConfig) -> ScreeningDecision:
    text = " ".join([article.title, article.abstract or "", article.publication_type or ""]).lower()
    include_hits = [term for term in config.include_terms if term.lower() in text]
    exclude_hits = [term for term in config.exclude_terms if term.lower() in text]
    score = min(100, len(include_hits) * 12)
    if article.year and config.year_from <= article.year <= config.year_to:
        score += 10
    if any(term in text for term in ["systematic review", "meta-analysis", "cohort", "cross-sectional"]):
        score += 10
    score = max(0, min(100, score - len(exclude_hits) * 20))
    include = score >= 35 and not exclude_hits
    return ScreeningDecision(
        stable_id=article.stable_id,
        include=include,
        relevance_score=score,
        reasons=[
            f"Matched include terms: {', '.join(include_hits[:8]) or 'none'}",
            f"Matched exclude terms: {', '.join(exclude_hits[:8]) or 'none'}",
        ],
        key_contribution="Potentially relevant to exposure, language, or early-childhood population."
        if include
        else None,
    )


def heuristic_synthesis(review: ReviewState) -> str:
    top = top_selected_articles(review, limit=5)
    lines = [
        "# Sintesis preliminar",
        "",
        f"Tema-problema: {review.config.title}.",
        "",
        "## Proceso de busqueda",
        "",
        f"Bases consultadas: {', '.join(review.config.sources)}.",
        f"Periodo: {review.config.year_from}-{review.config.year_to}.",
        "",
        "## Articulos priorizados",
        "",
    ]
    if not top:
        lines.append("No se seleccionaron articulos con el cribado heuristico.")
    for article in top:
        lines.append(f"- {format_citation(article)}")
    lines.extend(
        [
            "",
            "## Uso de IA",
            "",
            "Se uso un flujo agentico: agente PECO, agente de busqueda, agente de deduplicacion, "
            "agente de cribado y agente de sintesis. En este reporte el cribado fue heuristico; "
            "para una entrega final se recomienda rerun con provider gemini, claude-cli o codex-cli "
            "y verificacion manual de PMID/DOI.",
            "",
            "## Auditoria",
            "",
        ]
    )
    lines.extend([f"- {entry}" for entry in review.audit])
    return "\n".join(lines)


def top_selected_articles(review: ReviewState, *, limit: int) -> list[Article]:
    score_by_id = {decision.stable_id: decision.relevance_score for decision in review.decisions}
    return sorted(
        review.selected_records,
        key=lambda article: score_by_id.get(article.stable_id, 0),
        reverse=True,
    )[:limit]


def article_for_prompt(article: Article) -> dict[str, Any]:
    return {
        "stable_id": article.stable_id,
        "title": article.title,
        "authors": article.authors[:8],
        "year": article.year,
        "journal": article.journal,
        "doi": article.doi,
        "pmid": article.pmid,
        "url": str(article.url) if article.url else None,
        "publication_type": article.publication_type,
        "abstract": article.abstract[:2500] if article.abstract else None,
    }


def format_citation(article: Article) -> str:
    authors = ", ".join(article.authors[:3])
    if len(article.authors) > 3:
        authors += " et al."
    parts = [
        authors or "Authors not available",
        f"({article.year})" if article.year else "(year not available)",
        article.title,
        article.journal or "journal not available",
    ]
    suffix = []
    if article.doi:
        suffix.append(f"doi:{article.doi}")
    if article.pmid:
        suffix.append(f"PMID:{article.pmid}")
    if article.url:
        suffix.append(str(article.url))
    return ". ".join(part for part in parts if part) + ". " + " ".join(suffix)


def chunked(items: list[Any], *, size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _shorten(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _retrieval_summary(record_count: int, hit_count: int | None) -> str:
    if hit_count is None:
        return f"retrieved {record_count} records"
    if record_count >= hit_count:
        return f"retrieved {record_count} records (all {hit_count} source hits)"
    return f"retrieved {record_count} records of {hit_count} source hits"
