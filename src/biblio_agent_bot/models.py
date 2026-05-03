from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopicConfig(BaseModel):
    project_slug: str
    title: str
    clinical_context: str
    question: str
    population: str
    exposure: str
    comparator: str | None = None
    outcome: str
    setting: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["English", "Spanish", "Portuguese"])
    year_from: int = 2016
    year_to: int = 2026
    max_records_per_source: int = 25
    sources: list[Literal["pubmed", "europepmc", "crossref"]] = Field(
        default_factory=lambda: ["pubmed", "europepmc", "crossref"]
    )
    provider: str = "none"
    model_name: str | None = None
    embedding_provider: str = "none"
    embedding_model: str = "google/gemini-embedding-2-preview"
    embedding_top_k: int = 80
    embedding_batch_size: int = 64
    embedding_min_score: float | None = None
    include_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class SearchConcept(StrictOutputModel):
    name: str
    terms: list[str]


class SearchPlan(StrictOutputModel):
    peco_question: str
    concepts: list[SearchConcept]
    pubmed_query: str
    europepmc_query: str
    crossref_query: str
    filters: list[str]
    rationale: str


class Article(BaseModel):
    source: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    url: HttpUrl | str | None = None
    abstract: str | None = None
    publication_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def stable_id(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.pmcid:
            return f"pmcid:{self.pmcid}"
        return f"title:{self.title.lower()[:120]}"


class ScreeningDecision(StrictOutputModel):
    stable_id: str
    include: bool
    relevance_score: int = Field(ge=0, le=100)
    reasons: list[str] = Field(default_factory=list)
    key_contribution: str | None = None
    limitations: str | None = None


class EmbeddingRank(BaseModel):
    stable_id: str
    rank: int
    score: float
    text_source: str


class ReviewState(BaseModel):
    config: TopicConfig
    plan: SearchPlan | None = None
    raw_records: list[Article] = Field(default_factory=list)
    deduped_records: list[Article] = Field(default_factory=list)
    embedding_ranks: list[EmbeddingRank] = Field(default_factory=list)
    embedding_selected_records: list[Article] = Field(default_factory=list)
    decisions: list[ScreeningDecision] = Field(default_factory=list)
    selected_records: list[Article] = Field(default_factory=list)
    synthesis_markdown: str | None = None
    audit: list[str] = Field(default_factory=list)
