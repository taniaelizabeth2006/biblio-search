from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from Bio import Entrez, Medline
from habanero import Crossref
from tenacity import retry, stop_after_attempt, wait_exponential

from biblio_agent_bot.models import Article


@dataclass(frozen=True)
class CrossrefRequest:
    query: str | None
    filters: dict[str, str]
    query_kwargs: dict[str, str]


class SourceClient(ABC):
    name: str
    last_hit_count: int | None = None

    @abstractmethod
    def search(self, query: str, *, max_records: int) -> list[Article]:
        ...


class PubMedClient(SourceClient):
    name = "pubmed"

    def __init__(self) -> None:
        email = os.getenv("NCBI_EMAIL")
        if not email:
            raise RuntimeError("NCBI_EMAIL is required for PubMed searches")
        Entrez.email = email
        Entrez.tool = "biblio-agent-bot"
        api_key = os.getenv("NCBI_API_KEY")
        if api_key:
            Entrez.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, *, max_records: int) -> list[Article]:
        self.last_hit_count = None
        with Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_records,
            sort="relevance",
            retmode="xml",
        ) as handle:
            search_result = Entrez.read(handle)
        self.last_hit_count = _safe_int(search_result.get("Count"))
        pmids = search_result.get("IdList", [])
        if not pmids:
            return []
        records = []
        for batch in _chunks(pmids, size=200):
            with Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text",
            ) as handle:
                records.extend(Medline.parse(handle))
        return [_article_from_medline(record) for record in records]


class EuropePMCClient(SourceClient):
    name = "europepmc"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, *, max_records: int) -> list[Article]:
        self.last_hit_count = None
        results = []
        cursor_mark = "*"
        with httpx.Client(timeout=30) as client:
            while len(results) < max_records:
                page_size = min(1000, max_records - len(results))
                params = {
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": page_size,
                    "cursorMark": cursor_mark,
                }
                response = client.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
                if self.last_hit_count is None:
                    self.last_hit_count = _safe_int(payload.get("hitCount"))
                page = payload.get("resultList", {}).get("result", [])
                results.extend(page)
                next_cursor = payload.get("nextCursorMark")
                if not page or not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor
        return [_article_from_europepmc(item) for item in results]


class CrossrefClient(SourceClient):
    name = "crossref"

    def __init__(self) -> None:
        self.client = Crossref(mailto=os.getenv("NCBI_EMAIL"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, *, max_records: int) -> list[Article]:
        self.last_hit_count = None
        request = _crossref_request_from_query(query)
        page_size = min(max_records, 1000)
        cursor = "*" if max_records > page_size else None
        response = self.client.works(
            query=request.query,
            filter=request.filters,
            limit=page_size,
            sort="relevance",
            cursor=cursor,
            cursor_max=max_records,
            **request.query_kwargs,
        )
        pages = response if isinstance(response, list) else [response]
        items: list[dict[str, Any]] = []
        for page in pages:
            message = page.get("message", {})
            if self.last_hit_count is None:
                self.last_hit_count = _safe_int(message.get("total-results"))
            items.extend(message.get("items", []))
            if len(items) >= max_records:
                break
        return [_article_from_crossref(item) for item in items[:max_records]]


def source_client(name: str) -> SourceClient:
    if name == "pubmed":
        return PubMedClient()
    if name == "europepmc":
        return EuropePMCClient()
    if name == "crossref":
        return CrossrefClient()
    raise ValueError(f"Unknown source: {name}")


def _crossref_request_from_query(query: str) -> CrossrefRequest:
    text = " ".join(query.split())
    filters = _crossref_filters_from_query(text)

    search_text = text.split(";", 1)[0].strip()
    bibliographic_match = re.match(
        r"query[._-]bibliographic\s*=\s*(?P<value>.+)\Z",
        search_text,
        flags=re.IGNORECASE,
    )
    if bibliographic_match:
        return CrossrefRequest(
            query=None,
            filters=filters,
            query_kwargs={
                "query_bibliographic": bibliographic_match.group("value").strip(),
            },
        )
    return CrossrefRequest(query=search_text or None, filters=filters, query_kwargs={})


def _crossref_filters_from_query(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    aliases = {
        "from-pub-date": "from_pub_date",
        "until-pub-date": "until_pub_date",
        "type": "type",
    }
    for raw_name, value in re.findall(
        r"\b(from-pub-date|until-pub-date|type)\s*:\s*([^,;]+)",
        query,
        flags=re.IGNORECASE,
    ):
        filters[aliases[raw_name.lower()]] = value.strip()
    return filters


def _article_from_medline(record: dict) -> Article:
    year = _year(record.get("DP"))
    doi = None
    for aid in record.get("AID", []):
        if "[doi]" in aid.lower():
            doi = aid.split()[0]
            break
    pmid = str(record.get("PMID")) if record.get("PMID") else None
    return Article(
        source="pubmed",
        title=record.get("TI", "").strip(),
        authors=record.get("AU", []),
        year=year,
        journal=record.get("JT") or record.get("TA"),
        doi=doi,
        pmid=pmid,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        abstract=record.get("AB"),
        publication_type="; ".join(record.get("PT", [])) or None,
        raw=record,
    )


def _article_from_europepmc(item: dict) -> Article:
    pmid = item.get("pmid")
    pmcid = item.get("pmcid")
    doi = item.get("doi")
    return Article(
        source="europepmc",
        title=item.get("title", "").strip(),
        authors=_split_authors(item.get("authorString")),
        year=_year(item.get("pubYear")),
        journal=item.get("journalTitle"),
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        url=_best_url(doi=doi, pmid=pmid, pmcid=pmcid),
        abstract=item.get("abstractText"),
        publication_type=item.get("pubType"),
        raw=item,
    )


def _article_from_crossref(item: dict) -> Article:
    title = " ".join(item.get("title", [])).strip()
    authors = [
        " ".join(part for part in [a.get("given"), a.get("family")] if part)
        for a in item.get("author", [])
    ]
    date_parts = (
        item.get("published-print", {}).get("date-parts")
        or item.get("published-online", {}).get("date-parts")
        or item.get("created", {}).get("date-parts")
        or [[]]
    )
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    doi = item.get("DOI")
    return Article(
        source="crossref",
        title=title,
        authors=authors,
        year=year,
        journal="; ".join(item.get("container-title", [])) or None,
        doi=doi,
        url=f"https://doi.org/{doi}" if doi else item.get("URL"),
        abstract=item.get("abstract"),
        publication_type="; ".join(item.get("type", "").split("-")) or None,
        raw=item,
    )


def _split_authors(author_string: str | None) -> list[str]:
    if not author_string:
        return []
    return [author.strip() for author in author_string.split(",") if author.strip()]


def _year(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value)
    for token in text.replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _best_url(*, doi: str | None, pmid: str | None, pmcid: str | None) -> str | None:
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    return None


def _chunks(items: list[str], *, size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
