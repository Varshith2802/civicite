"""Tokenisation, stop words and a tiny bilingual (English/Swedish) suffix stemmer."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\d+(?:[.,:/-]\d+)*|[a-zåäöéüæø]+", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ0-9\"'(\[])|\n{2,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")

STOP_EN = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with you your yours yourself
yourselves also may must can need needs get gets got one use used using via per etc anything something
anyone someone does""".split())
STOP_SV = set("""och i att det som en på är av för med till den har de inte om ett han men var jag sig från
vi så kan man när år säger hon under också efter eller nu sin där vid mot ska skulle kommer ut får finns
vara hade alla andra mycket än här då sedan över bara blir upp även vad få två vill ha många hur mer går
sverige du dig din ditt dina mig min mitt mina vår vårt våra er ert era dem denna detta dessa vilken vilket
vilka hos utan göra gör gjort""".split())
STOPWORDS = STOP_EN | STOP_SV

_SUFFIXES = ("arnas", "ernas", "ornas", "arna", "erna", "orna", "ande", "ende", "ings", "ing",
             "ed", "es", "en", "et", "er", "ar", "or", "na", "s", "e")


def stem(tok: str) -> str:
    """Very small English/Swedish suffix stripper (good enough for matching inflections)."""
    if tok.isdigit() or len(tok) <= 4:
        return tok
    if tok.endswith(("ies", "ied")) and len(tok) > 5:
        return tok[:-3] + "y"
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            break
    if tok.endswith("e") and len(tok) > 5:
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def content_terms(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed terms used for retrieval and verification."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p)
        if p:
            out.append(p)
    return out


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM.findall(text)}


#: Swedish -> English glossary for query expansion (the demo corpus is in English with Swedish terms).
#: Multilingual dense retrieval (``civicite index --dense``) covers the general case.
GLOSSARY_SV_EN = {
    "flytt": "move moving", "flytta": "move", "flyttat": "moved", "flyttanmälan": "move report",
    "anmäla": "report", "anmälan": "report", "tid": "time deadline", "lång": "long", "senast": "deadline later",
    "dag": "day", "dagar": "days", "vecka": "week", "veckor": "weeks", "månad": "month", "månader": "months",
    "barn": "child", "barnet": "child", "föräldrar": "parents", "förälder": "parent",
    "föräldrapenning": "parental benefit", "föräldraledig": "parental leave", "sjuk": "sick ill",
    "sjukpenning": "sickness benefit", "sjuklön": "sick pay", "läkarintyg": "medical certificate",
    "skatt": "tax", "deklaration": "tax return", "inkomstdeklaration": "income tax return", "deklarera": "file tax return",
    "ansöka": "apply", "ansöker": "apply", "ansökan": "application apply", "kostar": "cost free", "kostnad": "cost",
    "gratis": "free", "legitimation": "identity card", "id-kort": "identity card", "körkort": "driving licence",
    "studier": "studies", "studera": "study", "lån": "loan", "bidrag": "grant", "arbetslös": "unemployed",
    "jobb": "job", "arbete": "work job", "arbetsgivare": "employer", "betala": "pay", "betalar": "pays",
    "svenska": "swedish", "kurs": "course", "personnummer": "personal identity number",
    "samordningsnummer": "coordination number", "vabba": "vab sick child", "hur": "", "många": "many",
    # a few English synonyms that matter for public-service questions
    "cost": "fee free", "costs": "fee free", "price": "fee cost", "charge": "fee", "charged": "fee",
}


def expand_query(text: str) -> str:
    extra = []
    for tok in tokenize(text):
        if tok in GLOSSARY_SV_EN and GLOSSARY_SV_EN[tok]:
            extra.append(GLOSSARY_SV_EN[tok])
    return text if not extra else text + " " + " ".join(extra)
