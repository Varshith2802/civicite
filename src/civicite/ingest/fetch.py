"""Polite crawler that turns official agency web pages into Markdown documents.

    civicite fetch --seed https://www.skatteverket.se/privat --prefix https://www.skatteverket.se/privat \
                   --agency "Skatteverket" --max-pages 150 --out data/skatteverket

* respects robots.txt (urllib.robotparser) and waits --delay seconds between requests
* stays inside the given URL prefix(es), skips binaries, de-duplicates by content hash
* writes one Markdown file per page with front matter (title, agency, url, fetched)
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from .loader import content_hash, html_to_markdown

UA = "CiviCite/0.1 (+research prototype; contact: see README)"
_BINARY = re.compile(r"\.(pdf|zip|jpg|jpeg|png|gif|svg|mp4|mp3|docx?|xlsx?|pptx?|ics|css|js)(\?|$)", re.I)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)


def _slug(url: str) -> str:
    p = urllib.parse.urlparse(url)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (p.netloc + p.path).strip("/")).strip("-").lower()
    return (s[:120] or "index")


def crawl(seeds: list[str], prefixes: list[str], out_dir: Path, agency: str, max_pages: int = 100,
          delay: float = 1.0, timeout: float = 20.0, verbose: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    robots: dict[str, urllib.robotparser.RobotFileParser] = {}
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    queue = deque(seeds)
    written: list[Path] = []

    def allowed(url: str) -> bool:
        host = urllib.parse.urlparse(url)
        base = f"{host.scheme}://{host.netloc}"
        if base not in robots:
            rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # type: ignore[assignment]
            robots[base] = rp  # type: ignore[assignment]
        rp = robots[base]
        return True if rp is None else rp.can_fetch(UA, url)

    while queue and len(written) < max_pages:
        url = urllib.parse.urldefrag(queue.popleft())[0]
        if url in seen_urls or _BINARY.search(url) or not any(url.startswith(p) for p in prefixes):
            continue
        seen_urls.add(url)
        if not allowed(url):
            if verbose:
                print(f"robots.txt disallows {url}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "sv,en;q=0.8"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if "html" not in r.headers.get("Content-Type", ""):
                    continue
                html = r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")
        except Exception as e:
            if verbose:
                print(f"skip {url}: {e}")
            continue
        finally:
            time.sleep(delay)
        title, body = html_to_markdown(html)
        h = content_hash(body)
        if len(body.split()) >= 60 and h not in seen_hashes:
            seen_hashes.add(h)
            path = out_dir / f"{_slug(url)}.md"
            safe_title = (title or url).replace("\n", " ").replace('"', "'")
            path.write_text(
                f"---\ntitle: {safe_title}\nagency: {agency}\nurl: {url}\nfetched: {date.today()}\n---\n\n{body}\n",
                encoding="utf-8")
            written.append(path)
            if verbose:
                print(f"[{len(written)}/{max_pages}] {url}")
        lp = _Links()
        lp.feed(html)
        for href in lp.links:
            nxt = urllib.parse.urljoin(url, href)
            if nxt.startswith("http") and nxt not in seen_urls:
                queue.append(nxt)
    return written
