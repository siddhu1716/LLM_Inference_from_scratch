"""Check links against the GitHub Pages artifact, which contains only docs/."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

DOCS = Path(__file__).resolve().parents[1] / "docs"


class PageLinks(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.anchors: set[str] = set()
        self.references: list[str] = []
        self.feed(path.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value and key == "id":
                self.anchors.add(value)
            if value and key in {"href", "src"}:
                self.references.append(value)


def test_static_targets_resolve_inside_pages_artifact() -> None:
    pages = {path: PageLinks(path) for path in DOCS.rglob("*.html")}
    broken = []
    for source, page in pages.items():
        for reference in page.references:
            url = urlsplit(reference)
            if url.scheme or url.netloc:
                continue
            target = (source.parent / unquote(url.path)).resolve() if url.path else source
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(DOCS) or not target.is_file():
                broken.append(f"{source.relative_to(DOCS)} has no deployed target: {reference}")
            elif (
                target in pages
                and url.fragment
                and unquote(url.fragment) not in pages[target].anchors
            ):
                broken.append(f"{source.relative_to(DOCS)} links to a missing anchor: {reference}")
    assert not broken, "\n".join(broken)
