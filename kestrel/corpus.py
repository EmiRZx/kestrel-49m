"""The training mix: which public corpora, in what proportion, and why.

Both the tokenizer trainer and the data preparation script read this, so they cannot disagree about
what the corpus is.

Choices, briefly:

* **FineWeb-Edu** carries the mix. Of the openly available web corpora it gives by far the largest
  gains on exactly the benchmark family this track is scored on (ARC, HellaSwag, PIQA) at small
  model scale, because its educational-quality classifier filters out the web's long tail of
  near-content-free pages. A 49M model cannot afford to spend capacity on junk.
* **Wikipedia** is included deliberately: held-out WikiText-103 perplexity is a scored metric, and
  WikiText-103 *is* Wikipedia. Training on FineWeb alone would report a perplexity that reflects a
  domain mismatch rather than modelling quality. This is not test-set training -- the WikiText-103
  test articles are removed by the decontamination pass in `kestrel.decontaminate`.
* The remaining slot is held open for an ablation (candidates: OpenWebMath, public textbook text).
  Nothing goes into the final mix on the strength of an argument alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    name: str
    path: str                 # Hugging Face dataset id
    split: str = "train"
    config: str | None = None
    text_field: str = "text"
    weight: float = 0.0       # share of the token budget
    license: str = ""
    note: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        name="fineweb-edu",
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
        weight=0.82,
        license="ODC-By 1.0",
        note="educational-quality-filtered CommonCrawl; the benchmark workhorse",
    ),
    Source(
        name="wikipedia",
        path="wikimedia/wikipedia",
        config="20231101.en",
        text_field="text",
        weight=0.18,
        license="CC BY-SA 4.0 / GFDL",
        note="keeps scored WikiText-103 perplexity in-distribution; test articles decontaminated",
    ),
)

# Document separator. A single reserved id is enough: we do not need chat or instruction markers.
EOT_TOKEN = "<|endoftext|>"


def source_by_name(name: str) -> Source:
    for s in SOURCES:
        if s.name == name:
            return s
    raise KeyError(f"unknown source {name!r}; have {[s.name for s in SOURCES]}")


def weights_summary() -> str:
    return ", ".join(f"{s.name} {s.weight:.0%}" for s in SOURCES)
