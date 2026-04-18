# Results & provenance

Every method returns a `Result` — the shared return type across tamga.

## The Result

```python
@dataclass
class Result:
    method_name: str
    params: dict[str, Any]
    values: dict[str, Any]          # JSON-safe (ndarray encoded as {"__ndarray__": ...})
    tables: list[pd.DataFrame]      # exported as parquet
    figures: list[Any]              # matplotlib figures or raw bytes
    provenance: Provenance | None
```

### Persistence

```python
result.save("results/demo/pca")   # writes result.json + table_*.parquet
```

`result.save(directory)` writes:

- `result.json` — `method_name`, `params`, `values` (with numpy encoded), `provenance`
- `table_0.parquet`, `table_1.parquet`, … — one per DataFrame in `tables`
- Figures are deferred to the viz layer (`render_figures.py` per example)

Round-trip with `Result.from_json("results/demo/pca/result.json")`.

## Provenance

Every Result's `.provenance` carries the full reproducibility envelope:

```python
@dataclass
class Provenance:
    tamga_version: str
    python_version: str
    spacy_model: str
    spacy_version: str
    corpus_hash: str
    feature_hash: str | None
    seed: int
    timestamp: datetime
    resolved_config: dict[str, Any]
    # Forensic (all optional):
    questioned_description: str | None
    known_description: str | None
    hypothesis_pair: str | None
    acquisition_notes: str | None
    custody_notes: str | None
    source_hashes: dict[str, str]
```

`Provenance.current(...)` builds a record from the runtime + your inputs.
`Provenance.from_dict(...)` round-trips from a saved `result.json`.

### Reproducibility contract

Two runs of the same `study.yaml` against the same corpus with the same seed produce
**byte-identical** `result.json`. The runner threads `cfg.seed` through:

- numpy's default RNG in any sampling method
- scikit-learn's `random_state` on every stochastic estimator (k-means, LogReg
  cross-validation, RandomForest, …)
- PyMC's `random_seed` on every Bayesian `pm.sample()` call
- The Stratified K-Fold shuffle

Non-determinism would be a bug — please report.

## Loading multi-method runs

`tamga run study.yaml` produces a directory structure:

```
results/demo/
├── resolved_config.json
├── burrows/
│   └── result.json
├── pca/
│   ├── result.json
│   └── figure.png          # if rendered
└── zeta/
    ├── result.json
    ├── table_0.parquet
    └── table_1.parquet
```

`build_report(results/demo, output="report.html")` loads every `result.json` under the
directory and renders a single HTML report.

## Next

- [Forensic toolkit](../forensic/index.md) — where Provenance's forensic fields come
  into play.
