# Echo Chamber Analysis on Reddit

Do political subreddits form measurable echo chambers? We tested this across 7 political
subreddits and roughly 2.3M posts and comments, using Apache Spark for distributed
processing and GDELT global news data as an external signal.

Course project for **EAS 587**, University at Buffalo (September - December 2025).

## Findings

- **Subreddit is predictable from writing style alone.** Spark MLlib classifiers reached
  approximately **60% accuracy** at picking which of the 7 subreddits a post or comment came
  from, against a **14% random baseline** for 7 classes. Communities have distinct
  linguistic fingerprints.
- **Sentiment divergence with world events.** Cross-referencing GDELT (150K+ events per day)
  showed that more negative global news correlated with *higher* Reddit engagement, not lower.
- **Pandas silently accepted corrupt data that Spark caught.** 

## The data problem

The raw export arrived as JSONL and was converted to **14 CSVs** (7 subreddits x posts and
comments). Several files had **column-shifting corruption**: embedded delimiters and newlines
in post bodies pushed fields into the wrong columns on some rows.

Pandas parsed these rows without complaint and produced silently wrong values. We added
explicit schema validation and a row-level data firewall in the Spark ingest layer, which
rejected **100% of the malformed rows** into a quarantine path instead of letting them reach
the models.

Subreddits covered: `r/Conservative`, `r/Libertarian`, `r/PoliticalDiscussion`,
`r/neutralnews`, `r/politics`, `r/socialism`, `r/worldnews`.

## Pipeline

A Medallion architecture in Apache Spark:

| Layer | Contents |
|---|---|
| **Bronze** | Raw JSONL converted to CSV, no transformation, full fidelity |
| **Silver** | Schema-validated and cleaned rows; malformed rows quarantined |
| **Gold** | Feature-engineered, analysis-ready tables for MLlib and reporting |

Models are trained with cross-validation and tuned with **Optuna**. Classifiers compared:
logistic regression, linear SVM, and naive Bayes.

## MCP server

`mcp_server.py` wraps the trained Spark pipelines as a **Model Context Protocol** server, so
an LLM can query the models and the dataset directly as callable tools. It exposes five:

| Tool | Purpose |
|---|---|
| `predict_subreddit_from_title` | Classify a post title into one of the 7 subreddits |
| `predict_subreddit_from_comment` | Same, for comment text |
| `get_subreddit_stats` | Summary statistics for one subreddit |
| `compare_keyword_frequency` | Compare a keyword's frequency across subreddits |
| `get_global_news_sentiment` | GDELT sentiment for a given date |

On startup the server initializes a Spark session and loads both `PipelineModel` artifacts
plus the posts and GDELT frames into memory, then reports readiness. Loading is **eager**, so
the first tool call responds without warm-up cost; the trade-off is a slower boot and a
larger resident memory footprint.

## Repository layout

```
mcp_server.py                  MCP server exposing the trained models as LLM tools
jsonl_to_csv.py                Raw JSONL -> CSV conversion
csv_cleanup.py                 Repairs column-shifted rows
csv_analysis.py                Column and schema inspection
create_sample.py               Builds the small committed sample set

phase_1_eda.ipynb              Exploratory data analysis
phase_2_ml_analysis.ipynb      First ML pass and baselines
phase_3_data_pipeline.ipynb    Medallion pipeline (Bronze/Silver/Gold)
phase_3_mllib_models.ipynb     Spark MLlib classifiers and tuning
phase_3_comments_model.ipynb   Comment-level classifier
phase_3_multisource.ipynb      GDELT integration and sentiment analysis

data_sample/original_jsonl/    Sample of the raw JSONL input
data_sample/cleaned_csv/       Same records after cleaning
Dockerfile, docker-compose.yml Spark environment
hadoop/bin/                    winutils.exe and hadoop.dll for local Spark on Windows
```

The full dataset is not committed. `data_sample/` holds a small slice so the notebooks can be
run end to end without it.

## Running it

Bring up the Spark environment:

```bash
docker compose up -d
```

Install Python dependencies and run the notebooks in phase order:

```bash
pip install -r requirements.txt
```

Start the MCP server:

```bash
python mcp_server.py
```

## Known limitations

- `mcp_server.py` currently hard-codes macOS paths for `JAVA_HOME` and the PySpark
  interpreter. It needs those values parameterized before it will run elsewhere.
- The server expects `data_cleaned/`, `models/`, and `gdelt_data/` beside the script. These
  are produced by the phase 3 notebooks and are not committed.
- Spark requires Java 17. Java 21+ will crash the session.

## Team and contributions

Most of this project was built in pair-programming sessions on a shared machine, so **commit
counts do not reflect how the work was divided.** Contributions were:

**Akash Kamble** ([@kambleakash0](https://github.com/kambleakash0)) - repository owner
- Project setup, repository scaffolding, and dependency management
- Reddit data collection across the 7 subreddits and the raw JSONL export
- Reddit and GDELT data loading and filtering
- Ingest and cleaning scripts: `jsonl_to_csv.py`, `csv_cleanup.py`, `csv_analysis.py`
- `create_sample.py` and the committed sample dataset under `data_sample/`
- Phase 1 exploratory data analysis
- Phase 2 ML analysis and the baseline classifiers that phase 3 was measured against
- Phase 3 notebook structure and scaffolding
- Docker Compose and the Hadoop/winutils environment for running Spark locally on Windows

**Anirudh Raj Sharma** ([@DEZ-byte](https://github.com/DEZ-byte))
- MCP server (`mcp_server.py`) and its five tool endpoints
- Spark session initialization and logging configuration
- Cross-validation training loop and Optuna hyperparameter tuning
- Feature engineering stages for the MLlib pipelines

## License

MIT. See [LICENSE](LICENSE).
