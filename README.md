# CodeRescue AI: Unanswered-Question Prediction & Solution Finder

Predicts whether a newly posted Stack Exchange question will remain unanswered after 7 days, retrieves
previously solved threads, and drafts a cited troubleshooting answer.

## Stakeholder
A developer-support platform that wants to spot at-risk questions early and point askers to existing fixes.

## Data
Stack Exchange Data Dump (Posts.xml, Users.xml, PostLinks.xml). Get it from the official data-dump page
(stackoverflow.com, logged in, "Data dump" in your profile settings) or the archive.org mirror of older dumps.
Full Stack Overflow is ~100 GB uncompressed, so filter by tag and date:
```bash
7z x stackoverflow.com-Posts.7z -o data/raw     # also Users and PostLinks
python src/parse_dump.py --tags python pandas --since 2021-01-01
```
A smaller site (e.g. `datascience.stackexchange.com`, `unix.stackexchange.com`) works unchanged without filters.

## Method
| Question | Approach | Evaluation |
|---|---|---|
| Will it be unanswered in 7 days? | Post-time features only (length, code blocks, error text, links, tags' historical rates fitted on train only, asker's prior history computed strictly from the past, account age, hour/weekday) + TF-IDF text. LightGBM, logistic regression, validation-tuned blend | Temporal split 70/15/15. ROC-AUC, PR-AUC with bootstrap CIs, precision/lift in the riskiest 10%, Brier score |
| Can retrieval find relevant solved threads? | BM25 (sparse-matrix implementation), sentence-transformer embeddings, hybrid via reciprocal rank fusion. Corpus = training-period questions with an accepted or upvoted answer | Ground truth = moderator duplicate links (`PostLinks`, type 3) for questions posted after training: Recall@1/5/10, MRR@10 |
| Does retrieval improve LLM answers? | Same LLM with vs without top-5 threads; prompt requires [n] citations | Blind pairwise LLM judge with the accepted answer as reference, random order, sign test + Wilcoxon; citation validity rate |

**Leakage guards:** no score, views, answer count, comments, edits, close status or current reputation; questions
younger than 7 days at dump time are dropped (outcome unknown); tag rates fitted on train only.

## Run
```bash
pip install -r requirements.txt
python src/parse_dump.py [--tags ...] [--since ...]
python src/train.py               # outputs/model_results.json
python src/build_index.py
python src/eval_retrieval.py      # outputs/retrieval_results.json
export ANTHROPIC_API_KEY=...      # optional; enables generation and the next step
python src/eval_llm.py            # outputs/llm_eval_results.json (LLM_EVAL_N=100 by default)
streamlit run app/app.py
python tests/smoke_test.py        # synthetic-data pipeline check
```

## Results (real data; `outputs/*.json`)
Data Science Stack Exchange dump (April 2024): 36,775 questions, 41,470 answers, 179 duplicate links.
- 36,724 labelled questions; 34.4% unanswered after 7 days overall, 50.3% in the test period (newest 15%).
- Time-based test set: blended LightGBM + TF-IDF model ROC-AUC 0.636 (95% CI 0.622-0.651), PR-AUC 0.621 vs 0.503
  base rate; metadata-only 0.614, text-only 0.623. Top-10% riskiest: 67% unanswered (1.34x lift).
- Retrieval, as of posting time (each duplicate searches only solved questions whose question and answer existed
  before it was posted; 144 pairs, 24,000-question corpus): hybrid Recall@10 0.417 (95% CI 0.34-0.50), Recall@1
  0.174, MRR@10 0.238; BM25 0.354; LSA 0.389. Post-training-window subset: only 26 pairs (Recall@10 0.27, too few).
- Question 3 (LLM with vs without retrieval) not run: needs an ANTHROPIC_API_KEY.

## Limitations
Deleted questions are absent from dumps, so the hardest cases are under-represented. "Answered" means any answer,
not a correct one. Duplicate links cover only questions moderators caught. The LLM judge is itself a model and
the generator may have seen Stack Overflow during training; the accepted-answer reference reduces but does not
remove this bias.
