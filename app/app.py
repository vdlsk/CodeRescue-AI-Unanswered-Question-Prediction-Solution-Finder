"""CodeRescue AI demo. Run:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import OUT_DIR  # noqa: E402
from rag import CodeRescue, llm_available  # noqa: E402

st.set_page_config(page_title="CodeRescue AI", layout="wide")


@st.cache_resource
def engine():
    return CodeRescue()


cr = engine()
st.title("CodeRescue AI")
st.caption("Predicts whether a new question is likely to go unanswered for 7 days, finds solved threads, "
           "and drafts a cited troubleshooting answer.")

tab_ask, tab_eval = st.tabs(["Ask", "Model & retrieval evaluation"])

with tab_ask:
    title = st.text_input("Question title", "pandas merge creates duplicate rows")
    body = st.text_area("Question body (HTML or plain text; wrap code in <pre><code>...</code></pre>)",
                        "<p>After merging two dataframes on 'id' I get many more rows than expected.</p>"
                        "<pre><code>df = a.merge(b, on='id')</code></pre>", height=180)
    tags = st.text_input("Tags (space separated)", "python pandas").split()
    if st.button("Analyze", type="primary"):
        p, feats = cr.risk(title, body if "<" in body else f"<p>{body}</p>", tags)
        level = "High" if p >= 0.5 else "Medium" if p >= 0.3 else "Low"
        c1, c2 = st.columns([1, 2])
        c1.metric("Risk of no answer in 7 days", f"{100 * p:.0f}%", level)
        tips = []
        if feats["n_code_blocks"] == 0:
            tips.append("Add a minimal reproducible code sample.")
        if not feats["mentions_error"]:
            tips.append("Paste the exact error message or traceback.")
        if feats["n_unseen_tags"] > 0 or feats["tag_min_popularity"] < 3:
            tips.append("One of your tags is rare; add a broader, popular tag too.")
        if feats["body_words"] < 40:
            tips.append("Describe what you expected and what happened instead.")
        c2.write("**Ways to improve the question**")
        c2.write("\n".join(f"- {t}" for t in tips) or "- Looks well-formed.")

        with st.spinner("Retrieving similar solved threads..."):
            out = cr.answer(title, body if "<" in body else f"<p>{body}</p>", use_retrieval=True)
        st.subheader("Suggested fix")
        st.markdown(out["answer"])
        if not llm_available():
            st.caption("No ANTHROPIC_API_KEY set: showing the top retrieved answer instead of a generated one.")
        st.subheader("Evidence")
        for r in out["hits"].itertuples():
            st.markdown(f"**[{r.rank}] [{r.title}]({r.url})** · tags: {r.tags.replace('|', ', ')} · "
                        f"{'accepted' if r.is_accepted else 'top'} answer score {r.answer_score}")

with tab_eval:
    for name in ("model_results.json", "retrieval_results.json", "llm_eval_results.json"):
        p = OUT_DIR / name
        st.subheader(name.replace("_", " ").replace(".json", ""))
        st.json(json.loads(p.read_text())) if p.exists() else st.info(f"Run the script that writes {name}.")
