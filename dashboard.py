"""
Streamlit monitoring dashboard for the Wine Sommelier RAG system.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "wine_assistant"))
load_dotenv()

from db_query import (
    get_conversations,
    get_requests_per_day,
    get_search_method_stats,
    get_relevance_stats,
    get_stats,
    get_user_feedback_stats,
)

st.set_page_config(page_title="Wine Assistant Dashboard", page_icon="📊", layout="wide")
st.title("📊 Wine Sommelier — Monitoring Dashboard")

if st.button("🔄 Refresh"):
    st.rerun()

stats = get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats["total"])
col2.metric("Avg response time", f"{stats['avg_response_time']:.2f}s")
col3.metric("Total cost", f"${stats['total_cost']:.4f}")
col4.metric("Avg tokens / query", f"{stats['avg_tokens']:.0f}")

st.divider()

# Chart 1: Requests per day
st.subheader("Requests per day (last 14 days)")
rpd = get_requests_per_day(days=14)
if rpd:
    rpd_df = pd.DataFrame(rpd, columns=["day", "count"])
    rpd_df["day"] = pd.to_datetime(rpd_df["day"])
    st.bar_chart(rpd_df.set_index("day")["count"])
else:
    st.info("No data yet.")

col_left, col_right = st.columns(2)

# Chart 2: Cost over time
with col_left:
    st.subheader("Cost over time")
    convs = get_conversations(limit=200)
    if convs:
        df = pd.DataFrame(
            [{"timestamp": c.timestamp, "cost": c.cost, "response_time": c.response_time}
             for c in convs]
        )
        df = df.sort_values("timestamp")
        st.line_chart(df.set_index("timestamp")["cost"])
    else:
        st.info("No data yet.")

# Chart 3: Response time over time
with col_right:
    st.subheader("Response time over time (s)")
    if convs:
        st.line_chart(df.set_index("timestamp")["response_time"])
    else:
        st.info("No data yet.")

col_left2, col_right2 = st.columns(2)

# Chart 4: Relevance distribution
with col_left2:
    st.subheader("Answer relevance (LLM judge)")
    relevance = get_relevance_stats()
    if relevance:
        rel_df = pd.DataFrame(list(relevance.items()), columns=["relevance", "count"])
        st.bar_chart(rel_df.set_index("relevance")["count"])
    else:
        st.info("No feedback yet.")

# Chart 5: User feedback
with col_right2:
    st.subheader("User feedback")
    thumbs_up, thumbs_down = get_user_feedback_stats()
    fb_df = pd.DataFrame(
        {"type": ["👍 Thumbs up", "👎 Thumbs down"], "count": [thumbs_up, thumbs_down]}
    )
    st.bar_chart(fb_df.set_index("type")["count"])

# Chart 6: Search method usage
st.subheader("Search method usage")
sm_stats = get_search_method_stats()
if sm_stats:
    sm_df = pd.DataFrame(sm_stats, columns=["method", "count", "avg_time"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.bar_chart(sm_df.set_index("method")["count"])
    with col_b:
        st.bar_chart(sm_df.set_index("method")["avg_time"])
else:
    st.info("No data yet.")

st.divider()
st.subheader("Recent conversations")
recent = get_conversations(limit=20)
for c in recent:
    with st.expander(f"[{c.timestamp.strftime('%Y-%m-%d %H:%M')}] {c.question[:80]}"):
        st.write(c.answer[:300] + ("..." if len(c.answer) > 300 else ""))
        st.caption(
            f"Method: {c.search_method} | Style: {c.prompt_style} | "
            f"Time: {c.response_time:.2f}s | Cost: ${c.cost:.5f}"
        )
