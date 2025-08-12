# FPL Chat Agent — Streamlit + LangChain + Full Live KB
# -----------------------------------------------------
# - Uses official FPL endpoints to build a FULL knowledge base (players + fixtures)
# - Optionally enriches players with recent (last N) gameweek history
# - LangChain ConversationChain for chat with session memory
# - Injects the FULL KB once (system prompt) so every turn sees the full dataset
# - Users must provide their own OpenAI API key (stored per-session only)

import os
import hashlib
from datetime import datetime
from functools import lru_cache

import pandas as pd
import pytz
import requests
import streamlit as st

# LangChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain

# ---------------- Config ----------------
TZ = pytz.timezone("Europe/London")
FPL_API = "https://fantasy.premierleague.com/api"
MODEL_NAME = os.getenv("FPL_LLM_MODEL", "gpt-5-mini")  # faster by default
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

st.set_page_config(page_title="FPL Chat Agent", page_icon="⚽", layout="wide")
st.title("⚽ FPL Chat Agent")

# --------------- Sidebar: API key & options ---------------
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

with st.sidebar:
    st.subheader("🔐 API & Options")
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        value=st.session_state.openai_key,
        type="password",
        key="openai_api_key_input",
        help="Stored only in your session.",
    )
    if api_key_input:
        st.session_state.openai_key = api_key_input.strip()

    include_hist = st.checkbox("Include recent player history", value=True)
    last_n = st.slider("Recent GWs", 3, 8, 5, 1)
    if st.button("🔄 Refresh live KB"):
        st.session_state.pop("full_kb", None)
        st.session_state.pop("kb_meta", None)
        st.session_state.pop("conversation", None)
        st.session_state.pop("kb_hash", None)

    st.caption(f"API key present: {'Yes' if st.session_state.openai_key else 'No'}")


# --------------- FPL fetchers ---------------
@lru_cache(maxsize=8)
def fetch_bootstrap():
    r = requests.get(f"{FPL_API}/bootstrap-static/")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=8)
def fetch_fixtures():
    r = requests.get(f"{FPL_API}/fixtures/")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=512)
def fetch_player_history(player_id: int):
    r = requests.get(f"{FPL_API}/element-summary/{player_id}/")
    r.raise_for_status()
    return r.json()


# --------------- KB builder ---------------


def _recent_block(pid: int, last_n: int = 5):
    """Compact recent summary for the last N matches."""
    try:
        h = fetch_player_history(pid)
        hist = h.get("history", [])[-last_n:]
        if not hist:
            return "RECENT: n/a"
        pts = [int(g.get("total_points", 0)) for g in hist]
        mins = [int(g.get("minutes", 0)) for g in hist]
        goals = sum(int(g.get("goals_scored", 0)) for g in hist)
        assists = sum(int(g.get("assists", 0)) for g in hist)
        cs = sum(int(g.get("clean_sheets", 0)) for g in hist)
        last_pts = ",".join(map(str, pts))
        avg = round(sum(pts) / len(pts), 2)
        m90 = round(sum(mins) / 90.0, 1)
        return f"RECENT({len(pts)}): pts[{last_pts}] | avg {avg} | mins/90 {m90} | G{goals} A{assists} CS{cs}"
    except Exception:
        return "RECENT: n/a"


def build_full_kb(include_history: bool = True, last_n: int = 5):
    """Build a single, full knowledge base string + meta + DataFrames for UI."""
    bs = fetch_bootstrap()
    fixtures = fetch_fixtures()

    events = pd.DataFrame(bs.get("events", []))
    players = pd.DataFrame(bs.get("elements", []))
    teams = pd.DataFrame(bs.get("teams", []))

    team_short = teams.set_index("id")["short_name"].to_dict()

    # Players
    players = players.copy()
    players["team_short"] = players["team"].map(team_short)
    players["price"] = players["now_cost"] / 10.0
    players["pos"] = players["element_type"].map(POS)
    # selected_by_percent is often a string, normalize -> float
    players["selected_by"] = pd.to_numeric(
        players.get("selected_by_percent", 0), errors="coerce"
    ).fillna(0.0)

    cols = [
        "id",
        "web_name",
        "team_short",
        "pos",
        "price",
        "form",
        "selected_by",
        "status",
        "news",
        "minutes",
        "points_per_game",
        "total_points",
        "ict_index",
    ]
    keep = [c for c in cols if c in players.columns]
    p_lines = []
    for _, r in players[keep].iterrows():
        pid = int(r.get("id"))
        base = (
            f"PLAYER: {r.get('web_name')} | TEAM: {r.get('team_short')} | POS: {r.get('pos')} | "
            f"PRICE: £{float(r.get('price', 0)):.1f}m | FORM: {r.get('form')} | OWN: {float(r.get('selected_by', 0)):.1f}% | "
            f"PPG: {r.get('points_per_game')} | TOT: {r.get('total_points')} | MINS: {r.get('minutes')} | "
            f"ICT: {r.get('ict_index')} | STATUS: {r.get('status')} | NEWS: {str(r.get('news') or '')[:120]}"
        )
        if include_history:
            base = base + " | " + _recent_block(pid, last_n=last_n)
        p_lines.append(base)

    # Fixtures (next 6 per team)
    fx = pd.DataFrame(fixtures)
    fx = fx[fx["finished"] == False].copy()
    team_fx_lines = []
    if not fx.empty:
        for tid in sorted(teams["id"].tolist()):
            sub = (
                fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)]
                .sort_values("kickoff_time")
                .head(last_n)
            )
            if sub.empty:
                continue
            parts = []
            for _, g in sub.iterrows():
                is_home = g["team_h"] == tid
                opp = g["team_a"] if is_home else g["team_h"]
                fdr = g["team_h_difficulty"] if is_home else g["team_a_difficulty"]
                opps = team_short.get(int(opp), str(opp))
                gw = g.get("event")
                parts.append(f"GW{gw} {'vs' if is_home else '@'} {opps} (FDR {fdr})")
            team_fx_lines.append(
                f"TEAM_FIX: {team_short.get(tid, str(tid))} → " + "; ".join(parts)
            )

    # Header/meta
    gw_now = None
    if not events.empty:
        if "is_current" in events.columns and events["is_current"].any():
            gw_now = int(events.loc[events["is_current"] == True, "id"].iloc[0])
        else:
            upcoming = events[events["finished"] == False].sort_values("deadline_time")
            if not upcoming.empty:
                gw_now = int(upcoming["id"].iloc[0])

    header = f"KB_BUILT: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} | CURRENT_GW: {gw_now} | PLAYERS: {len(p_lines)}"
    full_kb = (
        f"{header}\n\n[FIXTURES]\n"
        + "\n".join(team_fx_lines)
        + "\n\n[PLAYERS]\n"
        + "\n".join(p_lines)
    )

    # return full text + meta + data for UI tables
    return full_kb, {"gw": gw_now, "players": len(p_lines),"kb_built": header}, players, team_fx_lines


# --------------- Build / refresh FULL KB ---------------
if "full_kb" not in st.session_state:
    full_kb, kb_meta, players_df, fixtures_text = build_full_kb(last_n=last_n)
    st.session_state.full_kb = full_kb
    st.session_state.kb_meta = kb_meta
    st.session_state.players_df = players_df
    st.session_state.fixtures_text = fixtures_text
    st.session_state.kb_hash = hashlib.sha256(full_kb.encode("utf-8")).hexdigest()
else:
    # Use existing unless user refreshed options
    full_kb, kb_meta, players_df, fixtures_text = (
        st.session_state.full_kb,
        st.session_state.kb_meta,
        st.session_state.players_df,
        st.session_state.fixtures_text,
    )

st.caption(kb_meta['header'])

# --------------- Stats Tabs ---------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Top 20 Overall",
        "Top 10 by Position",
        "Top Budget Picks",
        "Fixtures",
        "Chat Agent",
    ]
)

with tab1:
    st.subheader("Top 20 Players by Ownership")
    st.dataframe(
        players_df.sort_values("selected_by", ascending=False).head(20)[
            ["web_name", "team_short", "pos", "price", "form", "selected_by"]
        ]
    )

with tab2:
    st.subheader("Top 10 by Position (Ownership)")
    for pos_name in ["GK", "DEF", "MID", "FWD"]:
        st.markdown(f"**{pos_name}**")
        st.dataframe(
            players_df[players_df["pos"] == pos_name]
            .sort_values("selected_by", ascending=False)
            .head(10)[["web_name", "team_short", "price", "form", "selected_by"]]
        )

with tab3:
    st.subheader("Top Budget Picks (≤ £5.0m)")
    st.dataframe(
        players_df[players_df["price"] <= 5.0]
        .sort_values("selected_by", ascending=False)
        .head(15)[["web_name", "team_short", "pos", "price", "form", "selected_by"]]
    )

with tab4:
    st.subheader("Upcoming Fixtures by Team")
    for row in fixtures_text:
        st.write(row)

# --------------- Chat with LangChain (FULL KB injected once) ---------------
with tab5:
    st.subheader("💬 Chat with the FPL Agent")

    # Build/rebuild conversation chain when API key or KB changes
    def _make_chain(api_key: str, kb_text: str):
        llm = ChatOpenAI(openai_api_key=api_key, model_name=MODEL_NAME, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an FPL expert with access to a knowledge base of real-time player stats and upcoming fixtures.
                Use only this knowledge base to provide data-driven advice. Do not make up facts or speculate about player performance.Give fast, data-driven answers.
                Core duties: Recommend team, transfers , captain and chip strategies based on current form and fixtures. Weigh template vs differential picks.Use injury and rotation data.
                Rules: Specific player picks, price and reasoning. Max 3 players per club, budget limits. Include set pieces, penalities and bonus points. EPL data only
                Goal: Maximize points with optimal team selection and transfers.
                Response format: Give specific player recommendations with prices and reasoning. Reference current game week fixtures and upcoming runs.""",
                ),
                ("system", kb_text),  # FULL KB injected here once
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

    api_key = st.session_state.openai_key

    kb_hash = hashlib.sha256(st.session_state.full_kb.encode("utf-8")).hexdigest()
    if (
        ("conversation" not in st.session_state)
        or (st.session_state.get("kb_hash") != kb_hash)
        or st.button("Rebuild chat with current KB")
    ):
        if api_key:
            st.session_state.conversation = _make_chain(
                api_key, st.session_state.full_kb
            )
            st.session_state.kb_hash = kb_hash
        else:
            st.info("Enter your API key in the sidebar to enable chat.")

    # Render prior messages from chain memory (if exists)
    if "conversation" in st.session_state:
        mem_msgs = st.session_state.conversation.memory.chat_memory.messages
        for m in mem_msgs:
            role = "user" if m.type == "human" else "assistant"
            st.chat_message(role).write(m.content)

    # Chat input
    user_input = st.chat_input(
        "Ask about FPL (e.g., best £6.5m mids, who to captain, wildcard draft)..."
    )
    if user_input:
        st.chat_message("user").write(user_input)
        if not api_key:
            assistant_reply = (
                "Please enter your OpenAI API key in the sidebar to use the chat agent."
            )
        elif "conversation" not in st.session_state:
            assistant_reply = "Chat is not initialized. Click 'Rebuild chat with current KB' after entering your API key."
        else:
            with st.spinner("Thinking..."):
                try:
                    assistant_reply = st.session_state.conversation.predict(
                        input=user_input
                    )
                except Exception as e:
                    assistant_reply = f"Error: {e}"
        st.chat_message("assistant").write(assistant_reply)
