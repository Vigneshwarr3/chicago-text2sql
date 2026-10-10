"""
sql_agent_pipeline.py
=====================
The two LangGraph agents compared in the benchmark. The query-generation system
prompt is a parameter, so swapping the prompt (prompt_stages.py) is the only
thing that changes between stages.

  agent_1   schema fetch -> generate SQL -> run -> summarize
            (no relevance check, no validation, no retry)
  agent_2   relevance check -> schema fetch -> generate SQL -> EXPLAIN-validate
            (up to 3 regeneration attempts) -> run -> summarize

Usage
-----
    from langchain_groq import ChatGroq
    from langchain_community.utilities import SQLDatabase
    from prompt_stages import STAGE_PROMPTS
    from sql_agent_pipeline import AGENTS, invoke_agent

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=...)
    db = SQLDatabase.from_uri(DB_URI)
    app = AGENTS["agent_2"](STAGE_PROMPTS["stage2_rules"], llm, db)
    result = invoke_agent(app, "How many thefts were reported in 2024?")
"""

import json
import re
import threading
from typing import Literal, Optional, TypedDict

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    schema: Optional[str]
    results: Optional[list]
    answer: Optional[str]
    error_log: list
    retry_count: int
    is_relevant: bool
    last_error: Optional[str]


_STRIP_FENCE_RE = re.compile(r"^```(?:sql)?\n?|```\s*$", re.IGNORECASE)
_FENCED_SQL_RE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _relevance_checker(state: AgentState, llm: ChatGroq) -> AgentState:
    question = state["question"].lower().strip()

    query_action_verbs = {
        "how many", "how often", "what are the", "list", "show",
        "find", "get", "count", "total", "sum", "average", "top", "most", "least",
        "rank", "filter", "between", "after", "before", "during", "in the year", "percentage",
    }
    data_context_words = {
        "crime", "chicago", "crimes", "report", "reported", "database", "table",
        "records", "data", "district", "year", "date", "type", "arrest", "victim",
        "incidents", "cases", "statistics", "breakdown", "distribution",
    }
    irrelevant_patterns = [
        (r"what\s+is\s+\d+\s*[\+\-\*/]\s*\d+", "math expression"),
        (r"what is the meaning", "meaning of life"),
        (r"^what is\s+\w+\s*\?*\s*$", "generic what is"),
        (r"^define\s+", "define"),
        (r"^explain\s+", "explain"),
        (r"^tell me about\s+", "tell me about"),
        (r"who (are|is|was|were)", "who question"),
        (r"when (are|is|was|were)", "when question"),
        (r"why (are|is|was|were)", "why question"),
    ]
    invalid_data_questions = [
        (r"what\s+is\s+(chicago_crime|the chicago crime|the crimes|the database|the table)(?:\s|$|\?)",
         "non-specific data"),
    ]

    for pattern, reason in invalid_data_questions:
        if re.search(pattern, question, re.IGNORECASE):
            state["is_relevant"] = False
            state["error_log"].append(f"Relevance: {reason}")
            break
    else:
        for pattern, reason in irrelevant_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                state["is_relevant"] = False
                state["error_log"].append(f"Relevance: {reason}")
                break
        else:
            has_action_verb = any(verb in question for verb in query_action_verbs)
            has_data_context = any(word in question for word in data_context_words)

            if has_action_verb and has_data_context:
                state["is_relevant"] = True
                state["error_log"].append("Relevance: Approved")
                return state

            state["is_relevant"] = False
            state["error_log"].append("Relevance: Rejected - missing action verb or data context")

    if not state["is_relevant"]:
        messages = [
            SystemMessage(content="You are a helpful assistant. The user's question is not related to a database or data analysis. Politely explain that their question is outside your scope and is not answerable using the database. Be concise but friendly."),
            HumanMessage(content=f"User question: {state['question']}"),
        ]
        response = llm.invoke(messages)
        state["answer"] = response.content

    return state


_SCHEMA_CACHE: dict[int, str] = {}
_SCHEMA_LOCK = threading.Lock()


def _schema_fetcher(state: AgentState, db: SQLDatabase) -> AgentState:
    # The schema is static for the duration of a run, and fetching it over the
    # network costs ~5s per question, so fetch once per SQLDatabase and reuse.
    try:
        with _SCHEMA_LOCK:
            if id(db) not in _SCHEMA_CACHE:
                _SCHEMA_CACHE[id(db)] = db.get_table_info()
            state["schema"] = _SCHEMA_CACHE[id(db)]
    except Exception as e:
        state["error_log"].append(f"Schema fetch error: {str(e)}")
        state["schema"] = ""
    return state


def _query_generator(state: AgentState, llm: ChatGroq, system_prompt: str) -> AgentState:
    schema = state.get("schema", "")
    question = state["question"]
    last_error = state.get("last_error")

    user_content = f"Schema:\n{schema}\n\nQuestion: {question}"
    if last_error:
        user_content += f"\n\nPrevious query failed validation with this error:\n{last_error}\nPlease fix the query."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    sql_query = response.content.strip()

    if "INVALID_REQUEST" in sql_query:
        state["error_log"].append("Generator: Request cannot be fulfilled with SELECT query")
        state["sql_query"] = None
    else:
        # Strip accidental markdown fences — stage1's bare prompt gives the
        # LLM no "no markdown" instruction, so this must not depend on the
        # prompt being tested.
        # A bare prompt can also add a preamble ("**SQL Query**") before the
        # fence, so prefer the fenced block's contents when one exists.
        fenced = _FENCED_SQL_RE.search(sql_query)
        sql_query = fenced.group(1).strip() if fenced else _STRIP_FENCE_RE.sub("", sql_query).strip()
        state["sql_query"] = sql_query
        state["last_error"] = None

    return state


def _query_validator(state: AgentState, db: SQLDatabase) -> AgentState:
    if not state.get("sql_query"):
        state["error_log"].append("Validator: No SQL query to validate")
        return state

    try:
        query = state["sql_query"]
        if re.search(r"\b(UPDATE|DELETE|DROP|INSERT|ALTER)\b", query, re.IGNORECASE):
            msg = f"Validator: DML statement detected in query: {query}"
            state["error_log"].append(msg)
            state["last_error"] = msg
            state["sql_query"] = None
            return state

        db.run(f"EXPLAIN {query}", fetch="all")
        state["error_log"].append("Validator: Query syntax is valid")
        state["last_error"] = None
    except Exception as e:
        error_msg = f"SQL syntax error: {str(e)}"
        state["error_log"].append(f"Validator: {error_msg}")
        state["last_error"] = error_msg
        state["sql_query"] = None

    return state


def _should_retry_query(state: AgentState) -> Literal["query_generator", "query_runner", "answer_synthesizer"]:
    if not state.get("sql_query"):
        state["retry_count"] += 1
        if state["retry_count"] >= 3:
            state["error_log"].append("Max retries reached (3). Ending query generation.")
            state["answer"] = "I was unable to generate a valid SQL query after 3 attempts. Please rephrase your question."
            return "answer_synthesizer"
        return "query_generator"
    return "query_runner"


def _query_runner(state: AgentState, db: SQLDatabase) -> AgentState:
    try:
        result = db.run(state["sql_query"], fetch="all")
        state["results"] = result if isinstance(result, list) else [result]
        state["error_log"].append(f"Runner: Query executed successfully. Rows: {len(state['results'])}")
    except Exception as e:
        state["error_log"].append(f"Runner: Query execution error: {str(e)}")
        state["results"] = []
    return state


def _answer_synthesizer(state: AgentState, llm: ChatGroq) -> AgentState:
    if state.get("answer"):
        return state

    if not state.get("results"):
        state["answer"] = "No results found for your query."
        return state

    results_str = json.dumps(state["results"][:10], indent=2, default=str)
    messages = [
        SystemMessage(content="You are a data analyst. Provide a clear, concise natural language summary of the SQL query results."),
        HumanMessage(content=f"Question: {state['question']}\n\nSQL Results:\n{results_str}\n\nProvide a summary."),
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state


def _should_continue_after_relevance(state: AgentState) -> Literal["schema_fetcher", "__end__"]:
    return "schema_fetcher" if state.get("is_relevant") else END


def build_sql_agent(query_gen_prompt: str, llm: ChatGroq, db: SQLDatabase):
    """Compile the LangGraph SQL agent, parameterized only by the
    query-generation system prompt. Every other node is fixed."""

    graph = StateGraph(AgentState)

    graph.add_node("relevance_checker", lambda s: _relevance_checker(s, llm))
    graph.add_node("schema_fetcher", lambda s: _schema_fetcher(s, db))
    graph.add_node("query_generator", lambda s: _query_generator(s, llm, query_gen_prompt))
    graph.add_node("query_validator", lambda s: _query_validator(s, db))
    graph.add_node("query_runner", lambda s: _query_runner(s, db))
    graph.add_node("answer_synthesizer", lambda s: _answer_synthesizer(s, llm))

    graph.add_edge(START, "relevance_checker")
    graph.add_conditional_edges("relevance_checker", _should_continue_after_relevance)
    graph.add_edge("schema_fetcher", "query_generator")
    graph.add_edge("query_generator", "query_validator")
    graph.add_conditional_edges("query_validator", _should_retry_query)
    graph.add_edge("query_runner", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", END)

    return graph.compile()


def build_agent_1(query_gen_prompt: str, llm: ChatGroq, db: SQLDatabase):
    """agent_1: the simple baseline. No relevance check, no validation, no
    retry loop — START > schema_fetcher > query_generator > query_runner >
    answer_synthesizer. Shares node implementations with agent_2 so the only
    differences are the relevance gate and the validate/retry loop."""

    graph = StateGraph(AgentState)

    graph.add_node("schema_fetcher", lambda s: _schema_fetcher(s, db))
    graph.add_node("query_generator", lambda s: _query_generator(s, llm, query_gen_prompt))
    graph.add_node("query_runner", lambda s: _query_runner_simple(s, db))
    graph.add_node("answer_synthesizer", lambda s: _answer_synthesizer(s, llm))

    graph.add_edge(START, "schema_fetcher")
    graph.add_edge("schema_fetcher", "query_generator")
    graph.add_edge("query_generator", "query_runner")
    graph.add_edge("query_runner", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", END)

    return graph.compile()


def _query_runner_simple(state: AgentState, db: SQLDatabase) -> AgentState:
    """Runner for agent_1: the generator may return no SQL (INVALID_REQUEST)
    and there is no validator in front of it, so guard against None."""
    if not state.get("sql_query"):
        state["results"] = []
        state["error_log"].append("Runner: No SQL query to execute")
        return state
    return _query_runner(state, db)


def build_agent_2(query_gen_prompt: str, llm: ChatGroq, db: SQLDatabase):
    """agent_2: the full pipeline (relevance check, validation, retry loop)."""
    return build_sql_agent(query_gen_prompt, llm, db)


AGENTS = {"agent_1": build_agent_1, "agent_2": build_agent_2}


def invoke_agent(app, question: str, config: Optional[dict] = None) -> dict:
    """Run one question through a compiled agent and return the final state.
    `config` is passed straight to `app.invoke` (e.g. to attach Langfuse
    callbacks/metadata) and defaults to None."""
    initial_state: AgentState = {
        "question": question,
        "sql_query": None,
        "schema": None,
        "results": None,
        "answer": None,
        "error_log": [],
        "retry_count": 0,
        "is_relevant": False,
        "last_error": None,
    }
    return app.invoke(initial_state, config=config) if config else app.invoke(initial_state)
