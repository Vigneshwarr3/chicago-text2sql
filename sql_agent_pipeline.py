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


# The state dictionary that LangGraph passes from node to node. Each node reads
# the fields it needs and writes its own output back into the same dict.
class AgentState(TypedDict):
    question: str               # the user's natural-language question
    sql_query: Optional[str]    # SQL written by the query generator (None if none / rejected)
    schema: Optional[str]       # CREATE TABLE text + sample rows shown to the model
    results: Optional[list]     # what the query runner got back from PostgreSQL
    answer: Optional[str]       # the final plain-English answer
    error_log: list             # one line per step; useful for debugging a run
    retry_count: int            # agent_2: how many generations have failed validation
    is_relevant: bool           # agent_2: verdict of the relevance checker
    last_error: Optional[str]   # agent_2: validation error passed back to the generator


# Models sometimes wrap SQL in markdown fences (```sql ... ```); these patterns
# recover the bare SQL (see _query_generator).
_STRIP_FENCE_RE = re.compile(r"^```(?:sql)?\n?|```\s*$", re.IGNORECASE)
_FENCED_SQL_RE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _relevance_checker(state: AgentState, llm: ChatGroq) -> AgentState:
    """agent_2 only. A rule-based (keyword) filter that decides whether the
    question is a data question at all; no model call is made to decide.

    Known limitation, kept unchanged so the published results reproduce: the
    rule needs one of the listed "action" phrases, so in-scope questions worded
    "Which ... had ..." or "What is the ... rate ..." are rejected. In the
    benchmark this rejects the same 8 of 96 questions in every run."""
    question = state["question"].lower().strip()

    # A question must contain at least one phrase from each of these two sets.
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
    # Questions matching any of these patterns are rejected outright.
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

    # Checked in order: vague "what is the database" questions, then clearly
    # off-topic patterns, then the action-phrase + data-word rule. Python's
    # for/else runs the `else` block only when the loop did not `break`.
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

    # Rejected: the model is called only to word a polite out-of-scope reply.
    # The graph then ends (see _should_continue_after_relevance).
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
    """Put the table definitions (and a few sample rows per table, as produced
    by LangChain's SQLDatabase.get_table_info) into the state. Only the tables
    the SQLDatabase was created with are included (INCLUDE_TABLES in
    experiment_runner.py)."""
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
    """Ask the model for one SQL query. `system_prompt` is the prompt stage under
    test (prompt_stages.py); the user message carries the schema and question.
    On a retry (agent_2), the previous validation error is appended so the model
    can correct its query."""
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

    # The stage 2-4 prompts tell the model to answer INVALID_REQUEST when a
    # question cannot be answered with a SELECT.
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
    """agent_2 only. Reject data-modifying statements, then ask PostgreSQL to
    plan the query with EXPLAIN (which checks syntax, table and column names
    without running it). On failure, sql_query is cleared and the error is kept
    in last_error for the next generation attempt."""
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
    """agent_2 routing after validation: a valid query goes to the runner; an
    invalid one goes back to the generator, at most 3 generations in total;
    after the third failure a fixed apology becomes the answer."""
    if not state.get("sql_query"):
        state["retry_count"] += 1
        if state["retry_count"] >= 3:
            state["error_log"].append("Max retries reached (3). Ending query generation.")
            state["answer"] = "I was unable to generate a valid SQL query after 3 attempts. Please rephrase your question."
            return "answer_synthesizer"
        return "query_generator"
    return "query_runner"


def _query_runner(state: AgentState, db: SQLDatabase) -> AgentState:
    """Execute the query. SQLDatabase.run returns the rows as one formatted
    string, so `results` is normally a one-element list holding that string.
    Errors are logged and produce an empty result."""
    try:
        result = db.run(state["sql_query"], fetch="all")
        state["results"] = result if isinstance(result, list) else [result]
        state["error_log"].append(f"Runner: Query executed successfully. Rows: {len(state['results'])}")
    except Exception as e:
        state["error_log"].append(f"Runner: Query execution error: {str(e)}")
        state["results"] = []
    return state


def _answer_synthesizer(state: AgentState, llm: ChatGroq) -> AgentState:
    """Turn the query result into a plain-English answer with one model call."""
    # An answer already set earlier (relevance rejection, or 3 failed
    # validations) is kept as is.
    if state.get("answer"):
        return state

    if not state.get("results"):
        state["answer"] = "No results found for your query."
        return state

    # `results` is normally a one-element list holding the whole result as one
    # string (see _query_runner), so [:10] does not limit the rows: the model
    # sees the complete result.
    results_str = json.dumps(state["results"][:10], indent=2, default=str)
    messages = [
        SystemMessage(content="You are a data analyst. Provide a clear, concise natural language summary of the SQL query results."),
        HumanMessage(content=f"Question: {state['question']}\n\nSQL Results:\n{results_str}\n\nProvide a summary."),
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state


# agent_2 routing after the relevance checker: continue, or stop the graph.
def _should_continue_after_relevance(state: AgentState) -> Literal["schema_fetcher", "__end__"]:
    return "schema_fetcher" if state.get("is_relevant") else END


def build_sql_agent(query_gen_prompt: str, llm: ChatGroq, db: SQLDatabase):
    """Compile the LangGraph SQL agent, parameterized only by the
    query-generation system prompt. Every other node is fixed."""

    graph = StateGraph(AgentState)

    # Each node is a function of the state; lambdas bind the model, database
    # and prompt to it.
    graph.add_node("relevance_checker", lambda s: _relevance_checker(s, llm))
    graph.add_node("schema_fetcher", lambda s: _schema_fetcher(s, db))
    graph.add_node("query_generator", lambda s: _query_generator(s, llm, query_gen_prompt))
    graph.add_node("query_validator", lambda s: _query_validator(s, db))
    graph.add_node("query_runner", lambda s: _query_runner(s, db))
    graph.add_node("answer_synthesizer", lambda s: _answer_synthesizer(s, llm))

    # Fixed edges always go to the next node; conditional edges call a routing
    # function that returns the name of the next node.
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


# Name -> builder. Every builder takes (query_gen_prompt, llm, db) and returns
# a compiled graph that invoke_agent() can run.
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
