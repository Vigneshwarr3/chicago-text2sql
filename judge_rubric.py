"""
judge_rubric.py
===============
The grading rubrics used for the two LLM-judged metrics. They are shared by
the manual grading pass (grades.json) and the scripted judge
(judge_results.py --mode api), so both apply the same criteria.

JUDGE_MODEL is the model used by the scripted path; the released grades.json
was produced by Claude Sonnet 5.5 reading the exported items directly
(see judge_results.py).
"""

JUDGE_MODEL = "claude-sonnet-5-5"

# --- EX fallback: is a result that failed strict/relational matching still equivalent? ---------

EX_RUBRIC = """You are checking a text-to-SQL agent's query result against a ground-truth result.
Both are the first rows (up to 20) returned by running each query.

Decide whether the agent result is EQUIVALENT to the ground truth for the question asked.
Equivalent means it contains the same information, even if column names, column order,
row order, aliases or numeric precision differ, or it adds/omits a harmless identifier column
(e.g. a name alongside an id).
NOT equivalent if values differ materially, rows are missing or extra in a way that changes
the answer, the result is truncated relative to the ground truth, a different quantity is
computed, or a different but plausible interpretation yields different numbers.

Reply with JSON only: {"equivalent": true|false, "reason": "<one short sentence>"}"""

# --- SQ: synthesis quality of the plain-English answer -----------------------------------------

SQ_RUBRIC = """You are scoring the natural-language answer a text-to-SQL agent gave to a user.
You see the question, the agent's query result (first rows), and the agent's answer.
Score the ANSWER's faithfulness and usefulness on a 1-5 scale:

5 = Fully faithful to the result rows, directly answers the question, numbers/names correct,
    nothing important omitted, no invented facts.
4 = Faithful and answers the question, with a minor flaw (slight omission, awkward framing,
    unnecessary hedging, small rounding looseness).
3 = Partly useful: answers only part of the question, ignores part of the result, or is
    vague, or presents the result with a noticeable misreading that does not flip the answer.
2 = Substantially misleading: misstates the results, states a conclusion the rows do not
    support, or answers a different question.
1 = Unusable or wrong: fabricated numbers, contradicts the rows, an empty/error response,
    or reports "no results" / "out of scope" when the question is in scope and data exists.

Judge the answer against the result rows shown, not against any hidden reference.
If the query returned an empty set and the answer says nothing was found, that is faithful.
If the query errored (the result shows an error) or no query ran, an answer that says
"no results found" or "out of scope" instead of reporting a failure is misleading: score 1.

Reply with JSON only: {"score": 1-5, "reason": "<one short sentence>"}"""
