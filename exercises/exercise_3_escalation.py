"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
        "pr_author": pr.author,
    }


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)

    system_prompt = """You are an expert code reviewer. Analyze the given pull request diff and produce a structured review.

## Confidence score calibration (IMPORTANT)

The confidence score (0.0–1.0) measures how complete and correct your review is — NOT how risky the PR is.

| Score range | Meaning | Typical PR |
|-------------|---------|------------|
| > 0.72 | You have reviewed everything and are certain no important issue was missed | Typo fix, dependency bump, tiny refactor, formatting |
| 0.58 – 0.72 | You found issues but one or two questions remain that a human should confirm | Small feature, schema addition, straightforward logic change |
| < 0.58 | Significant uncertainty — security red flags, missing context, multiple ambiguous patterns | Auth code, password handling, SQL queries, cloud sync, no tests |

Be honest: most small features deserve 0.60–0.68. Only genuinely trivial PRs deserve > 0.72.
Only assign < 0.58 when there are concrete security issues or you truly cannot assess correctness without more context.
Do NOT round to exactly 0.58, 0.72, or any threshold boundary — pick a value that reflects your actual confidence.

## Your tasks
1. Summarize what the PR does (one paragraph).
2. List risk factors you observed.
3. Propose specific review comments (file, line if known, severity, body).
4. Assign a confidence score using the table above.
5. Explain your reasoning for that score in confidence_reasoning.
6. If confidence < 0.58, populate escalation_questions with 2–4 specific, context-rich questions.
   - Reference the exact file and section/line in the diff for each question.
   - Examples: "In auth.py line 42, MD5 is used for password hashing — is this intentional or a placeholder?",
     "In storage.py, SYNC_URL is hard-coded as HTTP — should this be HTTPS in production?"
   - Bad: "Are there security concerns?" (too vague)
   - Good: "Why does login() in auth.py use md5() instead of bcrypt or argon2?" (specific)"""

    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"PR Title: {state['pr_title']}\n"
                f"Author: {state.get('pr_author', 'unknown')}\n"
                f"Files changed: {', '.join(state['pr_files'])}\n\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c <= ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

    # Pause here and show the reviewer the specific questions.
    # answers will be a dict[question -> answer] returned via Command(resume=...).
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    a = state["analysis"]
    answers = state["escalation_answers"]  # dict[question, answer]

    # Format the Q&A context for the LLM
    qa_text = "\n".join(
        f"Q: {q}\nA: {ans}" for q, ans in answers.items()
    )

    system_prompt = """You are an expert code reviewer. You previously reviewed a PR but had low confidence.
A human reviewer has now answered your clarifying questions.
Using the original diff, your initial analysis, and the reviewer's answers,
produce a refined and more complete review. Your confidence should now be higher
since the ambiguities have been resolved."""

    user_prompt = (
        f"PR Title: {state['pr_title']}\n"
        f"Files changed: {', '.join(state['pr_files'])}\n\n"
        f"Original diff:\n{state['pr_diff']}\n\n"
        f"Your initial analysis:\n"
        f"- Summary: {a.summary}\n"
        f"- Risk factors: {', '.join(a.risk_factors)}\n"
        f"- Initial confidence: {a.confidence:.0%}\n"
        f"- Reasoning: {a.confidence_reasoning}\n\n"
        f"Reviewer Q&A:\n{qa_text}\n\n"
        f"Please produce a refined, complete review taking into account the reviewer's answers."
    )

    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM synthesizing refined review...[/dim]"):
        refined = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}, {len(refined.comments)} comment(s)")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("escalate", "synthesize")   # escalate → synthesize → commit
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)

    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            f"[bold]Summary:[/bold] {payload['summary']}\n\n"
            f"[bold]Risk factors:[/bold]\n" + "\n".join(f"  • {r}" for r in payload.get("risk_factors", [])),
            title=f"[red]Escalation[/red] conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        console.print(f"[dim]{payload['confidence_reasoning']}[/dim]\n")
        # Collect one answer per question; returns dict[question -> answer]
        return {q: console.input(f"[bold]Q:[/bold] {q}\n[bold]A:[/bold] ").strip()
                for q in payload["questions"]}
    raise ValueError(f"Unknown interrupt kind: {kind}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()