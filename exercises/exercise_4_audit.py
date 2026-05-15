"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table."""
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


# ─── Reference example ─────────────────────────────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
        "pr_author": pr.author,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
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
6. If confidence < 0.58, populate escalation_questions with 2–4 specific, context-rich questions
   that reference exact file/line in the diff."""

    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"PR Title: {state['pr_title']}\n"
                f"Author: {state.get('pr_author', 'unknown')}\n"
                f"Files changed: {', '.join(state['pr_files'])}\n\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c <= ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=decision,
        reason=f"Routed to {decision} based on confidence={c:.0%}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    console.print("[cyan]→ human_approval[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]

    # Audit BEFORE interrupt — reviewer hasn't responded yet
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="pending",
        reason="Waiting for human approval",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    # Audit AFTER resume — now we know the reviewer's choice
    reviewer = os.environ.get("GITHUB_USER")
    choice = resp.get("choice", "unknown")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer,
        decision=choice,
        reason=resp.get("feedback") or f"Reviewer chose: {choice}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    return {"human_choice": choice, "human_feedback": resp.get("feedback")}


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


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()

    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=state["analysis"].confidence,
        risk_level=risk_level_for(state["analysis"].confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=action,
        reason=state.get("human_feedback") or ("Posted after escalation Q&A" if state.get("escalation_answers") else action),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,   # no human involved
        decision="auto",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    console.print("[cyan]→ escalate[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # Audit BEFORE interrupt — reviewer hasn't answered yet
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="escalate",
        reason=f"Escalating with {len(questions)} question(s): {'; '.join(questions[:2])}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    # Audit AFTER resume — reviewer answered
    reviewer = os.environ.get("GITHUB_USER")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer,
        decision="pending",
        reason=f"Reviewer answered {len(answers)} question(s)",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in (state.get("escalation_answers") or {}).items())

    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "You are an expert code reviewer. You previously reviewed a PR but had low confidence. "
                "A human reviewer has now answered your clarifying questions. "
                "Using the original diff, your initial analysis, and the reviewer's answers, "
                "produce a refined and more complete review. Your confidence should now be higher."
            )},
            {"role": "user", "content": (
                f"PR Title: {state['pr_title']}\n"
                f"Files changed: {', '.join(state['pr_files'])}\n\n"
                f"Original diff:\n{state['pr_diff']}\n\n"
                f"Initial summary: {a.summary}\n"
                f"Initial confidence: {a.confidence:.0%}\n"
                f"Initial reasoning: {a.confidence_reasoning}\n\n"
                f"Reviewer Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="pending",
        reason=f"Refined from {a.confidence:.0%} → {refined.confidence:.0%} after Q&A",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
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
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            f"[bold]Summary:[/bold] {payload['summary']}\n\n"
            f"[dim]{payload['confidence_reasoning']}[/dim]",
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        choice = console.input("\napprove/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}

    # kind == "escalation"
    console.print(Panel.fit(
        f"[bold]Summary:[/bold] {payload['summary']}\n\n"
        + "\n".join(f"  • {r}" for r in payload.get("risk_factors", [])),
        title=f"[red]Escalation[/red] conf={payload['confidence']:.0%}",
        border_style="yellow",
    ))
    console.print(f"[dim]{payload['confidence_reasoning']}[/dim]\n")
    return {q: console.input(f"[bold]Q:[/bold] {q}\n[bold]A:[/bold] ").strip()
            for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()