"""Exercise 1 — Confidence scoring + routing.
Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.
"""
from __future__ import annotations

import argparse
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)

console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
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


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)

    system_prompt = """You are an expert code reviewer. Analyze the given pull request diff and produce a structured review.

## Confidence score calibration (IMPORTANT)

The confidence score (0.0–1.0) measures how complete and correct your review is — NOT how risky the PR is.

| Score range | Meaning | Typical PR |
|-------------|---------|------------|
| > 0.72 | You have reviewed everything and are certain no important issue was missed | Typo fix, dependency bump, tiny refactor, formatting |
| 0.59 – 0.72 | You found issues but one or two questions remain that a human should confirm | Small feature, schema addition, straightforward logic change |
| < 0.59 | Significant uncertainty — security red flags, missing context, multiple ambiguous patterns | Auth code, password handling, SQL queries, cloud sync, no tests |

Be honest: most small features deserve 0.60–0.68. Only genuinely trivial PRs deserve > 0.72.
Only assign < 0.59 when there are concrete security issues or you truly cannot assess correctness without more context.

## Your tasks
1. Summarize what the PR does (one paragraph).
2. List risk factors you observed.
3. Propose specific review comments (file, line if known, severity, body).
4. Assign a confidence score using the table above.
5. Explain your reasoning for that score in confidence_reasoning.
6. If score < 0.59, populate escalation_questions with specific questions to ask the human reviewer."""

    human_prompt = f"""PR Title: {state['pr_title']}
Author: {state.get('pr_author', 'unknown')}
Files changed: {', '.join(state['pr_files'])}

Diff:
{state['pr_diff']}

Please analyze this PR and provide a structured review."""

    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

    console.print(f"  [green]✓[/green] confidence = {analysis.confidence:.0%} | {len(analysis.comments)} comment(s)")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    confidence = state["analysis"].confidence

    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"

    console.print(f"  confidence={confidence:.0%} → [bold]{decision}[/bold]")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def route_by_confidence(state: ReviewState) -> str:
    """Conditional edge: return the node name to go to next."""
    return state["decision"]


def build_graph():
    g = StateGraph(ReviewState)

    # Add nodes
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    # Linear edges
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")

    # Conditional routing from "route" node
    g.add_conditional_edges(
        "route",
        route_by_confidence,
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )

    # Terminal edges
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)

    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()