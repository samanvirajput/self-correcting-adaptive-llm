import re
from dataclasses import dataclass

_CRITIQUE_TEMPLATE = """\
Evaluate the following assistant response to the given user query.
Identify specific issues in these categories: factual errors, tone mismatch, repetition, ignored context, verbosity.
List each issue on its own line as: ISSUE: <category> | <brief description>
If there are no issues, respond with: ISSUE: none | response is satisfactory

Query: {query}

Response: {response}

Evaluation:"""

_REVISE_TEMPLATE = """\
The following assistant response has issues that need fixing:
{issues_summary}

Original query: {query}
Original response: {response}

Write an improved response that directly addresses all identified issues. Be concise and accurate.

Improved response:"""


@dataclass
class ReflectionResult:
    original: str
    critique: str
    issues: list[dict]
    revised: str | None
    was_corrected: bool


def _parse_issues(critique_text: str) -> list[dict]:
    issues = []
    for line in critique_text.splitlines():
        m = re.match(r"ISSUE:\s*(.+?)\s*\|\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            category = m.group(1).strip().lower()
            description = m.group(2).strip()
            if category != "none":
                issues.append({"category": category, "description": description})
    return issues


def reflect(query: str, response: str, model) -> ReflectionResult:
    critique_prompt = _CRITIQUE_TEMPLATE.format(query=query, response=response)
    critique = model.generate(critique_prompt, max_tokens=256)

    issues = _parse_issues(critique)

    if not issues:
        return ReflectionResult(
            original=response,
            critique=critique,
            issues=[],
            revised=None,
            was_corrected=False,
        )

    issues_summary = "\n".join(f"- [{i['category']}] {i['description']}" for i in issues)
    revise_prompt = _REVISE_TEMPLATE.format(
        issues_summary=issues_summary,
        query=query,
        response=response,
    )
    revised = model.generate(revise_prompt, max_tokens=512)

    return ReflectionResult(
        original=response,
        critique=critique,
        issues=issues,
        revised=revised,
        was_corrected=True,
    )
