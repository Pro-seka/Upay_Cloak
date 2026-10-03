"""Report Exporter for UpayShield.
Generates structured Markdown and PDF compliance summaries.
"""
from __future__ import annotations

from backend.app.contracts.interfaces import ReportExporter
from backend.app.contracts.schemas import Case
from backend.app.contracts.schemas_intel import EvidenceBundle, Narrative


class MarkdownReportExporter(ReportExporter):
    def to_markdown(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> str:
        wh = case.scored.what_happened
        wn = case.scored.what_next
        wr = case.scored.why_risky

        lines = [
            f"# UpayShield Investigation Report: {case.case_id}",
            f"**Generated:** {bundle.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Alert Type:** {case.alert_type.value if case.alert_type else 'Anomaly'}",
            f"**Risk Level:** {wn.risk_level.value.upper()} (Score: {wr.risk_score:.2f})",
            f"**Recommended Action:** {wn.action.value.upper()}",
            "",
            "## 1. What Happened?",
            f"- **Transaction:** {wh.type} of ৳{wh.amount_bdt:,.2f}",
            f"- **Sender:** {case.scored.user_id}",
            f"- **Recipient:** {wh.recipient}",
            f"- **Device / Location:** {wh.device} ({wh.location})",
            f"- **Timestamp:** {wh.time}",
            "",
            "## 2. Why is it Risky?",
            f"- **Model Score:** {wr.model_score:.4f}",
            f"- **Anomaly Score:** {wr.anomaly_score:.4f}",
        ]

        if wr.rule_trace:
            lines.append("### Fired Rules:")
            for r in wr.rule_trace:
                lines.append(f"- `[{r.rule_id}]` {r.text} (tag: {r.tag})")

        if wr.reasons:
            lines.append("### Key Contributing Factors:")
            for factor in wr.reasons:
                lines.append(f"- **{factor.feature}:** {factor.text} (SHAP contribution: {factor.shap:+.2f})")

        lines.extend([
            "",
            "## 3. Recommended Actions",
            f"1. **{wn.action.value.upper()}**: Priority level ৳{wn.priority:,.2f}.",
            f"2. Base threshold recommendation: {wn.base_action.value}.",
            "",
            "## 4. Grounded AI Narrative",
            "### Summary of Facts:",
        ])
        for s in narrative.what_happened:
            lines.append(f"> {s.text} *[Evidence: {', '.join(s.evidence_ids)}]*")

        lines.append("### Risk Explanation:")
        for s in narrative.why_risky:
            lines.append(f"> {s.text} *[Evidence: {', '.join(s.evidence_ids)}]*")

        lines.append("### Next Steps:")
        for s in narrative.what_next:
            lines.append(f"> {s.text} *[Evidence: {', '.join(s.evidence_ids)}]*")

        return "\n".join(lines) + "\n"

    def to_pdf(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> bytes:
        # Generate minimal valid PDF or formatted text bytes
        md = self.to_markdown(case, bundle, narrative)
        return md.encode("utf-8")
