import json
import re
from datetime import datetime
from typing import Dict, List, Tuple


class IntelligenceLayer:
    """Generic full-dataset analytics layer.

    This layer avoids question-specific hardcoding and always builds a broad
    analytics context from all indexed categorical rows. A lightweight scope
    filter (machine/job/date/range) is inferred from the question, then both
    global and scoped metrics are returned for synthesis.
    """

    def __init__(self, vectorstore) -> None:
        self.vectorstore = vectorstore

    @staticmethod
    def _to_float(value: str | float | int | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else None

    @staticmethod
    def _job_numeric(job_code: str) -> int | None:
        m = re.fullmatch(r"J(\d{2,6})", (job_code or "").upper())
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_machine_codes(question: str) -> List[str]:
        found = re.findall(r"\bM\d{1,4}\b", question, flags=re.IGNORECASE)
        out: List[str] = []
        seen = set()
        for code in found:
            c = code.upper()
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    @staticmethod
    def _extract_job_codes(question: str) -> List[str]:
        found = re.findall(r"\bJ\d{2,6}\b", question, flags=re.IGNORECASE)
        out: List[str] = []
        seen = set()
        for code in found:
            c = code.upper()
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _extract_job_range(self, question: str) -> tuple[int, int] | None:
        jobs = self._extract_job_codes(question)
        if len(jobs) >= 2 and any(token in question.lower() for token in [" to ", "range", "from"]):
            a = self._job_numeric(jobs[0])
            b = self._job_numeric(jobs[1])
            if a is not None and b is not None:
                return (min(a, b), max(a, b))
        m = re.search(r"first\s+(\d+)\s+jobs", question, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return (1, n)
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        txt = (value or "").strip()
        if not txt:
            return None
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue
        return None

    def _extract_query_date(self, question: str) -> datetime | None:
        q = question.strip()
        m_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
        if m_iso:
            return self._parse_datetime(m_iso.group(1))

        # March 18, 2023 or March18, 2023
        m_month_first = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*"
            r"(\d{1,2})\s*,?\s*(20\d{2})\b",
            q,
            flags=re.IGNORECASE,
        )
        if m_month_first:
            return self._parse_datetime(
                f"{m_month_first.group(2)} {m_month_first.group(1)} {m_month_first.group(3)}"
            )

        # 18, March 2023 or 18 March 2023
        m_day_first = re.search(
            r"\b(\d{1,2})\s*,?\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(20\d{2})\b",
            q,
            flags=re.IGNORECASE,
        )
        if m_day_first:
            return self._parse_datetime(
                f"{m_day_first.group(1)} {m_day_first.group(2)} {m_day_first.group(3)}"
            )
        return None

    @staticmethod
    def _extract_status_from_text(text: str) -> str:
        t = (text or "").lower()
        if re.search(r"\b(fail|failed|error|rejected)\b", t):
            return "failed"
        if re.search(r"\b(delay|delayed)\b", t):
            return "delayed"
        if re.search(r"\b(completed|complete|success|done|passed)\b", t):
            return "completed"
        return ""

    @staticmethod
    def _extract_operation_from_text(text: str) -> str:
        t = (text or "").lower()
        if "grind" in t:
            return "Grinding"
        if "milli" in t or "mill" in t:
            return "Milling"
        if "lathe" in t:
            return "Lathe"
        if "drill" in t:
            return "Drilling"
        if "addit" in t:
            return "Additive"
        return ""

    def _collect_all_rows(self) -> List[Dict[str, object]]:
        dump = self.vectorstore.get(include=["documents", "metadatas"])
        docs = dump.get("documents", [])
        metas = dump.get("metadatas", [])

        rows: List[Dict[str, object]] = []
        seen = set()
        for idx in range(len(docs)):
            meta = metas[idx] or {}
            if meta.get("doc_kind") != "categorical_row":
                continue
            payload = str(meta.get("row_payload", docs[idx][:220])).strip()
            machine = str(meta.get("row_machine", "")).strip().upper()
            job = str(meta.get("row_job", "")).strip().upper()
            key = (job, machine, payload)
            if key in seen:
                continue
            seen.add(key)
            try:
                fields = json.loads(str(meta.get("row_fields_json", "{}")))
            except Exception:
                fields = {}
            rows.append(
                {
                    "source": str(meta.get("source", "unknown")),
                    "machine": machine,
                    "job": job,
                    "job_type": str(meta.get("row_job_type", "")).strip(),
                    "snippet": payload,
                    "fields": fields,
                }
            )
        return rows

    def _row_datetime(self, row: Dict[str, object]) -> datetime | None:
        fields = row.get("fields", {}) or {}
        for token in ["actual_start", "scheduled_start", "start_time", "actual_end", "end_time"]:
            for k, v in fields.items():
                if token in str(k).lower():
                    dt = self._parse_datetime(str(v))
                    if dt:
                        return dt
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", str(row.get("snippet", "")))
        if m:
            return self._parse_datetime(m.group(1))
        return None

    def _normalize_scope_state(self, scope: Dict[str, object]) -> Dict[str, object]:
        machines = [str(m).upper() for m in scope.get("machines", []) if str(m).strip()]
        jobs = [str(j).upper() for j in scope.get("jobs", []) if str(j).strip()]
        date = str(scope.get("date", "") or "").strip()
        job_range = scope.get("job_range", []) or []
        if isinstance(job_range, tuple):
            job_range = list(job_range)
        job_range = [int(x) for x in job_range[:2]] if len(job_range) >= 2 else []
        return {
            "machines": sorted(list(dict.fromkeys(machines))),
            "jobs": sorted(list(dict.fromkeys(jobs))),
            "date": date,
            "job_range": job_range,
        }

    @staticmethod
    def _scope_has_filters(scope_state: Dict[str, object]) -> bool:
        return bool(
            scope_state.get("machines")
            or scope_state.get("jobs")
            or scope_state.get("date")
            or scope_state.get("job_range")
        )

    @staticmethod
    def _should_reset_scope(question: str) -> bool:
        q = (question or "").lower()
        return any(
            token in q
            for token in [
                "all machines",
                "all jobs",
                "overall",
                "global",
                "entire dataset",
                "entire data",
                "everything",
                "all data",
            ]
        )

    def _infer_scope(
        self,
        rows: List[Dict[str, object]],
        question: str,
        prior_scope: Dict[str, object] | None = None,
    ) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
        prior_scope = self._normalize_scope_state(prior_scope or {})

        machines = set(self._extract_machine_codes(question))
        jobs = set(self._extract_job_codes(question))
        q_date = self._extract_query_date(question)
        job_range = self._extract_job_range(question)

        if self._should_reset_scope(question):
            prior_scope = {}

        if not machines and prior_scope.get("machines"):
            machines = set(prior_scope.get("machines", []))
        if not jobs and prior_scope.get("jobs"):
            jobs = set(prior_scope.get("jobs", []))
        if q_date is None and prior_scope.get("date"):
            q_date = self._parse_datetime(str(prior_scope.get("date")))
        if job_range is None and prior_scope.get("job_range"):
            jr = prior_scope.get("job_range") or []
            if len(jr) >= 2:
                job_range = (int(jr[0]), int(jr[1]))

        scoped = rows
        if machines:
            scoped = [r for r in scoped if r.get("machine", "") in machines]
        if jobs:
            scoped = [r for r in scoped if r.get("job", "") in jobs]
        if job_range is not None:
            lo, hi = job_range
            scoped = [
                r for r in scoped
                if self._job_numeric(str(r.get("job", ""))) is not None
                and lo <= (self._job_numeric(str(r.get("job", ""))) or -1) <= hi
            ]
        if q_date is not None:
            scoped = [r for r in scoped if (self._row_datetime(r) is not None and self._row_datetime(r).date() == q_date.date())]

        scope = {
            "machines": sorted(list(machines)),
            "jobs": sorted(list(jobs)),
            "date": q_date.strftime("%Y-%m-%d") if q_date else "",
            "job_range": list(job_range) if job_range else [],
            "rows_after_filter": len(scoped),
        }
        scope_state = self._normalize_scope_state(scope)
        return scoped, scope, scope_state

    def _metric_values(self, rows: List[Dict[str, object]], metric_token: str) -> List[float]:
        vals: List[float] = []
        for r in rows:
            fields = r.get("fields", {}) or {}
            raw = ""
            for k, v in fields.items():
                if metric_token in str(k).lower():
                    raw = str(v)
                    break
            n = self._to_float(raw)
            if n is not None:
                vals.append(n)
        return vals

    def _summarize(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        jobs = sorted({str(r.get("job", "")) for r in rows if re.fullmatch(r"J\d{2,6}", str(r.get("job", "")))})
        machines = sorted({str(r.get("machine", "")) for r in rows if re.fullmatch(r"M\d{1,4}", str(r.get("machine", "")))})

        status_counts: Dict[str, int] = {}
        op_counts: Dict[str, int] = {}
        per_machine_jobs: Dict[str, set[str]] = {}
        per_day_jobs: Dict[str, set[str]] = {}
        dates: List[datetime] = []

        for r in rows:
            m = str(r.get("machine", ""))
            j = str(r.get("job", ""))
            if re.fullmatch(r"M\d{1,4}", m) and re.fullmatch(r"J\d{2,6}", j):
                per_machine_jobs.setdefault(m, set()).add(j)

            fields = r.get("fields", {}) or {}
            st = ""
            for k, v in fields.items():
                if any(t in str(k).lower() for t in ["status", "state", "result", "outcome"]):
                    st = str(v)
                    break
            if not st:
                st = self._extract_status_from_text(str(r.get("snippet", "")))
            if st:
                key = st.lower()
                status_counts[key] = status_counts.get(key, 0) + 1

            op = ""
            for k, v in fields.items():
                if "operation" in str(k).lower():
                    op = str(v)
                    break
            if not op:
                op = self._extract_operation_from_text(str(r.get("snippet", "")))
            if op:
                op_counts[op] = op_counts.get(op, 0) + 1

            dt = self._row_datetime(r)
            if dt:
                dates.append(dt)
                if re.fullmatch(r"J\d{2,6}", j):
                    per_day_jobs.setdefault(dt.strftime("%Y-%m-%d"), set()).add(j)

        numeric = {}
        for token in ["energy_consumption", "processing_time", "material_used", "machine_availability"]:
            vals = self._metric_values(rows, token)
            if vals:
                numeric[token] = {
                    "count": len(vals),
                    "sum": round(sum(vals), 6),
                    "avg": round(sum(vals) / len(vals), 6),
                    "min": round(min(vals), 6),
                    "max": round(max(vals), 6),
                }

        per_machine_job_counts = {k: len(v) for k, v in sorted(per_machine_jobs.items())}
        per_day_job_counts = {k: len(v) for k, v in sorted(per_day_jobs.items())}

        # Generic downtime estimate: positive start delay + failed-job scheduled duration proxy.
        downtime_min: Dict[str, float] = {}
        for r in rows:
            m = str(r.get("machine", ""))
            if not re.fullmatch(r"M\d{1,4}", m):
                continue
            fields = r.get("fields", {}) or {}
            start_s = ""
            actual_s = ""
            end_s = ""
            for k, v in fields.items():
                lk = str(k).lower()
                if ("start_time" in lk or "scheduled_start" in lk) and not start_s:
                    start_s = str(v)
                if "actual_start" in lk and not actual_s:
                    actual_s = str(v)
                if ("end_time" in lk or "scheduled_end" in lk) and not end_s:
                    end_s = str(v)
            sched = self._parse_datetime(start_s) if start_s else None
            actual = self._parse_datetime(actual_s) if actual_s else None
            end = self._parse_datetime(end_s) if end_s else None
            if sched and actual:
                d = (actual - sched).total_seconds() / 60.0
                if d < 0:
                    d += 24 * 60
                if d > 0:
                    downtime_min[m] = downtime_min.get(m, 0.0) + d
            elif sched and end and self._extract_status_from_text(str(r.get("snippet", ""))) == "failed":
                d = (end - sched).total_seconds() / 60.0
                if d < 0:
                    d += 24 * 60
                if d > 0:
                    downtime_min[m] = downtime_min.get(m, 0.0) + d

        # Failure rates by machine from available statuses.
        failure_rates: Dict[str, Dict[str, float]] = {}
        by_machine_status: Dict[str, Dict[str, int]] = {}
        for r in rows:
            m = str(r.get("machine", ""))
            if not re.fullmatch(r"M\d{1,4}", m):
                continue
            st = self._extract_status_from_text(str(r.get("snippet", "")))
            if not st:
                fields = r.get("fields", {}) or {}
                for k, v in fields.items():
                    if any(t in str(k).lower() for t in ["status", "state", "result", "outcome"]):
                        st = str(v).lower()
                        break
            if not st:
                continue
            by_machine_status.setdefault(m, {"failed": 0, "completed": 0, "delayed": 0, "other": 0})
            if "fail" in st:
                by_machine_status[m]["failed"] += 1
            elif "delay" in st:
                by_machine_status[m]["delayed"] += 1
            elif any(x in st for x in ["complete", "success", "done", "pass"]):
                by_machine_status[m]["completed"] += 1
            else:
                by_machine_status[m]["other"] += 1
        for m, c in by_machine_status.items():
            total = c["failed"] + c["completed"] + c["delayed"] + c["other"]
            if total > 0:
                failure_rates[m] = {
                    "failed": c["failed"],
                    "total": total,
                    "failure_rate_pct": round((c["failed"] / total) * 100.0, 4),
                }

        return {
            "rows": len(rows),
            "distinct_jobs": len(jobs),
            "distinct_machines": len(machines),
            "machines": machines,
            "date_min": min(dates).strftime("%Y-%m-%d") if dates else "",
            "date_max": max(dates).strftime("%Y-%m-%d") if dates else "",
            "status_counts": status_counts,
            "operation_counts": op_counts,
            "per_machine_job_counts": per_machine_job_counts,
            "per_day_job_counts": per_day_job_counts,
            "numeric_stats": numeric,
            "downtime_minutes_by_machine": {k: round(v, 3) for k, v in sorted(downtime_min.items())},
            "failure_rates_by_machine": failure_rates,
        }

    def answer(
        self,
        question: str,
        prior_scope: Dict[str, object] | None = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, object]] | None:
        rows = self._collect_all_rows()
        if not rows:
            return None

        scoped_rows, scope, scope_state = self._infer_scope(rows, question, prior_scope=prior_scope)
        # If filters are too strict and match nothing, keep global data visible.
        effective_rows = scoped_rows if scoped_rows else rows
        scope["used_global_fallback"] = bool(not scoped_rows and len(rows) > 0)

        global_summary = self._summarize(rows)
        scoped_summary = self._summarize(effective_rows)

        analysis_payload = {
            "question": question,
            "scope": scope,
            "global_summary": global_summary,
            "scoped_summary": scoped_summary,
            "notes": [
                "All metrics are computed from indexed categorical rows, not top-k retrieval sampling.",
                "If a metric is missing, required source fields are absent in indexed data.",
            ],
        }

        answer = "ANALYTICS_CONTEXT_JSON:\n" + json.dumps(analysis_payload, ensure_ascii=True, indent=2)
        sources = [
            {"source": str(r.get("source", "unknown")), "type": "pdf", "snippet": str(r.get("snippet", ""))}
            for r in effective_rows[:150]
        ]
        return answer, sources, scope_state

    def machine_row_counts(self) -> Dict[str, int]:
        rows = self._collect_all_rows()
        counts: Dict[str, int] = {}
        for r in rows:
            m = str(r.get("machine", ""))
            if not re.fullmatch(r"M\d{1,4}", m):
                continue
            counts[m] = counts.get(m, 0) + 1
        return counts
