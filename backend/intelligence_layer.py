import json
import math
import re
from datetime import datetime
from typing import Dict, List, Tuple


class IntelligenceLayer:
    """Dataset-agnostic analytics layer over indexed categorical rows."""

    def __init__(self, vectorstore) -> None:
        self.vectorstore = vectorstore

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]

    @staticmethod
    def _field_name_tokens(name: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]

    @staticmethod
    def _field_name_compact(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

    @staticmethod
    def _to_float(value: str | float | int | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        # Treat mixed alphanumeric identifiers (e.g., M01, J123, B02512) as non-numeric.
        if re.search(r"[A-Za-z]", text):
            return None
        cleaned = text.replace(",", "")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            return float(cleaned)
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        txt = (value or "").strip()
        if not txt:
            return None
        txt = txt.replace("T", " ").replace("Z", "").strip()
        txt = re.sub(r"\s+", " ", txt)
        # Keep only a likely datetime substring if noise exists around it.
        m = re.search(
            r"(20\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?)",
            txt,
        )
        if m:
            txt = m.group(1)
        txt = txt.replace("/", "-")
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue
        # Final relaxed attempt for ISO-like values.
        try:
            return datetime.fromisoformat(txt)
        except ValueError:
            return None

    @staticmethod
    def _extract_quoted_labels(question: str) -> List[str]:
        out: List[str] = []
        for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", question or ""):
            value = (a or b).strip()
            if value:
                out.append(value)
        return out

    @staticmethod
    def _detect_status(text: str) -> str:
        t = (text or "").lower()
        if re.search(r"\b(fail|failed|error|rejected)\b", t):
            return "failed"
        if re.search(r"\b(delay|delayed|late)\b", t):
            return "delayed"
        if re.search(r"\b(completed|complete|success|done|passed|ok)\b", t):
            return "completed"
        return ""

    def _collect_all_rows(self) -> List[Dict[str, object]]:
        dump = self.vectorstore.get(include=["documents", "metadatas"])
        docs = dump.get("documents", [])
        metas = dump.get("metadatas", [])

        rows: List[Dict[str, object]] = []
        for idx in range(len(docs)):
            meta = metas[idx] or {}
            if meta.get("doc_kind") != "categorical_row":
                continue
            payload = str(meta.get("row_payload", docs[idx][:220])).strip()
            try:
                fields = json.loads(str(meta.get("row_fields_json", "{}")))
            except Exception:
                fields = {}
            rows.append(
                {
                    "source": str(meta.get("source", "unknown")),
                    "snippet": payload,
                    "fields": fields,
                }
            )
        return self._unify_sparse_named_fields(rows)

    @staticmethod
    def _is_generic_col(name: str) -> bool:
        return bool(re.fullmatch(r"col_\d+", name or ""))

    @staticmethod
    def _is_generic_numeric(name: str) -> bool:
        return bool(re.fullmatch(r"numeric_\d+", name or ""))

    def _unify_sparse_named_fields(self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Backfill sparse semantic field names from high-coverage generic col_N fields.

        Some documents are indexed with named headers (e.g., gender/math_score), while other
        pages of the same dataset are indexed as generic col_1..col_n. This function infers an
        order-preserving alias map and propagates semantic names to all rows.
        """
        if not rows:
            return rows

        total_rows = len(rows)
        # Coverage of generic cols.
        col_coverage: Dict[str, int] = {}
        for row in rows:
            fields = row.get("fields", {}) or {}
            for k, v in fields.items():
                if self._is_generic_col(k) and str(v).strip():
                    col_coverage[k] = col_coverage.get(k, 0) + 1
        if not col_coverage:
            return rows

        # Choose reliable generic columns only.
        strong_cols = sorted(
            [k for k, c in col_coverage.items() if c / max(total_rows, 1) >= 0.5],
            key=lambda x: int(x.split("_")[1]),
        )
        if not strong_cols:
            return rows

        # Find the most common ordered sparse semantic schema among rows that do not use col_N.
        schema_orders: Dict[Tuple[str, ...], int] = {}
        for row in rows:
            fields = row.get("fields", {}) or {}
            keys_in_order = list(fields.keys())
            has_generic = any(self._is_generic_col(k) for k in keys_in_order)
            if has_generic:
                continue
            semantic_keys = [
                k
                for k in keys_in_order
                if k != "raw_record" and not self._is_generic_numeric(k)
            ]
            if len(semantic_keys) < 3:
                continue
            schema_orders[tuple(semantic_keys)] = schema_orders.get(tuple(semantic_keys), 0) + 1

        if not schema_orders:
            return rows

        sparse_schema = max(schema_orders.items(), key=lambda x: x[1])[0]
        if not sparse_schema:
            return rows

        # Build positional alias map: sparse semantic field i -> col_{i+1}.
        alias_map: Dict[str, str] = {}
        max_pairs = min(len(sparse_schema), len(strong_cols))
        for i in range(max_pairs):
            alias_map[sparse_schema[i]] = strong_cols[i]

        if not alias_map:
            return rows

        # Backfill semantic names from generic columns on rows where semantic values are missing.
        for row in rows:
            fields = row.get("fields", {}) or {}
            for semantic_name, col_name in alias_map.items():
                current = str(fields.get(semantic_name, "")).strip()
                if current:
                    continue
                col_val = str(fields.get(col_name, "")).strip()
                if col_val:
                    fields[semantic_name] = col_val
            row["fields"] = fields
        return rows

    def _build_schema(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        profiles: Dict[str, Dict[str, object]] = {}
        for row in rows:
            fields = row.get("fields", {}) or {}
            for key, raw in fields.items():
                name = str(key)
                profiles.setdefault(
                    name,
                    {
                        "count": 0,
                        "numeric_count": 0,
                        "datetime_count": 0,
                        "examples": [],
                        "unique": set(),
                    },
                )
                p = profiles[name]
                p["count"] += 1
                value = "" if raw is None else str(raw).strip()
                if value:
                    p["unique"].add(value)
                if len(p["examples"]) < 5 and value:
                    p["examples"].append(value)
                if self._to_float(value) is not None:
                    p["numeric_count"] += 1
                if self._parse_datetime(value) is not None:
                    p["datetime_count"] += 1

        field_meta: Dict[str, Dict[str, object]] = {}
        for name, p in profiles.items():
            total = max(int(p["count"]), 1)
            field_meta[name] = {
                "count": total,
                "numeric_ratio": round(p["numeric_count"] / total, 4),
                "datetime_ratio": round(p["datetime_count"] / total, 4),
                "unique_count": len(p["unique"]),
                "examples": p["examples"],
            }

        numeric_fields: List[str] = []
        datetime_fields: List[str] = []
        categorical_fields: List[str] = []
        measure_fields: List[str] = []
        identifier_fields: List[str] = []
        status_candidates: List[str] = []
        scheduled_start_candidates: List[str] = []
        actual_start_candidates: List[str] = []
        dimension_fields: List[str] = []

        for name, meta in field_meta.items():
            tokens = self._field_name_tokens(name)
            compact = self._field_name_compact(name)
            if meta["numeric_ratio"] >= 0.6:
                numeric_fields.append(name)
                is_identifier = any(t in tokens for t in ["id", "code", "number", "num"])
                if not is_identifier:
                    measure_fields.append(name)
            has_datetime_hint = any(t in tokens for t in ["date", "time", "timestamp", "datetime"]) or any(
                kw in compact for kw in ["date", "time", "timestamp", "datetime"]
            )
            if meta["datetime_ratio"] >= 0.6 or (has_datetime_hint and meta["datetime_ratio"] >= 0.3):
                datetime_fields.append(name)
            if 2 <= meta["unique_count"] <= 120 and meta["count"] >= 5:
                categorical_fields.append(name)
            if 2 <= meta["unique_count"] <= 60 and meta["count"] >= 5 and meta["numeric_ratio"] < 0.5:
                dimension_fields.append(name)
            unique_ratio = meta["unique_count"] / max(meta["count"], 1)
            if any(t in tokens for t in ["id", "code", "number", "num", "key"]) or (
                unique_ratio >= 0.5 and unique_ratio <= 1.0 and meta["numeric_ratio"] < 0.4
            ):
                identifier_fields.append(name)

            if any(t in tokens for t in ["status", "state", "result", "outcome"]) or any(
                kw in compact for kw in ["status", "state", "result", "outcome"]
            ):
                status_candidates.append(name)
            status_hits = sum(1 for ex in meta.get("examples", []) if self._detect_status(str(ex)))
            if status_hits >= 2 or (status_hits >= 1 and meta["unique_count"] <= 12):
                status_candidates.append(name)
            has_sched = any(t in tokens for t in ["scheduled", "schedule", "planned", "sched"]) or any(
                kw in compact for kw in ["scheduled", "schedule", "planned", "sched"]
            )
            has_start_like = any(t in tokens for t in ["start", "time", "date"]) or any(
                kw in compact for kw in ["start", "time", "date"]
            )
            if has_sched and has_start_like:
                scheduled_start_candidates.append(name)
            has_actual = ("actual" in tokens) or ("actual" in compact)
            if has_actual and has_start_like:
                actual_start_candidates.append(name)

        return {
            "fields": field_meta,
            "numeric_fields": sorted(numeric_fields),
            "datetime_fields": sorted(datetime_fields),
            "categorical_fields": sorted(categorical_fields),
            "measure_fields": sorted(measure_fields),
            "identifier_fields": sorted(list(dict.fromkeys(identifier_fields))),
            "dimension_fields": sorted(dimension_fields),
            "candidates": {
                "status": sorted(list(dict.fromkeys(status_candidates))),
                "scheduled_start": sorted(scheduled_start_candidates),
                "actual_start": sorted(actual_start_candidates),
            },
        }

    @staticmethod
    def _pick_first(values: List[str]) -> str:
        return values[0] if values else ""

    def _extract_query_date(self, question: str) -> datetime | None:
        q = question.strip()
        m_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
        if m_iso:
            return self._parse_datetime(m_iso.group(1))
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

    def _row_datetime(self, row: Dict[str, object], datetime_fields: List[str]) -> datetime | None:
        fields = row.get("fields", {}) or {}
        for name in datetime_fields:
            dt = self._parse_datetime(str(fields.get(name, "")))
            if dt:
                return dt
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", str(row.get("snippet", "")))
        if m:
            return self._parse_datetime(m.group(1))
        return None

    def _extract_literal_filters(
        self,
        question: str,
        rows: List[Dict[str, object]],
        dimension_fields: List[str],
    ) -> Dict[str, List[str]]:
        q_lower = (question or "").lower()
        q_tokens = [t for t in self._tokenize(question) if len(t) >= 3]
        scored: List[Tuple[str, List[str], int, int]] = []
        for field in dimension_fields:
            values = set()
            non_empty = 0
            for row in rows[:1000]:
                v = str((row.get("fields", {}) or {}).get(field, "")).strip()
                if v:
                    values.add(v)
                    non_empty += 1
                if len(values) >= 200:
                    break
            matches: List[str] = []
            for v in values:
                vl = v.lower()
                if vl in q_lower:
                    matches.append(v)
                    continue
                # Also match when question tokens appear inside compact labels
                # (e.g., "female" should match "femalegroup").
                if any(tok in vl for tok in q_tokens):
                    matches.append(v)
            if matches:
                matched_rows = 0
                match_set = {m.strip() for m in matches}
                for row in rows:
                    rv = str((row.get("fields", {}) or {}).get(field, "")).strip()
                    if rv in match_set:
                        matched_rows += 1
                scored.append((field, sorted(matches), matched_rows, non_empty))

        if not scored:
            return {}

        # Prefer the field that gives the broadest reliable coverage.
        # This prevents sparse alias fields from shrinking the whole dataset.
        scored.sort(key=lambda x: (x[2], x[3], len(x[1])), reverse=True)
        best_field, best_matches, _, _ = scored[0]
        return {best_field: best_matches}

    @staticmethod
    def _normalize_scope_state(scope: Dict[str, object]) -> Dict[str, object]:
        return {
            "filters": scope.get("filters", {}) or {},
            "date": str(scope.get("date", "") or "").strip(),
        }

    @staticmethod
    def _scope_has_filters(scope_state: Dict[str, object]) -> bool:
        return bool(scope_state.get("filters") or scope_state.get("date"))

    @staticmethod
    def _should_reset_scope(question: str) -> bool:
        q = (question or "").lower()
        return any(
            token in q
            for token in ["overall", "global", "all data", "entire dataset", "everything", "no filter"]
        )

    @staticmethod
    def _should_reuse_prior_scope(question: str) -> bool:
        q = (question or "").lower()
        follow_up_markers = [
            "same",
            "those",
            "them",
            "that one",
            "that machine",
            "that group",
            "previous",
            "earlier",
            "again",
            "for it",
            "for those",
        ]
        return any(marker in q for marker in follow_up_markers)

    def _infer_scope(
        self,
        rows: List[Dict[str, object]],
        question: str,
        schema: Dict[str, object],
        prior_scope: Dict[str, object] | None = None,
    ) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
        prior = self._normalize_scope_state(prior_scope or {})
        if prior and not self._should_reuse_prior_scope(question):
            prior = {"filters": {}, "date": ""}
        if self._should_reset_scope(question):
            prior = {"filters": {}, "date": ""}

        dimension_fields = schema.get("dimension_fields", []) or []
        datetime_fields = schema.get("datetime_fields", []) or []
        literal_filters = self._extract_literal_filters(question, rows, dimension_fields)
        q_date = self._extract_query_date(question)
        date_value = q_date.strftime("%Y-%m-%d") if q_date else str(prior.get("date", ""))

        filters = literal_filters if literal_filters else (prior.get("filters", {}) or {})

        scoped = rows
        for field, allowed in filters.items():
            allowed_set = {str(x).strip() for x in allowed}
            scoped = [
                row
                for row in scoped
                if str((row.get("fields", {}) or {}).get(field, "")).strip() in allowed_set
            ]

        if date_value:
            target = self._parse_datetime(date_value)
            if target:
                scoped = [
                    row
                    for row in scoped
                    if (self._row_datetime(row, datetime_fields) is not None)
                    and (self._row_datetime(row, datetime_fields).date() == target.date())
                ]

        scope = {
            "filters": filters,
            "date": date_value,
            "rows_after_filter": len(scoped),
        }
        return scoped, scope, self._normalize_scope_state(scope)

    def _summarize(self, rows: List[Dict[str, object]], schema: Dict[str, object]) -> Dict[str, object]:
        datetime_fields = schema.get("datetime_fields", []) or []
        categorical_fields = schema.get("categorical_fields", []) or []
        numeric_fields = schema.get("numeric_fields", []) or []

        dates: List[datetime] = []
        for row in rows:
            dt = self._row_datetime(row, datetime_fields)
            if dt:
                dates.append(dt)

        top_dimensions: Dict[str, Dict[str, int]] = {}
        for field in categorical_fields[:6]:
            counts: Dict[str, int] = {}
            for row in rows:
                v = str((row.get("fields", {}) or {}).get(field, "")).strip()
                if not v:
                    continue
                counts[v] = counts.get(v, 0) + 1
            if counts:
                top_dimensions[field] = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20])

        numeric_stats: Dict[str, Dict[str, float]] = {}
        for field in numeric_fields:
            values: List[float] = []
            for row in rows:
                v = self._to_float((row.get("fields", {}) or {}).get(field))
                if v is not None:
                    values.append(v)
            if values:
                numeric_stats[field] = {
                    "count": len(values),
                    "sum": round(sum(values), 6),
                    "avg": round(sum(values) / len(values), 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }

        return {
            "rows": len(rows),
            "date_min": min(dates).strftime("%Y-%m-%d") if dates else "",
            "date_max": max(dates).strftime("%Y-%m-%d") if dates else "",
            "categorical_counts": top_dimensions,
            "numeric_stats": numeric_stats,
        }

    @staticmethod
    def _group_key(dt: datetime, grain: str) -> str:
        if grain == "hour":
            return dt.strftime("%Y-%m-%d %H:00")
        if grain == "day":
            return dt.strftime("%Y-%m-%d")
        if grain == "month":
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")

    def _group_sum(
        self,
        rows: List[Dict[str, object]],
        datetime_field: str,
        value_field: str,
        grain: str,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        if not datetime_field or not value_field:
            return totals
        for row in rows:
            fields = row.get("fields", {}) or {}
            dt = self._parse_datetime(str(fields.get(datetime_field, "")))
            if not dt:
                continue
            value = self._to_float(fields.get(value_field))
            if value is None:
                continue
            key = self._group_key(dt, grain)
            totals[key] = totals.get(key, 0.0) + value
        return {k: round(v, 6) for k, v in sorted(totals.items())}

    @staticmethod
    def _sample_variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    @staticmethod
    def _trend_slope(points: List[float]) -> float:
        if len(points) < 2:
            return 0.0
        n = len(points)
        x_mean = (n - 1) / 2.0
        y_mean = sum(points) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(points):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx
        return (num / den) if den > 0 else 0.0

    @staticmethod
    def _trend_direction(slope: float) -> str:
        if slope > 0:
            return "increasing"
        if slope < 0:
            return "decreasing"
        return "flat"

    @staticmethod
    def _split_multi_value(value: str) -> List[str]:
        text = (value or "").strip()
        if not text:
            return []
        parts = re.split(r",|/|\|", text)
        out = [p.strip() for p in parts if p and p.strip()]
        return out if out else [text]

    @staticmethod
    def _normalize_text_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    def _question_labels(self, question: str) -> List[str]:
        quoted = [self._normalize_text_token(x) for x in self._extract_quoted_labels(question)]
        raw_tokens = [self._normalize_text_token(t) for t in self._tokenize(question)]
        stop = {
            "what", "which", "how", "many", "much", "total", "average", "mean", "median",
            "variance", "standard", "deviation", "across", "all", "there", "are", "is",
            "in", "on", "of", "the", "and", "or", "for", "to", "with", "than", "their",
            "its", "does", "do", "did", "from", "during", "week", "weeks", "month", "day",
            "hour", "dataset", "records", "entries",
        }
        tokens = [t for t in raw_tokens if len(t) >= 3 and t not in stop]
        out: List[str] = []
        for t in quoted + tokens:
            if t and t not in out:
                out.append(t)
        return out[:8]

    @staticmethod
    def _token_match_score(a_tokens: List[str], b_tokens: List[str]) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        bset = set(b_tokens)
        exact = sum(1 for t in a_tokens if t in bset)
        partial = sum(
            1
            for t in a_tokens
            for b in bset
            if t != b and len(t) >= 4 and (t in b or b in t)
        )
        return float(exact) + (0.25 * float(partial))

    def _field_question_score(
        self,
        field_name: str,
        question_tokens: List[str],
        field_meta: Dict[str, object] | None = None,
    ) -> float:
        if not question_tokens:
            return 0.0
        name_tokens = self._field_name_tokens(field_name)
        score = self._token_match_score(name_tokens, question_tokens)
        examples = (field_meta or {}).get("examples", []) if field_meta else []
        if examples:
            ex_text = " ".join(str(x).lower() for x in examples)
            example_hits = sum(1 for q in question_tokens if q and q in ex_text)
            score += 0.15 * float(example_hits)
        if re.fullmatch(r"col_\d+", field_name or ""):
            score -= 0.2
        return score

    def _rank_numeric_fields_by_question(
        self,
        question: str,
        schema: Dict[str, object],
    ) -> List[str]:
        numeric_fields = schema.get("numeric_fields", []) or []
        field_meta = schema.get("fields", {}) or {}
        q_tokens = self._tokenize(question)
        ranked: List[Tuple[str, float]] = []
        for field in numeric_fields:
            meta = field_meta.get(field, {}) or {}
            coverage = float(meta.get("count", 0))
            coverage_hint = min(coverage / 1000.0, 1.0)
            score = self._field_question_score(field, q_tokens, meta) + coverage_hint
            if not self._field_name_compact(field).isdigit():
                score += 0.1
            ranked.append((field, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in ranked]

    def _rank_dimension_fields_by_question(
        self,
        question: str,
        schema: Dict[str, object],
    ) -> List[str]:
        dim_fields = schema.get("dimension_fields", []) or []
        field_meta = schema.get("fields", {}) or {}
        q_tokens = self._tokenize(question)
        ranked: List[Tuple[str, float]] = []
        for field in dim_fields:
            meta = field_meta.get(field, {}) or {}
            score = self._field_question_score(field, q_tokens, meta)
            if not re.fullmatch(r"col_\d+", field or ""):
                score += 0.1
            ranked.append((field, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in ranked]

    def _compute_question_label_distribution(
        self,
        question: str,
        rows: List[Dict[str, object]],
        schema: Dict[str, object],
    ) -> Dict[str, object]:
        labels = self._question_labels(question)
        result: Dict[str, object] = {
            "labels": labels,
            "field_scores": [],
            "selected_fields": [],
            "coalesced_counts": {},
            "coverage_rows": 0,
            "total_rows": len(rows),
        }
        if len(labels) < 2:
            return result

        candidate_fields = list(
            dict.fromkeys(
                (schema.get("dimension_fields", []) or []) + (schema.get("categorical_fields", []) or [])
            )
        )[:20]
        if not candidate_fields:
            return result

        per_field: List[Dict[str, object]] = []
        norm_labels = [self._normalize_text_token(l) for l in labels if l]
        for field in candidate_fields:
            label_counts = {l: 0 for l in norm_labels}
            non_empty = 0
            for row in rows:
                value = str((row.get("fields", {}) or {}).get(field, "")).strip()
                if not value:
                    continue
                non_empty += 1
                norm_value = self._normalize_text_token(value)
                parts = [self._normalize_text_token(p) for p in self._split_multi_value(value)]
                for lbl in norm_labels:
                    if not lbl:
                        continue
                    if lbl == norm_value or lbl in parts or (lbl in norm_value and len(lbl) >= 4):
                        label_counts[lbl] += 1
            score = sum(label_counts.values())
            if score > 0:
                per_field.append(
                    {
                        "field": field,
                        "score": score,
                        "non_empty_rows": non_empty,
                        "label_counts": label_counts,
                    }
                )
        if not per_field:
            return result

        per_field.sort(key=lambda x: (x["score"], x["non_empty_rows"]), reverse=True)
        selected = per_field[:3]

        coalesced = {l: 0 for l in norm_labels}
        coverage = 0
        for row in rows:
            matched = False
            for field_info in selected:
                field = str(field_info["field"])
                value = str((row.get("fields", {}) or {}).get(field, "")).strip()
                if not value:
                    continue
                norm_value = self._normalize_text_token(value)
                parts = [self._normalize_text_token(p) for p in self._split_multi_value(value)]
                local_hit = False
                for lbl in norm_labels:
                    if lbl == norm_value or lbl in parts or (lbl in norm_value and len(lbl) >= 4):
                        coalesced[lbl] += 1
                        local_hit = True
                if local_hit:
                    matched = True
                    break
            if matched:
                coverage += 1

        result["field_scores"] = per_field[:10]
        result["selected_fields"] = [x["field"] for x in selected]
        result["coalesced_counts"] = coalesced
        result["coverage_rows"] = coverage
        return result

    def _compute_label_numeric_performance(
        self,
        label_distribution: Dict[str, object],
        rows: List[Dict[str, object]],
        schema: Dict[str, object],
    ) -> Dict[str, object]:
        labels = [self._normalize_text_token(x) for x in (label_distribution.get("labels") or [])]
        selected_fields = [str(x) for x in (label_distribution.get("selected_fields") or [])]
        if len(labels) < 2 or not selected_fields:
            return {}

        measure_hint = " ".join(str(x) for x in (label_distribution.get("labels") or []))
        ranked_numeric = self._rank_numeric_fields_by_question(measure_hint, schema)
        measures = [f for f in ranked_numeric if not self._field_name_compact(f).isdigit()][:6]
        if not measures:
            measures = ranked_numeric[:6]
        if not measures:
            return {}

        by_label_values: Dict[str, List[float]] = {lbl: [] for lbl in labels}
        by_label_measure: Dict[str, Dict[str, Dict[str, float]]] = {}
        coverage_rows = 0

        for row in rows:
            fields = row.get("fields", {}) or {}
            matched_label = ""
            for field in selected_fields:
                raw = str(fields.get(field, "")).strip()
                if not raw:
                    continue
                norm = self._normalize_text_token(raw)
                parts = [self._normalize_text_token(p) for p in self._split_multi_value(raw)]
                for lbl in labels:
                    if lbl == norm or lbl in parts or (lbl in norm and len(lbl) >= 4):
                        matched_label = lbl
                        break
                if matched_label:
                    break
            if not matched_label:
                continue

            local_values: List[float] = []
            for measure in measures:
                v = self._to_float(fields.get(measure))
                if v is None:
                    continue
                local_values.append(v)
                agg = by_label_measure.setdefault(matched_label, {}).setdefault(measure, {"sum": 0.0, "count": 0.0})
                agg["sum"] += v
                agg["count"] += 1.0
            if local_values:
                coverage_rows += 1
                by_label_values[matched_label].append(sum(local_values) / len(local_values))

        label_summary: Dict[str, Dict[str, float]] = {}
        for lbl, vals in by_label_values.items():
            label_summary[lbl] = self._compute_basic_math(vals) if vals else {
                "count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "variance": 0.0, "std_dev": 0.0
            }

        ranked = sorted(
            ((lbl, stats.get("avg", 0.0), int(stats.get("count", 0))) for lbl, stats in label_summary.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        per_measure_avg: List[Dict[str, object]] = []
        for lbl, measure_map in by_label_measure.items():
            for measure, agg in measure_map.items():
                if agg["count"] <= 0:
                    continue
                per_measure_avg.append(
                    {
                        "label": lbl,
                        "measure_field": measure,
                        "avg": round(agg["sum"] / agg["count"], 6),
                        "count": int(agg["count"]),
                    }
                )

        return {
            "labels": labels,
            "selected_fields": selected_fields,
            "candidate_measures": measures,
            "coverage_rows": coverage_rows,
            "label_summary": label_summary,
            "ranked_labels_by_avg": ranked,
            "best_label_by_avg": ranked[0][0] if ranked else "",
            "per_measure_avg": per_measure_avg,
        }

    @staticmethod
    def _parse_duration_to_minutes(value: str) -> float | None:
        txt = (value or "").strip().lower()
        if not txt:
            return None
        m_num = re.search(r"\d+(?:\.\d+)?", txt)
        if not m_num:
            return None
        num = float(m_num.group(0))
        if "season" in txt:
            # Approximate one season as 8 episodes * 45 minutes.
            return num * 360.0
        if "hour" in txt:
            return num * 60.0
        return num

    def _pick_measure_field(self, question: str, schema: Dict[str, object]) -> Tuple[str, str]:
        measure_fields = schema.get("measure_fields", []) or []
        numeric_fields = schema.get("numeric_fields", []) or []
        field_meta = schema.get("fields", {}) or {}
        q_tokens = self._tokenize(question)
        if len(measure_fields) == 1:
            return measure_fields[0], "single_measure"
        ranked_numeric = self._rank_numeric_fields_by_question(question, schema)
        candidate_pool = [f for f in ranked_numeric if f in set(measure_fields + numeric_fields)]
        if candidate_pool:
            best = candidate_pool[0]
            best_meta = field_meta.get(best, {}) or {}
            best_score = self._field_question_score(best, q_tokens, best_meta)
            if q_tokens and best_score < 0.35:
                return "", "low_confidence"
            for field in candidate_pool:
                if not self._field_name_compact(field).isdigit():
                    return field, "question_relevance"
            return candidate_pool[0], "question_relevance"
        if measure_fields:
            return measure_fields[0], "schema_measure"
        for field in ranked_numeric:
            if not self._field_name_compact(field).isdigit():
                return field, "numeric_fallback"
        return "", "none"

    def _pick_group_field(
        self,
        question: str,
        schema: Dict[str, object],
    ) -> Tuple[str, str]:
        dimension_fields = schema.get("dimension_fields", []) or []
        field_meta = schema.get("fields", {}) or {}
        q_tokens = self._tokenize(question)
        if not dimension_fields:
            return "", "none"
        ranked_dims = self._rank_dimension_fields_by_question(question, schema)
        if ranked_dims:
            best = ranked_dims[0]
            best_score = self._field_question_score(best, q_tokens, field_meta.get(best, {}) or {})
            if q_tokens and best_score < 0.35:
                return "", "low_confidence"
            return ranked_dims[0], "question_relevance"
        return dimension_fields[0], "schema_dimension"

    @staticmethod
    def _extract_numeric_values(rows: List[Dict[str, object]], field: str) -> List[float]:
        values: List[float] = []
        if not field:
            return values
        for row in rows:
            v = IntelligenceLayer._to_float((row.get("fields", {}) or {}).get(field))
            if v is not None:
                values.append(v)
        return values

    @staticmethod
    def _compute_basic_math(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
            }
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[(n // 2) - 1] + sorted_vals[n // 2]) / 2.0
        )
        mean = sum(sorted_vals) / n
        variance = (
            sum((v - mean) ** 2 for v in sorted_vals) / (n - 1)
            if n > 1
            else 0.0
        )
        return {
            "count": n,
            "sum": round(sum(sorted_vals), 6),
            "avg": round(mean, 6),
            "min": round(min(sorted_vals), 6),
            "max": round(max(sorted_vals), 6),
            "median": round(median, 6),
            "variance": round(variance, 6),
            "std_dev": round(math.sqrt(variance), 6),
        }

    def _compute_grouped_math(
        self,
        rows: List[Dict[str, object]],
        group_field: str,
        value_field: str,
    ) -> Dict[str, object]:
        result = {"group_field": group_field, "value_field": value_field, "groups": {}}
        if not group_field or not value_field:
            return result
        buckets: Dict[str, List[float]] = {}
        for row in rows:
            fields = row.get("fields", {}) or {}
            group_value = str(fields.get(group_field, "")).strip()
            if not group_value:
                continue
            value = self._to_float(fields.get(value_field))
            if value is None:
                continue
            buckets.setdefault(group_value, []).append(value)
        for key, vals in buckets.items():
            result["groups"][key] = self._compute_basic_math(vals)
        return result

    @staticmethod
    def _compute_group_percentages(group_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        totals = {k: v.get("sum", 0.0) for k, v in group_stats.items()}
        grand_total = sum(totals.values())
        if grand_total <= 0:
            return {}
        return {
            k: round((v / grand_total) * 100.0, 6)
            for k, v in totals.items()
        }

    def _derive_dynamic_metrics(
        self,
        question: str,
        rows: List[Dict[str, object]],
        schema: Dict[str, object],
    ) -> List[Dict[str, object]]:
        metrics: List[Dict[str, object]] = []

        candidates = schema.get("candidates", {}) or {}
        datetime_fields = schema.get("datetime_fields", []) or []
        categorical_fields = schema.get("categorical_fields", []) or []
        dimension_fields = schema.get("dimension_fields", []) or []
        measure_fields = schema.get("measure_fields", []) or []

        status_field = self._pick_first(candidates.get("status", []))
        scheduled_start = self._pick_first(candidates.get("scheduled_start", []))
        actual_start = self._pick_first(candidates.get("actual_start", []))
        datetime_field = datetime_fields[0] if datetime_fields else ""
        primary_measure, primary_reason = self._pick_measure_field(question, schema)
        ranked_numeric = self._rank_numeric_fields_by_question(question, schema)
        selected_measures = [m for m in [primary_measure] if m]
        for m in ranked_numeric:
            if m not in selected_measures and not self._field_name_compact(m).isdigit():
                selected_measures.append(m)
            if len(selected_measures) >= 3:
                break
        if not selected_measures:
            for m in measure_fields:
                if m not in selected_measures:
                    selected_measures.append(m)
                if len(selected_measures) >= 3:
                    break
        ranked_dims = self._rank_dimension_fields_by_question(question, schema)
        selected_group_fields = ranked_dims[:3] if ranked_dims else dimension_fields[:3]

        if scheduled_start and actual_start:
            data = {
                "scheduled_field": scheduled_start,
                "actual_field": actual_start,
                "count": 0,
                "early": 0,
                "on_time": 0,
                "late": 0,
                "early_pct": 0.0,
                "on_time_pct": 0.0,
                "late_pct": 0.0,
            }
            for row in rows:
                fields = row.get("fields", {}) or {}
                a = self._parse_datetime(str(fields.get(actual_start, "")))
                s = self._parse_datetime(str(fields.get(scheduled_start, "")))
                if not (a and s):
                    continue
                data["count"] += 1
                diff = (a - s).total_seconds()
                if diff == 0:
                    data["on_time"] += 1
                elif diff > 0:
                    data["late"] += 1
                else:
                    data["early"] += 1
            if data["count"] > 0:
                total = float(data["count"])
                data["early_pct"] = round((data["early"] / total) * 100.0, 4)
                data["on_time_pct"] = round((data["on_time"] / total) * 100.0, 4)
                data["late_pct"] = round((data["late"] / total) * 100.0, 4)
            metrics.append({"metric": "start_time_adherence", "data": data})

        for measure in selected_measures:
            values = self._extract_numeric_values(rows, measure)
            metrics.append(
                {
                    "metric": "basic_math_summary",
                    "data": {
                        "value_field": measure,
                        "stats": self._compute_basic_math(values),
                    },
                    "measure_reason": primary_reason if measure == primary_measure else "schema_measure",
                }
            )

        # Precompute grouped summaries for top dimensions and measures, independent of query keywords.
        for group_field in selected_group_fields:
            for measure in selected_measures[:2]:
                grouped = self._compute_grouped_math(rows, group_field, measure)
                group_stats = grouped.get("groups", {})
                if not isinstance(group_stats, dict) or not group_stats:
                    continue
                ranked = sorted(
                    ((g, s.get("sum", 0.0)) for g, s in group_stats.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )
                top_group = ranked[0][0]
                top_sum = ranked[0][1]
                trimmed_keys = {g for g, _ in ranked[:20]}
                trimmed_stats = {k: v for k, v in group_stats.items() if k in trimmed_keys}
                percentages = self._compute_group_percentages(trimmed_stats)
                metrics.append(
                    {
                        "metric": "grouped_math_summary",
                        "data": {
                            "group_field": group_field,
                            "value_field": measure,
                            "groups": trimmed_stats,
                            "group_percentages_by_sum": percentages,
                            "top_group_by_sum": top_group,
                            "top_group_sum": top_sum,
                        },
                        "measure_reason": primary_reason if measure == primary_measure else "schema_measure",
                    }
                )

        if datetime_field and selected_measures:
            for grain in ["hour", "day", "month"]:
                totals = self._group_sum(rows, datetime_field, selected_measures[0], grain)
                if totals:
                    metrics.append(
                        {
                            "metric": "time_series_sum",
                            "data": {
                                "datetime_field": datetime_field,
                                "value_field": selected_measures[0],
                                "grain": grain,
                                "totals": totals,
                                "variance": round(self._sample_variance(list(totals.values())), 6),
                            },
                            "measure_reason": primary_reason,
                        }
                    )

        # Generic status distribution over top dimensions when status can be inferred.
        if status_field:
            for group_field in dimension_fields[:3]:
                buckets: Dict[str, Dict[str, int]] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    group_val = str(fields.get(group_field, "")).strip()
                    if not group_val:
                        continue
                    status_val = str(fields.get(status_field, "")).strip().lower()
                    if not status_val:
                        status_val = self._detect_status(str(row.get("snippet", "")))
                    if not status_val:
                        continue
                    b = buckets.setdefault(group_val, {"failed": 0, "delayed": 0, "completed": 0, "other": 0, "total": 0})
                    b["total"] += 1
                    if "fail" in status_val:
                        b["failed"] += 1
                    elif "delay" in status_val:
                        b["delayed"] += 1
                    elif any(x in status_val for x in ["complete", "success", "done", "pass", "ok"]):
                        b["completed"] += 1
                    else:
                        b["other"] += 1
                if buckets:
                    for k, b in buckets.items():
                        total = b["total"]
                        b["failed_pct"] = round((b["failed"] / total) * 100.0, 4) if total else 0.0
                        b["delayed_pct"] = round((b["delayed"] / total) * 100.0, 4) if total else 0.0
                    metrics.append(
                        {
                            "metric": "status_by_dimension",
                            "data": {"group_field": group_field, "status_field": status_field, "groups": buckets},
                        }
                    )

        labels = self._extract_quoted_labels(question)
        if labels:
            label_field = ""
            for field in categorical_fields:
                values = set()
                for row in rows[:500]:
                    value = str((row.get("fields", {}) or {}).get(field, "")).strip()
                    if value:
                        values.add(value)
                    if len(values) > 60:
                        break
                if any(lbl.lower() in v.lower() for lbl in labels for v in values):
                    label_field = field
                    break
            buckets: Dict[str, Dict[str, float]] = {}
            if label_field:
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    label_value = str(fields.get(label_field, "")).strip()
                    if not label_value:
                        continue
                    status_val = str(fields.get(status_field, "")).strip().lower() if status_field else ""
                    if not status_val:
                        status_val = self._detect_status(str(row.get("snippet", "")))
                    for lbl in labels:
                        if lbl.lower() in label_value.lower():
                            b = buckets.setdefault(lbl, {"failed": 0, "delayed": 0, "total": 0})
                            b["total"] += 1
                            if "fail" in status_val:
                                b["failed"] += 1
                            elif "delay" in status_val:
                                b["delayed"] += 1
                for lbl, b in buckets.items():
                    total = b["total"]
                    b["failure_or_delay_pct"] = round(((b["failed"] + b["delayed"]) / total) * 100.0, 4) if total else 0.0
            metrics.append(
                {
                    "metric": "label_failure_probability",
                    "data": {"label_field": label_field, "status_field": status_field, "labels": buckets},
                }
            )

        label_distribution = self._compute_question_label_distribution(question, rows, schema)
        if label_distribution.get("coalesced_counts"):
            metrics.append(
                {
                    "metric": "question_label_distribution",
                    "data": label_distribution,
                }
            )
            label_perf = self._compute_label_numeric_performance(label_distribution, rows, schema)
            if label_perf:
                metrics.append(
                    {
                        "metric": "label_numeric_performance",
                        "data": label_perf,
                    }
                )

        return metrics

    def _build_universal_analytics(
        self,
        rows: List[Dict[str, object]],
        schema: Dict[str, object],
    ) -> Dict[str, object]:
        """Build a question-agnostic analytics catalog over the entire scoped rows."""
        numeric_fields = schema.get("numeric_fields", []) or []
        dimension_fields = schema.get("dimension_fields", []) or []
        datetime_fields = schema.get("datetime_fields", []) or []
        measure_fields = schema.get("measure_fields", []) or []
        identifier_fields = schema.get("identifier_fields", []) or []
        candidates = schema.get("candidates", {}) or {}
        status_field = self._pick_first(candidates.get("status", []))

        # Field-level quality snapshot.
        field_quality: Dict[str, Dict[str, object]] = {}
        for name, meta in (schema.get("fields", {}) or {}).items():
            field_quality[name] = {
                "count": meta.get("count", 0),
                "numeric_ratio": meta.get("numeric_ratio", 0.0),
                "datetime_ratio": meta.get("datetime_ratio", 0.0),
                "unique_count": meta.get("unique_count", 0),
                "examples": meta.get("examples", []),
            }

        # Numeric summary for all measures + a few extra numeric fields.
        primary_numeric = list(dict.fromkeys((measure_fields[:6] + numeric_fields[:8])))
        numeric_summary: Dict[str, Dict[str, float]] = {}
        for field in primary_numeric:
            values = self._extract_numeric_values(rows, field)
            if values:
                numeric_summary[field] = self._compute_basic_math(values)

        # Coverage-aware numeric metadata to avoid sparse-field bias.
        numeric_field_coverage: Dict[str, Dict[str, float]] = {}
        for field in numeric_fields:
            vals = self._extract_numeric_values(rows, field)
            if not vals:
                continue
            coverage_ratio = len(vals) / max(len(rows), 1)
            numeric_field_coverage[field] = {
                "count": float(len(vals)),
                "coverage_ratio": round(coverage_ratio, 6),
                "avg": round(sum(vals) / len(vals), 6),
            }

        high_coverage_numeric_fields = [
            f for f, m in numeric_field_coverage.items() if m["coverage_ratio"] >= 0.7
        ]

        # Dataset-wide composites from high-coverage fields (query-agnostic).
        dataset_wide_composites: Dict[str, object] = {
            "field_selection_basis": "high_coverage_numeric_fields",
            "selected_fields": high_coverage_numeric_fields[:8],
            "row_level_average_mean": 0.0,
            "field_average_mean": 0.0,
            "row_coverage_count": 0,
        }
        if high_coverage_numeric_fields:
            row_avgs: List[float] = []
            for row in rows:
                fields = row.get("fields", {}) or {}
                local_vals: List[float] = []
                for f in high_coverage_numeric_fields:
                    v = self._to_float(fields.get(f))
                    if v is not None:
                        local_vals.append(v)
                if local_vals:
                    row_avgs.append(sum(local_vals) / len(local_vals))
            if row_avgs:
                dataset_wide_composites["row_level_average_mean"] = round(sum(row_avgs) / len(row_avgs), 6)
                dataset_wide_composites["row_coverage_count"] = len(row_avgs)
            field_means = [numeric_field_coverage[f]["avg"] for f in high_coverage_numeric_fields if f in numeric_field_coverage]
            if field_means:
                dataset_wide_composites["field_average_mean"] = round(sum(field_means) / len(field_means), 6)

        # Categorical distributions for top dimensions.
        categorical_distributions: Dict[str, Dict[str, int]] = {}
        for field in dimension_fields[:6]:
            counts: Dict[str, int] = {}
            for row in rows:
                value = str((row.get("fields", {}) or {}).get(field, "")).strip()
                if not value:
                    continue
                counts[value] = counts.get(value, 0) + 1
            if counts:
                ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                categorical_distributions[field] = dict(ranked[:40])

        # Grouped numeric aggregations (dimension x measure).
        grouped_aggregations: List[Dict[str, object]] = []
        for dim in dimension_fields[:4]:
            for measure in primary_numeric[:4]:
                grouped = self._compute_grouped_math(rows, dim, measure).get("groups", {})
                if not isinstance(grouped, dict) or not grouped:
                    continue
                ranked_sum = sorted(
                    ((k, v.get("sum", 0.0)) for k, v in grouped.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                ranked_avg = sorted(
                    ((k, v.get("avg", 0.0)) for k, v in grouped.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                grouped_aggregations.append(
                    {
                        "dimension_field": dim,
                        "measure_field": measure,
                        "group_count": len(grouped),
                        "top_by_sum": ranked_sum[:10],
                        "top_by_avg": ranked_avg[:10],
                    }
                )

        # Pure categorical counts per dimension value (used for "how many X in category Y?" questions).
        dimension_value_counts: Dict[str, Dict[str, int]] = {}
        for dim in dimension_fields[:8]:
            counts: Dict[str, int] = {}
            for row in rows:
                value = str((row.get("fields", {}) or {}).get(dim, "")).strip()
                if not value:
                    continue
                counts[value] = counts.get(value, 0) + 1
            if counts:
                ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                dimension_value_counts[dim] = dict(ranked[:80])

        # Distinct identifier counts by dimension (e.g., distinct job_id per machine_id).
        distinct_identifier_counts_by_dimension: List[Dict[str, object]] = []
        for dim in dimension_fields[:6]:
            for ident in identifier_fields[:6]:
                if dim == ident:
                    continue
                buckets: Dict[str, set[str]] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    dim_val = str(fields.get(dim, "")).strip()
                    id_val = str(fields.get(ident, "")).strip()
                    if not dim_val or not id_val:
                        continue
                    buckets.setdefault(dim_val, set()).add(id_val)
                if not buckets:
                    continue
                counts = {k: len(v) for k, v in buckets.items()}
                ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                distinct_identifier_counts_by_dimension.append(
                    {
                        "dimension_field": dim,
                        "identifier_field": ident,
                        "counts": dict(ranked[:80]),
                    }
                )

        # Status breakdown and failure/delay rates by dimension.
        status_by_dimension: List[Dict[str, object]] = []
        if status_field:
            for dim in dimension_fields[:10]:
                buckets: Dict[str, Dict[str, int]] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    dim_val = str(fields.get(dim, "")).strip()
                    if not dim_val:
                        continue
                    status_val = str(fields.get(status_field, "")).strip().lower()
                    if not status_val:
                        status_val = self._detect_status(str(row.get("snippet", "")))
                    if not status_val:
                        continue
                    b = buckets.setdefault(dim_val, {"failed": 0, "delayed": 0, "completed": 0, "other": 0, "total": 0})
                    b["total"] += 1
                    if "fail" in status_val:
                        b["failed"] += 1
                    elif "delay" in status_val:
                        b["delayed"] += 1
                    elif any(x in status_val for x in ["complete", "success", "done", "pass", "ok"]):
                        b["completed"] += 1
                    else:
                        b["other"] += 1
                if buckets:
                    for _, b in buckets.items():
                        total = b["total"]
                        b["failed_pct"] = round((b["failed"] / total) * 100.0, 4) if total else 0.0
                        b["delayed_pct"] = round((b["delayed"] / total) * 100.0, 4) if total else 0.0
                    ranked = sorted(buckets.items(), key=lambda kv: kv[1]["total"], reverse=True)
                    status_by_dimension.append(
                        {
                            "dimension_field": dim,
                            "status_field": status_field,
                            "groups": {k: v for k, v in ranked[:80]},
                        }
                    )

        # Time-based aggregations when possible.
        time_aggregations: List[Dict[str, object]] = []
        datetime_field = datetime_fields[0] if datetime_fields else ""
        primary_measure = primary_numeric[0] if primary_numeric else ""
        if datetime_field and primary_numeric:
            measure = primary_measure
            for grain in ["hour", "day", "month"]:
                totals = self._group_sum(rows, datetime_field, measure, grain)
                if not totals:
                    continue
                vals = list(totals.values())
                ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
                slope = self._trend_slope(vals)
                time_aggregations.append(
                    {
                        "datetime_field": datetime_field,
                        "measure_field": measure,
                        "grain": grain,
                        "point_count": len(totals),
                        "variance": round(self._sample_variance(vals), 6),
                        "trend_slope": round(slope, 6),
                        "trend_direction": self._trend_direction(slope),
                        "top_points": ranked[:20],
                    }
                )

        # Date-derived analytics for weekday/weekend/peak-hour style questions.
        temporal_patterns: Dict[str, object] = {}
        if datetime_field and primary_measure:
            weekday_totals: Dict[str, float] = {}
            weekend_total = 0.0
            weekday_total = 0.0
            hour_totals: Dict[int, float] = {}
            peak_hour_by_weekday: Dict[str, Dict[int, float]] = {}
            for row in rows:
                fields = row.get("fields", {}) or {}
                dt = self._parse_datetime(str(fields.get(datetime_field, "")))
                if not dt:
                    continue
                val = self._to_float(fields.get(primary_measure))
                if val is None:
                    continue
                day_name = dt.strftime("%A")
                weekday_totals[day_name] = weekday_totals.get(day_name, 0.0) + val
                hour_totals[dt.hour] = hour_totals.get(dt.hour, 0.0) + val
                if dt.weekday() >= 5:
                    weekend_total += val
                else:
                    weekday_total += val
                key = day_name
                if key not in peak_hour_by_weekday:
                    peak_hour_by_weekday[key] = {}
                peak_hour_by_weekday[key][dt.hour] = peak_hour_by_weekday[key].get(dt.hour, 0.0) + val

            peak_hour = None
            if hour_totals:
                peak_hour = sorted(hour_totals.items(), key=lambda kv: kv[1], reverse=True)[0]
            peak_per_day: Dict[str, Dict[str, float]] = {}
            for day_name, by_hour in peak_hour_by_weekday.items():
                if not by_hour:
                    continue
                h, total = sorted(by_hour.items(), key=lambda kv: kv[1], reverse=True)[0]
                peak_per_day[day_name] = {"hour": h, "total": round(total, 6)}

            temporal_patterns = {
                "datetime_field": datetime_field,
                "measure_field": primary_measure,
                "weekday_totals": {k: round(v, 6) for k, v in sorted(weekday_totals.items())},
                "hour_totals": {str(k): round(v, 6) for k, v in sorted(hour_totals.items())},
                "peak_hour_overall": {
                    "hour": peak_hour[0],
                    "total": round(peak_hour[1], 6),
                } if peak_hour else {},
                "peak_hour_by_weekday": peak_per_day,
                "weekend_vs_weekday": {
                    "weekend_total": round(weekend_total, 6),
                    "weekday_total": round(weekday_total, 6),
                    "weekend_gt_weekday": bool(weekend_total > weekday_total),
                },
            }

        # Holiday comparison when a holiday-like field exists.
        holiday_comparison: Dict[str, object] = {}
        if primary_measure:
            holiday_field = ""
            for field in dimension_fields:
                tokens = self._field_name_tokens(field)
                if any(t in tokens for t in ["holiday", "is_holiday", "festival"]):
                    holiday_field = field
                    break
            if holiday_field:
                buckets = {"holiday": [], "non_holiday": []}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    raw = str(fields.get(holiday_field, "")).strip().lower()
                    val = self._to_float(fields.get(primary_measure))
                    if val is None:
                        continue
                    is_holiday = raw in {"1", "true", "yes", "y", "holiday"} or "holiday" in raw
                    buckets["holiday" if is_holiday else "non_holiday"].append(val)
                holiday_comparison = {
                    "holiday_field": holiday_field,
                    "measure_field": primary_measure,
                    "holiday_avg": round(sum(buckets["holiday"]) / len(buckets["holiday"]), 6) if buckets["holiday"] else 0.0,
                    "non_holiday_avg": round(sum(buckets["non_holiday"]) / len(buckets["non_holiday"]), 6) if buckets["non_holiday"] else 0.0,
                    "holiday_count": len(buckets["holiday"]),
                    "non_holiday_count": len(buckets["non_holiday"]),
                }

        # Top-N contribution share by dimension for percentage questions.
        top_n_share_by_dimension: List[Dict[str, object]] = []
        if primary_measure:
            for dim in dimension_fields[:8]:
                sums: Dict[str, float] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    d = str(fields.get(dim, "")).strip()
                    v = self._to_float(fields.get(primary_measure))
                    if not d or v is None:
                        continue
                    sums[d] = sums.get(d, 0.0) + v
                if not sums:
                    continue
                ranked = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
                total_sum = sum(sums.values())
                for n in [3, 5]:
                    top_sum = sum(v for _, v in ranked[:n])
                    top_n_share_by_dimension.append(
                        {
                            "dimension_field": dim,
                            "measure_field": primary_measure,
                            "top_n": n,
                            "share_pct": round((top_sum / total_sum) * 100.0, 6) if total_sum > 0 else 0.0,
                            "top_values": ranked[:n],
                        }
                    )

        # Multi-value categorical token counts (country/genre/listed-in style columns).
        multi_value_distributions: Dict[str, Dict[str, int]] = {}
        for dim in dimension_fields[:8]:
            sample_values = [
                str((row.get("fields", {}) or {}).get(dim, "")).strip()
                for row in rows[:400]
            ]
            non_empty = [v for v in sample_values if v]
            if not non_empty:
                continue
            has_multivalue = sum(1 for v in non_empty if ("," in v or "/" in v or "|" in v)) >= max(5, len(non_empty) // 5)
            if not has_multivalue:
                continue
            counts: Dict[str, int] = {}
            for row in rows:
                raw = str((row.get("fields", {}) or {}).get(dim, "")).strip()
                for token in self._split_multi_value(raw):
                    counts[token] = counts.get(token, 0) + 1
            if counts:
                ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                multi_value_distributions[dim] = dict(ranked[:80])

        # Duration analytics for movie/show-like datasets.
        duration_analytics: Dict[str, object] = {}
        duration_field = ""
        for field in list(schema.get("fields", {}).keys()):
            tokens = self._field_name_tokens(field)
            if any(t in tokens for t in ["duration", "runtime", "length"]):
                duration_field = field
                break
        if duration_field:
            durations: List[float] = []
            for row in rows:
                raw = str((row.get("fields", {}) or {}).get(duration_field, ""))
                mins = self._parse_duration_to_minutes(raw)
                if mins is not None:
                    durations.append(mins)
            if durations:
                duration_analytics["duration_field"] = duration_field
                duration_analytics["overall_minutes_stats"] = self._compute_basic_math(durations)

            # Optional split by a type-like dimension (e.g., Movie vs TV Show).
            type_field = ""
            for dim in dimension_fields:
                values = set()
                for row in rows[:300]:
                    v = str((row.get("fields", {}) or {}).get(dim, "")).strip().lower()
                    if v:
                        values.add(v)
                if any("movie" in v for v in values) and any("tv" in v or "show" in v for v in values):
                    type_field = dim
                    break
            if type_field:
                by_type: Dict[str, List[float]] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    t = str(fields.get(type_field, "")).strip()
                    mins = self._parse_duration_to_minutes(str(fields.get(duration_field, "")))
                    if not t or mins is None:
                        continue
                    by_type.setdefault(t, []).append(mins)
                duration_analytics["type_field"] = type_field
                duration_analytics["by_type_minutes_stats"] = {
                    k: self._compute_basic_math(vs) for k, vs in by_type.items() if vs
                }

        # Optional two-dimension contingency (for category relationships).
        contingency_tables: List[Dict[str, object]] = []
        dims_for_contingency = dimension_fields[:8]
        if status_field and status_field in dimension_fields:
            dims_for_contingency = [status_field] + [d for d in dims_for_contingency if d != status_field]
        for i in range(len(dims_for_contingency)):
            for j in range(i + 1, len(dims_for_contingency)):
                a = dims_for_contingency[i]
                b = dims_for_contingency[j]
                table: Dict[str, int] = {}
                for row in rows:
                    fields = row.get("fields", {}) or {}
                    av = str(fields.get(a, "")).strip()
                    bv = str(fields.get(b, "")).strip()
                    if not av or not bv:
                        continue
                    key = f"{av} | {bv}"
                    table[key] = table.get(key, 0) + 1
                if table:
                    ranked = sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:30]
                    contingency_tables.append(
                        {
                            "dimension_a": a,
                            "dimension_b": b,
                            "top_pairs": ranked,
                        }
                    )

        return {
            "row_count": len(rows),
            "field_quality": field_quality,
            "primary_measure_field": primary_measure,
            "primary_datetime_field": datetime_field,
            "numeric_summary": numeric_summary,
            "numeric_field_coverage": numeric_field_coverage,
            "high_coverage_numeric_fields": high_coverage_numeric_fields,
            "dataset_wide_composites": dataset_wide_composites,
            "categorical_distributions": categorical_distributions,
            "grouped_aggregations": grouped_aggregations,
            "dimension_value_counts": dimension_value_counts,
            "distinct_identifier_counts_by_dimension": distinct_identifier_counts_by_dimension,
            "status_by_dimension": status_by_dimension,
            "time_aggregations": time_aggregations,
            "temporal_patterns": temporal_patterns,
            "holiday_comparison": holiday_comparison,
            "top_n_share_by_dimension": top_n_share_by_dimension,
            "multi_value_distributions": multi_value_distributions,
            "duration_analytics": duration_analytics,
            "contingency_tables": contingency_tables,
        }

    def answer(
        self,
        question: str,
        prior_scope: Dict[str, object] | None = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, object]] | None:
        rows = self._collect_all_rows()
        if not rows:
            return None

        schema = self._build_schema(rows)
        scoped_rows, scope, scope_state = self._infer_scope(rows, question, schema, prior_scope=prior_scope)
        effective_rows = scoped_rows if scoped_rows else rows
        scope["used_global_fallback"] = bool(not scoped_rows and len(rows) > 0)

        analysis_payload = {
            "question": question,
            "scope": scope,
            "global_summary": self._summarize(rows, schema),
            "scoped_summary": self._summarize(effective_rows, schema),
            "universal_analytics": self._build_universal_analytics(effective_rows, schema),
            "dynamic_metrics": self._derive_dynamic_metrics(question, effective_rows, schema),
            "field_inference": {
                "fields": list(schema.get("fields", {}).keys()),
                "numeric_fields": schema.get("numeric_fields", []),
                "datetime_fields": schema.get("datetime_fields", []),
                "measure_fields": schema.get("measure_fields", []),
                "dimension_fields": schema.get("dimension_fields", []),
                "candidates": schema.get("candidates", {}),
            },
            "notes": [
                "Analytics are schema-driven and computed from indexed categorical rows.",
                "If a metric is missing, required fields were not confidently detected in indexed data.",
            ],
        }

        answer = "ANALYTICS_CONTEXT_JSON:\n" + json.dumps(analysis_payload, ensure_ascii=True, indent=2)
        sources = [
            {
                "source": str(row.get("source", "unknown")),
                "type": "pdf",
                "snippet": str(row.get("snippet", "")),
            }
            for row in effective_rows[:150]
        ]
        return answer, sources, scope_state

    def machine_row_counts(self) -> Dict[str, int]:
        rows = self._collect_all_rows()
        if not rows:
            return {}
        schema = self._build_schema(rows)
        candidates = schema.get("candidates", {}) or {}
        dimension_fields = schema.get("dimension_fields", []) or []

        entity_field = ""
        for field in dimension_fields:
            tokens = self._field_name_tokens(field)
            if any(t in tokens for t in ["machine", "base", "vehicle", "equipment", "device"]):
                entity_field = field
                break
        if not entity_field and dimension_fields:
            entity_field = dimension_fields[0]

        if not entity_field:
            return {}

        counts: Dict[str, int] = {}
        for row in rows:
            value = str((row.get("fields", {}) or {}).get(entity_field, "")).strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))

    def schema_summary(self) -> Dict[str, object]:
        rows = self._collect_all_rows()
        schema = self._build_schema(rows) if rows else {}
        return {
            "categorical_row_count": len(rows),
            "field_count": len(schema.get("fields", {})),
            "field_names": sorted(list((schema.get("fields", {}) or {}).keys()))[:100],
            "numeric_fields": schema.get("numeric_fields", []),
            "datetime_fields": schema.get("datetime_fields", []),
            "measure_fields": schema.get("measure_fields", []),
            "identifier_fields": schema.get("identifier_fields", []),
            "dimension_fields": schema.get("dimension_fields", []),
            "candidates": schema.get("candidates", {}),
        }
