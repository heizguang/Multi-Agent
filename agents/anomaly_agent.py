"""
异常检测子智能体

基于结构化查询结果做轻量级异常检测，优先覆盖数值型字段的离群点识别。
"""

from __future__ import annotations

import json
import math
from statistics import median
from typing import Any, Dict, List, Optional


class AnomalyDetectionAgent:
    """对列表型结构化数据执行本地异常检测。"""

    def _parse_data(self, data: Any) -> Optional[Any]:
        if isinstance(data, (list, dict)):
            return data
        if not isinstance(data, str):
            return None
        try:
            return json.loads(data)
        except Exception:
            return None

    def _percentile(self, values: List[float], ratio: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * ratio
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def _build_row_label(self, row: Dict[str, Any], index: int) -> str:
        for key in ("emp_name", "dept_name", "position", "id", "name"):
            value = row.get(key)
            if value not in (None, ""):
                return f"{key}={value}"
        return f"第{index + 1}行"

    def _collect_numeric_fields(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []
        fields: List[str] = []
        first = rows[0]
        for key, value in first.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields.append(key)
        return fields

    def detect(self, data: Any, context: str = "") -> Dict[str, Any]:
        parsed = self._parse_data(data)
        result = {
            "summary": "没有可检测的结构化数据。",
            "anomalies": [],
            "metrics": [],
            "error": None,
        }

        if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
            return result

        numeric_fields = self._collect_numeric_fields(parsed)
        if not numeric_fields:
            result["summary"] = "当前结果不包含可用于异常检测的数值字段。"
            return result

        anomalies: List[Dict[str, Any]] = []
        metrics: List[Dict[str, Any]] = []

        for field in numeric_fields:
            values = [
                float(row[field])
                for row in parsed
                if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
            ]
            if len(values) < 4:
                continue

            q1 = self._percentile(values, 0.25)
            q3 = self._percentile(values, 0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mid = median(values)

            field_anomalies: List[Dict[str, Any]] = []
            for index, row in enumerate(parsed):
                value = row.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                numeric_value = float(value)
                if numeric_value < lower or numeric_value > upper:
                    delta = numeric_value - mid
                    field_anomalies.append(
                        {
                            "field": field,
                            "row_index": index,
                            "label": self._build_row_label(row, index),
                            "value": numeric_value,
                            "median": round(mid, 2),
                            "delta": round(delta, 2),
                            "direction": "high" if numeric_value > upper else "low",
                            "row": row,
                        }
                    )

            metrics.append(
                {
                    "field": field,
                    "count": len(values),
                    "median": round(mid, 2),
                    "q1": round(q1, 2),
                    "q3": round(q3, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "anomaly_count": len(field_anomalies),
                }
            )
            anomalies.extend(field_anomalies)

        anomalies.sort(key=lambda item: abs(float(item.get("delta", 0))), reverse=True)
        result["anomalies"] = anomalies[:20]
        result["metrics"] = metrics

        if not anomalies:
            result["summary"] = "未发现明显的数值型异常点。"
            return result

        top_items = []
        for item in anomalies[:5]:
            direction = "偏高" if item["direction"] == "high" else "偏低"
            top_items.append(
                f"{item['label']} 的 {item['field']}={item['value']:.2f}，相对中位数{direction} {abs(item['delta']):.2f}"
            )

        context_prefix = f"在“{context}”场景下，" if context else ""
        result["summary"] = (
            f"{context_prefix}检测到 {len(anomalies)} 个潜在异常点，"
            f"重点包括：{'；'.join(top_items)}。"
        )
        return result
