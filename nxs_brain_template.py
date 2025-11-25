# nxs_brain.py
from typing import Tuple, Dict, Any, List
import json
import textwrap
import traceback

from nxs_intents import classify_intent
from nxs_supabase_client import execute_dynamic_query
from test_gemini_key import call_model_text

DB_SCHEMA_DESCRIPTION = textwrap.dedent("""
(تم اختصار الوصف هنا، ضعه كما في النسخة الكاملة التي أرسلتها لك سابقًا)
""").strip()

def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

def _plan_with_gemini(message: str, intent_info: Dict[str, Any]) -> Dict[str, Any]:
    helper = json.dumps(intent_info, ensure_ascii=False)
    prompt = f"""أنت NXS • AirportOps AI ... (ضع النص الكامل هنا)"""
    raw = call_model_text(prompt)
    try:
        plan = json.loads(_extract_json_block(raw))
    except Exception:
        plan = {
            "mode": "chat_only",
            "sql": "",
            "reason": f"فشل تحليل JSON من النموذج. النص الخام: {raw[:200]}",
            "preferred_language": "ar",
        }
    return plan

def _answer_with_data(message: str, plan: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False)
    prompt = f"""أنت NXS • AirportOps AI ... (ضع النص الكامل هنا)"""
    answer = call_model_text(prompt)
    return answer

def nxs_brain(message: str) -> Tuple[str, Dict[str, Any]]:
    intent_info = classify_intent(message)
    try:
        plan = _plan_with_gemini(message, intent_info)
        mode = plan.get("mode", "chat_only")
        sql = (plan.get("sql") or "").strip()

        if mode == "chat_only" or not sql:
            prompt = f"""أنت NXS • AirportOps AI ... (وضع برسونالتي عامة)"""
            answer = call_model_text(prompt)
            meta: Dict[str, Any] = {
                "mode": "chat_only",
                "intent_info": intent_info,
                "plan": plan,
            }
            return answer, meta

        rows: List[Dict[str, Any]] = execute_dynamic_query(sql)
        row_count = len(rows)
        answer = _answer_with_data(message, plan, rows)

        meta = {
            "mode": "sql_and_answer",
            "intent_info": intent_info,
            "plan": plan,
            "sql": sql,
            "row_count": row_count,
            "sample_rows": rows[:10],
        }
        return answer, meta

    except Exception as exc:
        tb = traceback.format_exc()
        err_txt = (
            "⚠️ حدث خطأ داخلي داخل NXS أثناء استخدام الذكاء الاصطناعي.\n"
            f"نوع الخطأ: {type(exc).__name__}\n"
        )
        meta = {
            "error": str(exc),
            "traceback": tb,
            "intent_info": intent_info,
        }
        return err_txt, meta
