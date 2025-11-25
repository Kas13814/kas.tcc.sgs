# -*- coding: utf-8 -*-
"""
nxs_app.py — TCC AI • AirportOps Analytic
Backend using Google Generative AI (Gemini) + Supabase, with:
- Tool-style orchestration (no tool_code shown to end user)
- Chat history
- Arabic/English language detection and matching
- Full access to all provided tables/columns
- No mention of "Gemini" in any user-facing reply (only "TCC AI")
"""

import os
import json
import logging
import datetime as _dt
from typing import Any, Dict, List, Tuple, Optional

import httpx
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
#  إعدادات عامة + تسجيل
# =========================

# تحميل .env (لـ Supabase فقط أو أي متغيرات أخرى)
load_dotenv(override=True)

logging.basicConfig(
    level=logging.WARNING,  # كان INFO
    format="%(asctime)s [%(levelname)s] TCC-AI: %(message)s",
)


SUPABASE_URL = (
    os.getenv("SUPABASE_URL")
    or os.getenv("SUPABASE_REST_URL")
    or os.getenv("SUPABASE_PROJECT_URL")
    or os.getenv("SUPABASE_API_URL")
)

SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_KEY")
)

# ⚠️ تم إزالة مفتاح الـ API المكشوف. يجب وضعه الآن في متغير البيئة GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-flash"

logging.info("🔑 Gemini key length in app: %d", len(GEMINI_API_KEY) if GEMINI_API_KEY else 0)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("✅ تم تهيئة محرك الذكاء الاصطناعي بنجاح (الموديل: %s).", GEMINI_MODEL_NAME)
else:
    logging.warning("⚠️ لم يتم العثور على مفتاح TCC AI في الكود.")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.warning("⚠️ إعدادات Supabase ناقصة. يرجى التأكد من SUPABASE_URL و SUPABASE_SERVICE_ROLE_KEY.")

# =========================
#       FastAPI app
# =========================

app = FastAPI(
    title="TCC AI • AirportOps Analytic",
    description="TCC AI • AirportOps Analytic powered by LLM backend + Supabase (Tools + Chat History + Safe Answers).",
    version="2.6.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكن تضييقها لاحقاً
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


# =========================
#   ذاكرة المحادثة البسيطة
# =========================

CHAT_HISTORY: List[Dict[str, str]] = []
MAX_HISTORY_MESSAGES = 20


def add_to_history(role: str, content: str) -> None:
    CHAT_HISTORY.append({"role": role, "content": content})
    if len(CHAT_HISTORY) > MAX_HISTORY_MESSAGES:
        del CHAT_HISTORY[0 : len(CHAT_HISTORY) - MAX_HISTORY_MESSAGES]


def history_as_text() -> str:
    lines: List[str] = []
    for item in CHAT_HISTORY[-MAX_HISTORY_MESSAGES:]:
        prefix = "user: " if item["role"] == "user" else "ai: "
        lines.append(prefix + item["content"])
    return "\n".join(lines)


# =========================
#   دوال مساعدة عامة
# =========================

def detect_lang(text: str) -> str:
    """يعيد 'ar' إذا كان النص عربيًا في الغالب، وإلا 'en'."""
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ar"
    return "en"


def supabase_select(
    table: str,
    filters: Optional[Dict[str, str]] = None,
    limit: Optional[int] = None,
    order: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """استعلام عام على Supabase، يعيد قائمة صفوف (dict)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.error("❌ لا يمكن الاتصال بـ Supabase: بيانات الاتصال ناقصة.")
        return []

    url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    params: Dict[str, Any] = {"select": "*"}

    if limit is not None:
        params["limit"] = limit

    if filters:
        for col, expr in filters.items():
            params[col] = expr

    if order:
        col, direction = order
        params["order"] = f"{col}.{direction}"

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            logging.info("📡 Supabase: %s rows from %s", len(data), table)
            return data
    except Exception as e:
        logging.exception("❌ خطأ أثناء جلب البيانات من Supabase للجدول %s: %s", table, e)
        return []


# =========================
#     وصف الجداول (SCHEMA)
# =========================

SCHEMA_SUMMARY = """
الجداول المتاحة في قاعدة البيانات (Supabase):

1) employee_master_db
   - "Employee ID" (PK, Unique)
   - "Record Date"
   - "Employee Name", "Gender", "Nationality"
   - "Hiring Date"
   - "Job Title", "Actual Role", "Grade"
   - "Department", "Previous Department", "Current Department"
   - "Employment Action Type", "Action Effective Date", "Exit Reason", "Note"

2) sgs_flight_delay
   (تأخيرات المحطة / المطار / الخدمات الأرضية SGS/GS)
   - id (PK, identity), created_at
   - "Date", "Shift"
   - "Flight Category", "Airlines", "Flight Number", "Destination", "Gate"
   - "STD", "ATD"
   - "Delay Code", "Note"

3) dep_flight_delay
   (تأخيرات إدارة مراقبة الحركة DEP / TCC والأقسام التابعة مثل TCC, FIC Saudia, FIC Nas, LC Saudia, LC Foreign)
   - "Title" (PK)
   - "Date", "Shift", "Department"
   - "Duty Manager ID", "Duty Manager Name"
   - "Supervisor ID", "Supervisor Name"
   - "Control ID", "Control Name"
   - "Employee ID", "Employee Name"
   - "Airlines", "Flight Category", "Flight Direction"
   - "Gate"
   - "Arrival Flight Number", "Arrival Destination", "STA", "ATA", "Arrival Violations"
   - "Departure Flight Number", "Departure Destination", "STD", "ATD", "Departure Violations"
   - "Description of Incident", "Failure Impact"
   - "Investigation status", "InvestigationID"
   - "Consent to send investigation", "Current reminder"
   - "Respond to the investigation", "Administrative procedure", "Final action"
   - "Investigation status2"
   - "Manager Notes", "Last Update"
   - "Item Type", "Path"

4) employee_overtime
   (ساعات العمل الإضافي)
   - "Employee ID" (PK, Unique)
   - "Title"
   - "Shift", "Department"
   - "Duty Manager ID", "Duty Manager Name"
   - "Employee Name"
   - "Notification Date", "Notification Time"
   - "Assignment Date", "Assignment Type", "Assignment Days", "Total Hours"
   - "Assignment Reason", "Notes"
   - "Item Type", "Path"

5) employee_sick_leave
   (الإجازات المرضية)
   - "Title" (Unique)
   - "Date", "Shift", "Department"
   - "Sick leave start date", "Sick leave end date"
   - "Employee ID", "Employee Name"
   - "Item Type", "Path"

6) employee_absence
   (الغياب)
   - "Title" (PK, Unique)
   - "Date", "Shift", "Department"
   - "Employee ID", "Employee Name"
   - "Absence Notification Status"
   - "InvestigationID", "Consent to send investigation", "Current reminder"
   - "Respond to the investigation", "Administrative procedure", "Final action"
   - "Investigation status", "Manager Notes", "Last Update"
   - "Item Type", "Path"

7) employee_delay
   (تأخيرات الموظف الشخصية)
   - "Title" (PK, Unique)
   - "Date", "Shift", "Department"
   - "Employee ID", "Employee Name"
   - "Delay Minutes", "Reason for Delay", "Delay Notification Status"
   - "InvestigationID", "Consent to send investigation", "Current reminder"
   - "Respond to the investigation", "Administrative procedure", "Final action"
   - "Investigation status", "Manager Notes", "Last Update"
   - "Item Type", "Path"

8) operational_event
   (أحداث تشغيلية أخرى مرتبطة بالموظف)
   - "Title" (PK, Unique)
   - "Shift", "Department"
   - "Employee ID", "Employee Name"
   - "Event Date", "Event Type"
   - "InvestigationID", "Consent to send investigation", "Current reminder"
   - "Respond to the investigation", "Administrative procedure", "Final action"
   - "Investigation status", "Manager Notes", "Last Update"
   - "Disciplinary Action"
   - "Item Type", "Path"

9) shift_report
   (تقرير المناوبة)
   - "Title" (PK, Unique)
   - "Date", "Shift", "Department"
   - "Control 1 ID", "Control 1 Name", "Control 1 Start Time", "Control 1 End Time"
   - "Control 2 ID", "Control 2 Name", "Control 2 Start Time", "Control 2 End Time"
   - "Duty Manager Domestic ID", "Duty Manager Domestic Name"
   - "Duty Manager International+Foreign ID", "Duty Manager International+Foreign Name"
   - "Duty Manager All Halls ID", "Duty Manager All Halls Name"
   - "Supervisor Domestic ID", "Supervisor Domestic Name"
   - "Supervisor International+Foreign ID", "Supervisor International+Foreign Name"
   - "Supervisor All Halls ID", "Supervisor All Halls Name"
   - "On Duty", "No Show"
   - "Cars In Service", "Cars Out Of Service"
   - "Wireless Devices In Service", "Wireless Devices Out Of Service"
   - "Arrivals Domestic", "Delayed Arrivals Domestic"
   - "Arrivals International+Foreign", "Delayed Arrivals International+Foreign"
   - "Departures Domestic", "Delayed Departures Domestic"
   - "Departures International+Foreign", "Delayed Departures International+Foreign"
   - "Comments Domestic", "Comments International+Foreign", "Comments All Halls"
"""

# =========================
#   System Instructions
# =========================

SYSTEM_INSTRUCTION_TOOLS = """
أنت TCC AI • AirportOps Analytic.
تعمل كمساعد تحليلي ذكي وخبير في بيانات عمليات المطار.

مرحلة "تحليل النية":
- مهمتك الآن هي فهم نية المستخدم فقط وتحديد نوع الأداة التي نحتاجها، مع المعطيات الضرورية (رقم موظف، قسم، شركة طيران، فترة زمنية...).

قاعدة مهمة جداً على الأرقام:
- لا تقوم بأي تصحيح أو تخمين لأرقام الموظفين أو الرحلات.
- إذا كتب المستخدم 1503814 فهذا رقم مختلف عن 15013814، والتعامل يكون مع الرقم كما كتبه المستخدم حرفياً.
- إذا لم تكن متأكداً من الرقم أو لم يظهر بوضوح في السؤال، اجعل "employee_id" = null،
  ولا تخترع رقماً من التاريخ السابق للمحادثة.

أجب دائماً بصيغة JSON فقط بدون أي نص آخر، بالشكل التالي (مثال):

{
  "intent": "employee_profile",
  "employee_id": "15013814"
}

قائمة النوايا (intent) المدعومة:
- "employee_profile"
- "employee_absence_summary"
- "employee_delay_summary"
- "employee_overtime_summary"
- "employee_sickleave_summary"
- "flight_delay_summary"
- "dep_employee_delay_summary"
- "operational_event_summary"
- "shift_report_summary"
- "airline_flight_stats"
- "free_talk"

المفاتيح الممكنة داخل JSON:
- "intent"
- "employee_id"   (نص كما كتبه المستخدم بالضبط، بدون تعديل)
- "department"
- "flight_number"
- "airline"
- "start_date"
- "end_date"

قواعد صارمة:
1) لا تضف أي حقول غير المذكورة.
2) لا تكتب أي شيء خارج JSON.
3) إذا لم تستطع تحديد نية واضحة، استخدم: { "intent": "free_talk" } فقط.
"""

SYSTEM_INSTRUCTION_ANSWER = """
أنت TCC AI • AirportOps Analytic.

المدخلات التي تصلك الآن في الـ prompt:
- سؤال المستخدم (باللغة العربية أو الإنجليزية).
- intent_info: يوضح نوع النية (مثلاً employee_overtime_summary، flight_delay_summary، ...) مع المعطيات (employee_id, department, airline...).
- data_summary: نص عادي (ليس كوداً وليس JSON) يحتوي على ملخص دقيق للبيانات التي تم جلبها من قاعدة البيانات.
  هذا الملخص هو الحقيقة الوحيدة التي يجب أن تعتمد عليها في الأرقام والتفاصيل.

قواعد صارمة جداً:
- لا تُظهر للمستخدم أي Tool Call أو Tool Output أو JSON أو كود.
- لا تذكر أسماء الجداول أو Supabase أو REST أو المتغيرات الداخلية.
- لا تعدّل أو تصحح أرقام الموظفين أو الرحلات أو أرقام السجلات.
- إذا احتجت كتابة أي رقم (مثل رقم موظف أو رحلة) انسخه حرفياً كما جاء في data_summary أو في سؤال المستخدم، ولا تغيّر أي رقم.
- إذا قال data_summary إنه لا توجد بيانات، التزم بذلك.
- استخدم نفس لغة المستخدم (عربي أو إنجليزي) كما هو محدد في التعليمات (lang_code)،
  إلا إذا طلب المستخدم صراحة داخل سؤاله أن تكون الإجابة بلغة أخرى.
- تجنّب استخدام تنسيق Markdown الغليظ (**مثل هذا**). اكتب نصاً عادياً منسقاً بأسطر ونقاط بدون **.

وضع الإيجاز (Short Response Mode):
- إذا كان سؤال المستخدم واضح أنه يطلب معلومة واحدة محددة فقط، مثل:
  • "ما اسم الموظف 15013814؟"
  • "كم عدد ساعات العمل الاضافي للموظف 15013814؟"
  • "كم عدد أيام الغياب لقسم TCC؟"
  ففي هذه الحالة:
  • أجب بجملة أو جملتين فقط تحتوي على المعلومة المطلوبة مباشرة.
  • لا تعطي تقريراً طويلاً أو ملخصاً كاملاً.
- إذا كان سؤال المستخدم عاماً مثل "اعطني ملخص عن الموظف 15013814" أو "اعطني تقريراً كاملاً عن الغياب والتأخير"،
  يمكنك عندها إعطاء ملخص تفصيلي أطول يعتمد على data_summary.

بخصوص الجداول والأعمدة:
- يمكنك الاعتماد على جميع الأعمدة المتاحة كما تم وصفها في SCHEMA.
- عند الحديث عن ملخص موظف، يمكن أن تذكر:
  • بياناته الأساسية من employee_master_db.
  • عدد سجلات الغياب، التأخير، الإجازات المرضية، العمل الإضافي، الأحداث التشغيلية، وتأخيرات DEP المرتبطة به.
  • أي تفاصيل مهمة أخرى تظهر في data_summary.

هدفك:
- إعادة صياغة ما في data_summary بشكل واضح، منظم، ومهني.
- يمكنك ترتيب النقاط، إضافة عناوين فرعية، أو تبسيط اللغة، لكن دون اختراع أي أرقام أو معلومات غير موجودة في data_summary.
"""

# =========================
#   استدعاء المحرك النصي
# =========================

def _call_llm(prompt: str) -> str:
    """استدعاء عام لمحرك النص مع إخفاء الاسم عن المستخدم."""
    if not GEMINI_API_KEY or not GEMINI_MODEL_NAME:
        return "⚠️ محرك TCC AI غير مهيأ حالياً على الخادم. يرجى مراجعة إعدادات مفتاح الذكاء الاصطناعي."

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    try:
        resp = model.generate_content(prompt)
    except Exception as e:
        logging.exception("❌ خطأ أثناء الاتصال بالمحرك النصي: %s", e)
        msg = str(e)
        if "API key expired" in msg or "API_KEY_INVALID" in msg:
            return "⚠️ مفتاح خدمة TCC AI غير صالح أو منتهي الصلاحية. يرجى تجديده في إعدادات الخادم."
        if "An internal error has occurred" in msg or "InternalServerError" in msg:
            return "⚠️ هناك مشكلة تقنية مؤقتة في محرك TCC AI، يمكنك المحاولة لاحقاً."
        return "⚠️ حدث خطأ أثناء الاتصال بمحرك TCC AI."

    text = ""
    try:
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        elif hasattr(resp, "candidates") and resp.candidates:
            parts: List[str] = []
            for cand in resp.candidates:
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    for p in cand.content.parts:
                        if getattr(p, "text", None):
                            parts.append(p.text)
            text = "\n".join(parts)
    except Exception:
        text = str(resp)

    if not text:
        text = "⚠️ لم أستطع توليد رد مفهوم من محرك TCC AI."
    return text.strip()


# =========================
#   مرحلة 1: تحليل النية
# =========================

def classify_intent_with_llm(message: str, lang: str) -> Dict[str, Any]:
    """استدعاء المحرك لتحليل نية السؤال وإرجاع JSON فقط."""
    history_text = history_as_text()

    prompt = (
        SYSTEM_INSTRUCTION_TOOLS
        + "\n\n"
        + "وصف الجداول (SCHEMA):\n"
        + SCHEMA_SUMMARY
        + "\n\n"
        + f"لغة السؤال الحالية (lang_code) = {lang}\n"
        + "\n"
        + "سجل المحادثة السابق (مختصر):\n"
        + (history_text if history_text else "(لا يوجد تاريخ سابق)")
        + "\n\n"
        + "سؤال المستخدم الحالي:\n"
        + message
        + "\n\n"
        + "تذكير مهم: أجب بصيغة JSON صالح فقط بدون أي تعليق إضافي."
    )

    raw = _call_llm(prompt)

    if raw.startswith("⚠️"):
        logging.error("❌ فشل تحليل النية بسبب خطأ من المحرك: %s", raw)
        return {"intent": "free_talk"}

    txt = raw.strip()
    # إزالة حاويات ``` إن وُجدت
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logging.error("❌ لم أستطع استخراج JSON صحيح من رد التصنيف: %s", raw)
        return {"intent": "free_talk"}

    json_part = txt[start : end + 1]
    try:
        data = json.loads(json_part)
        if not isinstance(data, dict):
            return {"intent": "free_talk"}
        if "intent" not in data:
            data["intent"] = "free_talk"
        # مهم: عدم تعديل employee_id، فقط تحويله لنص
        if "employee_id" in data and data["employee_id"] is not None:
            data["employee_id"] = str(data["employee_id"])
        return data
    except Exception as e:
        logging.exception("❌ خطأ أثناء parsing JSON لرد التصنيف: %s", e)
        return {"intent": "free_talk"}


# =========================
#   مرحلة 2: الأدوات (Supabase)
# =========================

def tool_employee_profile(employee_id: str) -> Dict[str, Any]:
    rows = supabase_select(
        "employee_master_db",
        filters={"Employee ID": f"eq.{employee_id}"},
        limit=1,
    )
    return {
        "employee_id": employee_id,
        "rows": rows,
    }


def tool_employee_absence_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"

    and_parts: List[str] = []
    if start_date:
        and_parts.append(f"Date.gte.{start_date}")
    if end_date:
        and_parts.append(f"Date.lte.{end_date}")
    if and_parts:
        filters["and"] = "(" + ",".join(and_parts) + ")"

    rows = supabase_select(
        "employee_absence",
        filters=filters if filters else None,
        limit=1000,
        order=("Date", "asc"),
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "start_date": start_date,
        "end_date": end_date,
        "rows": rows,
    }


def tool_employee_delay_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"

    and_parts: List[str] = []
    if start_date:
        and_parts.append(f"Date.gte.{start_date}")
    if end_date:
        and_parts.append(f"Date.lte.{end_date}")
    if and_parts:
        filters["and"] = "(" + ",".join(and_parts) + ")"

    rows = supabase_select(
        "employee_delay",
        filters=filters if filters else None,
        limit=1000,
        order=("Date", "asc"),
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "start_date": start_date,
        "end_date": end_date,
        "rows": rows,
    }


def tool_employee_overtime_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"

    rows = supabase_select(
        "employee_overtime",
        filters=filters if filters else None,
        limit=1000,
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "rows": rows,
    }


def tool_employee_sick_leave_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"

    rows = supabase_select(
        "employee_sick_leave",
        filters=filters if filters else None,
        limit=1000,
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "rows": rows,
    }


def tool_flight_delay_summary(
    flight_number: Optional[str] = None,
    airline: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    filters_sgs: Dict[str, str] = {}
    if flight_number:
        filters_sgs["Flight Number"] = f"eq.{flight_number}"
    if airline:
        filters_sgs["Airlines"] = f"eq.{airline}"

    and_parts_sgs: List[str] = []
    if start_date:
        and_parts_sgs.append(f"Date.gte.{start_date}")
    if end_date:
        and_parts_sgs.append(f"Date.lte.{end_date}")
    if and_parts_sgs:
        filters_sgs["and"] = "(" + ",".join(and_parts_sgs) + ")"

    sgs_rows = supabase_select(
        "sgs_flight_delay",
        filters=filters_sgs if filters_sgs else None,
        limit=1000,
        order=("Date", "asc"),
    )

    filters_dep: Dict[str, str] = {}
    if flight_number:
        filters_dep["Departure Flight Number"] = f"eq.{flight_number}"
    if airline:
        filters_dep["Airlines"] = f"eq.{airline}"

    and_parts_dep: List[str] = []
    if start_date:
        and_parts_dep.append(f"Date.gte.{start_date}")
    if end_date:
        and_parts_dep.append(f"Date.lte.{end_date}")
    if and_parts_dep:
        filters_dep["and"] = "(" + ",".join(and_parts_dep) + ")"

    dep_rows = supabase_select(
        "dep_flight_delay",
        filters=filters_dep if filters_dep else None,
        limit=1000,
        order=("Date", "asc"),
    )

    return {
        "flight_number": flight_number,
        "airline": airline,
        "start_date": start_date,
        "end_date": end_date,
        "sgs_rows": sgs_rows,
        "dep_rows": dep_rows,
    }


def tool_dep_employee_delay_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    airline: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"
    if airline:
        filters["Airlines"] = f"eq.{airline}"

    rows = supabase_select(
        "dep_flight_delay",
        filters=filters if filters else None,
        limit=2000,
        order=("Date", "asc"),
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "airline": airline,
        "rows": rows,
    }


def tool_operational_event_summary(
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if employee_id:
        filters["Employee ID"] = f"eq.{employee_id}"
    if department:
        filters["Department"] = f"eq.{department}"

    and_parts: List[str] = []
    if start_date:
        and_parts.append(f"Event Date.gte.{start_date}")
    if end_date:
        and_parts.append(f"Event Date.lte.{end_date}")
    if and_parts:
        filters["and"] = "(" + ",".join(and_parts) + ")"

    rows = supabase_select(
        "operational_event",
        filters=filters if filters else None,
        limit=1000,
        order=("Event Date", "asc"),
    )
    return {
        "employee_id": employee_id,
        "department": department,
        "start_date": start_date,
        "end_date": end_date,
        "rows": rows,
    }


def tool_shift_report_summary(
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, str] = {}
    if department:
        filters["Department"] = f"eq.{department}"

    and_parts: List[str] = []
    if start_date:
        and_parts.append(f"Date.gte.{start_date}")
    if end_date:
        and_parts.append(f"Date.lte.{end_date}")
    if and_parts:
        filters["and"] = "(" + ",".join(and_parts) + ")"

    rows = supabase_select(
        "shift_report",
        filters=filters if filters else None,
        limit=1000,
    )
    return {
        "department": department,
        "start_date": start_date,
        "end_date": end_date,
        "rows": rows,
    }


def tool_airline_flight_stats() -> Dict[str, Any]:
    rows = supabase_select(
        "sgs_flight_delay",
        filters=None,
        limit=5000,
    )

    stats: Dict[str, int] = {}
    for r in rows:
        airline = r.get("Airlines")
        if airline is None:
            continue
        name = str(airline).strip()
        if not name:
            continue
        stats[name] = stats.get(name, 0) + 1

    return {"stats": stats}


# =========================
#   تلخيص للبيانات من الأدوات
# =========================

def _nxs_parse_delay_to_minutes(raw):
    """تحويل قيمة حقل Delay Minutes (مثل 00:20:00) إلى دقائق عددية."""
    if raw is None:
        return 0
    # قيم رقمية مباشرة
    try:
        if isinstance(raw, (int, float)):
            return int(raw)
        text = str(raw).strip()
        if not text:
            return 0
        # إذا كانت على شكل HH:MM:SS أو MM:SS
        if ":" in text:
            parts = text.split(":")
            parts = [p or "0" for p in parts]
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, m, s = "0", parts[0], parts[1]
            else:
                # شكل غير متوقع، نحاول اعتباره دقائق
                return int(float(text))
            h = int(h)
            m = int(m)
            s = int(s)
            total_minutes = h * 60 + m + (1 if s >= 30 else 0)
            return total_minutes
        # بدون نقطتين: نعتبرها دقائق
        return int(float(text))
    except Exception:
        return 0


def _nxs_parse_date_safe(date_str: Any) -> Optional[_dt.date]:
    if not date_str:
        return None
    try:
        # يفضل دائماً تخزين التواريخ كـ YYYY-MM-DD (ISO 8601) في قاعدة البيانات
        if isinstance(date_str, str) and date_str.startswith("20") and date_str[4] == "-":
            return _dt.date.fromisoformat(date_str[:10])
        # محاولة تحليل تاريخ بصيغة Power Automate/SharePoint القديمة (DD-MM-YYYY)
        if isinstance(date_str, str) and len(date_str) >= 10 and date_str[2] == "-" and date_str[5] == "-":
            d, m, y = map(int, date_str.split("-")[:3])
            return _dt.date(y, m, d)
    except Exception:
        pass
    return None


def _nxs_find_key(data: Dict[str, Any], part: str) -> Optional[str]:
    part_l = part.lower().strip()
    for k in data.keys():
        if part_l in k.lower():
            return k
    return None


def _nxs_in_range(d: _dt.date, d_from: Optional[_dt.date], d_to: Optional[_dt.date]) -> bool:
    if d_from and d < d_from:
        return False
    if d_to and d > d_to:
        return False
    return True


def _summary_employee_profile(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id") or "غير معروف"

    if not rows:
        if lang == "ar":
            return f"لا توجد أي بيانات موظف بالرقم الوظيفي {emp_id} في قاعدة البيانات."
        else:
            return f"There is no employee with ID {emp_id} in the database."

    row = rows[0]
    name = row.get("Employee Name") or "غير متوفر"
    nat = row.get("Nationality") or "غير متوفر"
    gender = row.get("Gender") or "غير متوفر"
    hiring = row.get("Hiring Date")
    role = row.get("Actual Role") or row.get("Job Title") or "غير متوفر"
    dept = row.get("Department") or row.get("Current Department") or "غير متوفر"
    prev_dept = row.get("Previous Department") or "غير متوفر"
    grade = row.get("Grade") or "غير متوفر"
    action_type = row.get("Employment Action Type") or "غير متوفر"
    action_date = row.get("Action Effective Date")
    exit_reason = row.get("Exit Reason") or "غير متوفر"

    hiring_str = str(hiring) if hiring else "غير مسجّل"
    action_date_str = str(action_date) if action_date else "غير مسجّل"

    if lang == "ar":
        return (
            f"ملف الموظف (Employee ID = {emp_id}):\n"
            f"- الاسم: {name}\n"
            f"- الجنسية: {nat}\n"
            f"- الجنس: {gender}\n"
            f"- تاريخ التوظيف: {hiring_str}\n"
            f"- الدرجة الوظيفية: {grade}\n"
            f"- الدور الفعلي / المسمى الوظيفي: {role}\n"
            f"- القسم الحالي: {dept}\n"
            f"- القسم السابق: {prev_dept}\n"
            f"- نوع آخر إجراء وظيفي: {action_type}\n"
            f"- تاريخ آخر إجراء وظيفي: {action_date_str}\n"
            f"- سبب الخروج / آخر إجراء وظيفي (إن وجد): {exit_reason}"
        )
    else:
        return (
            f"Employee profile (Employee ID = {emp_id}):\n"
            f"- Name: {name}\n"
            f"- Nationality: {nat}\n"
            f"- Gender: {gender}\n"
            f"- Hiring Date: {hiring_str}\n"
            f"- Grade: {grade}\n"
            f"- Actual Role / Job Title: {role}\n"
            f"- Current Department: {dept}\n"
            f"- Previous Department: {prev_dept}\n"
            f"- Last Employment Action Type: {action_type}\n"
            f"- Last Employment Action Date: {action_date_str}\n"
            f"- Exit Reason / Last Action Reason (if any): {exit_reason}"
        )


def _summary_employee_absence(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    total = len(rows)

    dates = [r.get("Date") for r in rows if r.get("Date")]
    start = min(dates) if dates else None
    end = max(dates) if dates else None

    if lang == "ar":
        if emp_id:
            if total == 0:
                return f"لا توجد سجلات غياب للموظف {emp_id}."
            return (
                f"ملخص سجلات الغياب للموظف {emp_id}:\n"
                f"- عدد السجلات: {total}\n"
                f"- أول غياب مسجل: {start or 'غير متوفر'}\n"
                f"- آخر غياب مسجل: {end or 'غير متوفر'}"
            )
        if dept:
            if total == 0:
                return f"لا توجد سجلات غياب لقسم {dept}."
            return (
                f"ملخص سجلات الغياب لقسم {dept}:\n"
                f"- عدد السجلات: {total}\n"
                f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
            )
        if total == 0:
            return "لا توجد سجلات غياب في النظام."
        return (
            f"إجمالي سجلات الغياب: {total}\n"
            f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        if emp_id:
            if total == 0:
                return f"No absence records for employee {emp_id}."
            return (
                f"Absence records for employee {emp_id}:\n"
                f"- Total records: {total}\n"
                f"- First recorded absence: {start or 'N/A'}\n"
                f"- Most recent absence: {end or 'N/A'}"
            )
        if dept:
            if total == 0:
                return f"No absence records for department {dept}."
            return (
                f"Absence records for department {dept}:\n"
                f"- Total records: {total}\n"
                f"- From {start or 'N/A'} to {end or 'N/A'}"
            )
        if total == 0:
            return "No absence records in the system."
        return (
            f"Total absence records: {total}\n"
            f"- From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_employee_delay(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    total = len(rows)

    dates = [r.get("Date") for r in rows if r.get("Date")]
    start = min(dates) if dates else None
    end = max(dates) if dates else None

    total_delay_minutes = 0
    for r in rows:
        delay_key = _nxs_find_key(r, "delay minutes") or _nxs_find_key(r, "delay")
        val = r.get(delay_key) if delay_key else None
        total_delay_minutes += _nxs_parse_delay_to_minutes(val)

    if lang == "ar":
        scope = f"الموظف {emp_id}" if emp_id else (f"قسم {dept}" if dept else "كل الموظفين")
        if total == 0:
            return f"لا توجد سجلات تأخير شخصي لـ {scope}."
        return (
            f"ملخص التأخير الشخصي لـ {scope}:\n"
            f"- عدد سجلات التأخير: {total}\n"
            f"- إجمالي دقائق التأخير: {total_delay_minutes} دقيقة\n"
            f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        scope = f"employee {emp_id}" if emp_id else (f"department {dept}" if dept else "all employees")
        if total == 0:
            return f"No personal delay records for {scope}."
        return (
            f"Personal delay summary for {scope}:\n"
            f"- Total delay records: {total}\n"
            f"- Total delay minutes: {total_delay_minutes} minutes\n"
            f"- From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_employee_overtime(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    total = len(rows)

    total_hours = 0.0
    latest_date: Optional[str] = None
    detailed_lines: List[str] = []

    for r in rows:
        hours_val: Optional[float] = None
        try:
            val = r.get("Total Hours")
            if val is not None:
                hours_val = float(val)
                total_hours += hours_val
        except Exception:
            pass

        adate = r.get("Assignment Date")
        if adate:
            if latest_date is None or adate > latest_date:
                latest_date = adate
        
        nd = r.get("Notification Date")
        atype = r.get("Assignment Type") or ""
        days = r.get("Assignment Days") or ""
        reason = r.get("Assignment Reason") or ""
        dept_row = r.get("Department") or ""
        dm_id = r.get("Duty Manager ID")
        dm_name = r.get("Duty Manager Name")

        if lang == "ar":
            line = f"- التاريخ: {nd or adate or 'غير متوفر'} | النوع: {atype or 'غير محدد'}"
            if days: line += f" | عدد الأيام: {days}"
            if hours_val is not None: line += f" | الساعات: {hours_val:.1f}"
            if reason: line += f" | السبب: {reason}"
            if dept_row and (not dept or dept_row != dept): line += f" | القسم: {dept_row}"
            if dm_id or dm_name: line += f" | المدير المناوب المعتمد: {dm_name or 'غير متوفر'} (ID: {dm_id or 'غير متوفر'})"
        else:
            line = f"- Date: {nd or adate or 'N/A'} | Type: {atype or 'Unspecified'}"
            if days: line += f" | Days: {days}"
            if hours_val is not None: line += f" | Hours: {hours_val:.1f}"
            if reason: line += f" | Reason: {reason}"
            if dept_row and (not dept or dept_row != dept): line += f" | Department: {dept_row}"
            if dm_id or dm_name: line += f" | Approved Duty Manager: {dm_name or 'N/A'} (ID: {dm_id or 'N/A'})"
        
        detailed_lines.append(line)

    if lang == "ar":
        scope = f"الموظف {emp_id}" if emp_id else (f"قسم {dept}" if dept else "كل الموظفين")
        if total == 0:
            return f"لا توجد سجلات عمل إضافي لـ {scope}."
        
        header = (
            f"ملخص العمل الإضافي لـ {scope}:\n"
            f"- عدد سجلات العمل الإضافي: {total}\n"
            f"- إجمالي الساعات الإضافية المسجلة: {total_hours:.1f} ساعة\n"
            f"- آخر تاريخ تكليف: {latest_date or 'غير متوفر'}\n"
            f"\n"
            f"تفاصيل السجلات:"
        )
    else:
        scope = f"employee {emp_id}" if emp_id else (f"department {dept}" if dept else "all employees")
        if total == 0:
            return f"No overtime records for {scope}."

        header = (
            f"Overtime summary for {scope}:\n"
            f"- Total overtime records: {total}\n"
            f"- Total recorded overtime hours: {total_hours:.1f} hours\n"
            f"- Most recent assignment date: {latest_date or 'N/A'}\n"
            f"\n"
            f"Record details:"
        )

    return header + "\n" + "\n".join(detailed_lines)


def _summary_employee_sick_leave(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    total_records = len(rows)

    dates = [r.get("Date") for r in rows if r.get("Date")]
    start = min(dates) if dates else None
    end = max(dates) if dates else None

    if lang == "ar":
        scope = f"الموظف {emp_id}" if emp_id else (f"قسم {dept}" if dept else "كل الموظفين")
        if total_records == 0:
            return f"لا توجد سجلات إجازة مرضية لـ {scope}."
        return (
            f"ملخص الإجازات المرضية لـ {scope}:\n"
            f"- عدد سجلات الإجازة المرضية: {total_records}\n"
            f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        scope = f"employee {emp_id}" if emp_id else (f"department {dept}" if dept else "all employees")
        if total_records == 0:
            return f"No sick leave records for {scope}."
        return (
            f"Sick leave summary for {scope}:\n"
            f"- Number of sick leave records: {total_records}\n"
            f"- From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_flight_delay(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    sgs_rows = data.get("sgs_rows") or []
    dep_rows = data.get("dep_rows") or []

    flight_number = data.get("flight_number") or info.get("flight_number")
    airline = data.get("airline") or info.get("airline")

    total_sgs = len(sgs_rows)
    total_dep = len(dep_rows)

    dates_sgs = [r.get("Date") for r in sgs_rows if r.get("Date")]
    dates_dep = [r.get("Date") for r in dep_rows if r.get("Date")]
    all_dates = dates_sgs + dates_dep
    
    start = min(all_dates) if all_dates else None
    end = max(all_dates) if all_dates else None
    
    # حساب إجمالي دقائق التأخير SGS
    total_sgs_delay_minutes = 0
    for r in sgs_rows:
        delay_code_str = str(r.get("Delay Code") or "").strip()
        if delay_code_str:
            try:
                # يرجى ملاحظة: هذا يفترض أن Delay Code يمثل الدقائق،
                # إذا كان Delay Code رمزًا فعليًا، يجب تغيير هذه المنطقية بناءً على كيفية تسجيل الدقائق
                total_sgs_delay_minutes += int(delay_code_str)
            except ValueError:
                pass # تجاهل الرموز غير العددية

    
    if lang == "ar":
        scope_flight = f" الرحلة رقم {flight_number}" if flight_number else ""
        scope_airline = f" لشركة {airline}" if airline else ""
        
        header = f"ملخص تأخيرات الطيران{scope_flight}{scope_airline}:\n"
        
        if total_sgs == 0 and total_dep == 0:
            return header + "لا توجد سجلات تأخير مطابقة في أي من جداول sgs_flight_delay أو dep_flight_delay."

        sgs_summary = (
            f"- سجلات تأخير المحطة/الخدمات الأرضية (sgs_flight_delay): {total_sgs} سجل\n"
            f"- إجمالي دقائق التأخير المحسوبة (من sgs_flight_delay): {total_sgs_delay_minutes} دقيقة"
        )
        dep_summary = (
            f"- سجلات تأخير مراقبة الحركة (dep_flight_delay): {total_dep} سجل"
        )
        
        return (
            header
            + sgs_summary + "\n"
            + dep_summary + "\n"
            + f"- الفترة الزمنية التي تشملها السجلات: من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        scope_flight = f" flight {flight_number}" if flight_number else ""
        scope_airline = f" for airline {airline}" if airline else ""
        
        header = f"Flight Delay Summary{scope_flight}{scope_airline}:\n"
        
        if total_sgs == 0 and total_dep == 0:
            return header + "No matching delay records found in either sgs_flight_delay or dep_flight_delay tables."

        sgs_summary = (
            f"- Station/Ground Services Delay Records (sgs_flight_delay): {total_sgs} records\n"
            f"- Total calculated delay minutes (from sgs_flight_delay): {total_sgs_delay_minutes} minutes"
        )
        dep_summary = (
            f"- Movement Control Delay Records (dep_flight_delay): {total_dep} records"
        )

        return (
            header
            + sgs_summary + "\n"
            + dep_summary + "\n"
            + f"- Timeframe covered by records: From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_dep_employee_delay(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    airline = data.get("airline") or info.get("airline")

    count_emp = len(rows)

    if emp_id:
        if lang == "ar":
            scope_air = f" لشركة {airline}" if airline else ""
            if count_emp == 0:
                return f"لا توجد أي رحلات متأخرة في مراقبة الحركة للموظف {emp_id}{scope_air}."
            return (
                f"ملخص تأخيرات مراقبة الحركة للموظف {emp_id}{scope_air}:\n"
                f"- عدد السجلات التي يظهر فيها هذا الموظف في dep_flight_delay كمسؤول/مرتبط بالتأخير: {count_emp}"
            )
        else:
            scope_air = f" for airline {airline}" if airline else ""
            if count_emp == 0:
                return f"No DEP delayed flights found for employee {emp_id}{scope_air}."
            return (
                f"DEP delay summary for employee {emp_id}{scope_air}:\n"
                f"- Number of flights where this employee appears in dep_flight_delay: {count_emp}"
            )

    if not rows:
        if lang == "ar":
            scope = f" في قسم {dept}" if dept else ""
            return f"لا توجد سجلات تأخير في مراقبة الحركة{scope}."
        else:
            scope = f" in department {dept}" if dept else ""
            return f"No DEP delay records{scope}."

    counts: Dict[str, int] = {}
    names: Dict[str, str] = {}
    for r in rows:
        eid = r.get("Employee ID")
        ename = r.get("Employee Name") or eid
        if eid:
            counts[eid] = counts.get(eid, 0) + 1
            names[eid] = str(ename).strip()

    output_lines: List[str] = []
    if lang == "ar":
        output_lines.append(f"ملخص تأخيرات مراقبة الحركة في قسم {dept} ({count_emp} سجل):")
        for eid, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            output_lines.append(f"- الموظف {names.get(eid, 'غير متوفر')} (ID: {eid}): {count} سجل")
    else:
        output_lines.append(f"DEP Delay Summary for Department {dept} ({count_emp} records):")
        for eid, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            output_lines.append(f"- Employee {names.get(eid, 'N/A')} (ID: {eid}): {count} records")

    return "\n".join(output_lines)


def _summary_operational_event(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    emp_id = data.get("employee_id") or info.get("employee_id")
    dept = data.get("department") or info.get("department")
    total = len(rows)

    dates = [r.get("Event Date") for r in rows if r.get("Event Date")]
    start = min(dates) if dates else None
    end = max(dates) if dates else None

    with_disc = [r for r in rows if (r.get("Disciplinary Action") or "").strip() != ""]
    cnt_disc = len(with_disc)

    if lang == "ar":
        scope = f"الموظف {emp_id}" if emp_id else (f"قسم {dept}" if dept else "كل البيانات")
        if total == 0:
            return f"لا توجد أحداث تشغيلية مسجلة لـ {scope}."
        return (
            f"ملخص الأحداث التشغيلية لـ {scope}:\n"
            f"- عدد الأحداث المسجلة: {total}\n"
            f"- عدد الأحداث التي ترتب عليها إجراء تأديبي: {cnt_disc}\n"
            f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        scope = f"employee {emp_id}" if emp_id else (f"department {dept}" if dept else "all data")
        if total == 0:
            return f"No operational events recorded for {scope}."
        return (
            f"Operational events summary for {scope}:\n"
            f"- Total events: {total}\n"
            f"- Events with disciplinary action: {cnt_disc}\n"
            f"- From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_shift_report(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    rows = data.get("rows") or []
    dept = data.get("department") or info.get("department")
    total = len(rows)

    on_duty = 0
    no_show = 0

    for r in rows:
        try:
            if r.get("On Duty") is not None:
                on_duty += int(r.get("On Duty"))
        except Exception:
            pass
        try:
            if r.get("No Show") is not None:
                no_show += int(r.get("No Show"))
        except Exception:
            pass

    dates = [r.get("Date") for r in rows if r.get("Date")]
    start = min(dates) if dates else None
    end = max(dates) if dates else None

    if lang == "ar":
        scope = f"لقسم {dept}" if dept else "الإجمالي"
        if total == 0:
            return f"لا توجد تقارير مناوبات مسجلة {scope}."
        return (
            f"ملخص تقارير المناوبات {scope} ({total} تقرير):\n"
            f"- إجمالي الأفراد المسجلين (On Duty) في هذه التقارير: {on_duty} فرد\n"
            f"- إجمالي حالات الغياب المسجلة (No Show) في هذه التقارير: {no_show} حالة\n"
            f"- الفترة من {start or 'غير متوفر'} إلى {end or 'غير متوفر'}"
        )
    else:
        scope = f"for department {dept}" if dept else "Overall"
        if total == 0:
            return f"No shift reports recorded {scope}."
        return (
            f"Shift Report Summary {scope} ({total} reports):\n"
            f"- Total individuals recorded (On Duty) in these reports: {on_duty} individuals\n"
            f"- Total absences recorded (No Show) in these reports: {no_show} cases\n"
            f"- From {start or 'N/A'} to {end or 'N/A'}"
        )


def _summary_airline_flight_stats(info: Dict[str, Any], data: Dict[str, Any], lang: str) -> str:
    stats = data.get("stats") or {}
    items = sorted(stats.items(), key=lambda kv: kv[1], reverse=True)

    if lang == "ar":
        lines = [
            "عدد السجلات لكل شركة طيران (مبني على جدول sgs_flight_delay فقط):",
            "",
            "| شركة الطيران | عدد السجلات في البيانات |",
            "|--------------|--------------------------|",
        ]
        for airline, cnt in items:
            lines.append(f"| {airline} | {cnt} |")
        lines.append("")
        lines.append("ملاحظة: هذه الأرقام مبنية على سجلات التأخير في جدول sgs_flight_delay، وليست كل رحلات المطار.")
        return "\n".join(lines)
    else:
        lines = [
            "Flight record count per airline (based on sgs_flight_delay only):",
            "",
            "| Airline | Number of records in data |",
            "|---------|---------------------------|",
        ]
        for airline, cnt in items:
            lines.append(f"| {airline} | {cnt} |")
        lines.append("")
        lines.append("Note: These counts are based on delay records in sgs_flight_delay, not all airport flights.")
        return "\n".join(lines)


def _summary_employee_profile_full(info: Dict[str, Any], tool_results: Dict[str, Any], lang: str) -> str:
    """ملخص شامل للموظف من جميع الجداول."""
    parts: List[str] = []

    # 1. Profile Core
    core = _summary_employee_profile(info, tool_results.get("employee_profile", {}), lang)
    parts.append(core)

    # 2. Absence
    abs_data = tool_results.get("employee_absence")
    if abs_data is not None:
        parts.append("")
        parts.append(_summary_employee_absence(info, abs_data, lang))

    # 3. Delay (Personal)
    delay_data = tool_results.get("employee_delay")
    if delay_data is not None:
        parts.append("")
        parts.append(_summary_employee_delay(info, delay_data, lang))

    # 4. Sick Leave
    sick_data = tool_results.get("employee_sick_leave")
    if sick_data is not None:
        parts.append("")
        parts.append(_summary_employee_sick_leave(info, sick_data, lang))

    # 5. Overtime
    overtime_data = tool_results.get("employee_overtime")
    if overtime_data is not None:
        parts.append("")
        parts.append(_summary_employee_overtime(info, overtime_data, lang))

    # 6. DEP Delay (Related to DEP flights)
    dep_delay_data = tool_results.get("dep_employee_delay")
    if dep_delay_data is not None:
        parts.append("")
        parts.append(_summary_dep_employee_delay(info, dep_delay_data, lang))
    
    # 7. Operational Events
    op_event_data = tool_results.get("operational_event")
    if op_event_data is not None:
        parts.append("")
        parts.append(_summary_operational_event(info, op_event_data, lang))

    return "\n".join(parts)


def build_data_summary(
    intent: str, intent_info: Dict[str, Any], tool_results: Dict[str, Any], lang: str
) -> str:
    """يبني نص الملخص النهائي اعتماداً على النية والنتائج."""

    if intent == "employee_profile":
        return _summary_employee_profile_full(intent_info, tool_results, lang)
    elif intent == "employee_absence_summary":
        return _summary_employee_absence(intent_info, tool_results.get("employee_absence", {}), lang)
    elif intent == "employee_delay_summary":
        return _summary_employee_delay(intent_info, tool_results.get("employee_delay", {}), lang)
    elif intent == "employee_overtime_summary":
        return _summary_employee_overtime(intent_info, tool_results.get("employee_overtime", {}), lang)
    elif intent == "employee_sickleave_summary":
        return _summary_employee_sick_leave(intent_info, tool_results.get("employee_sick_leave", {}), lang)
    elif intent == "flight_delay_summary":
        return _summary_flight_delay(intent_info, tool_results.get("flight_delay", {}), lang)
    elif intent == "dep_employee_delay_summary":
        return _summary_dep_employee_delay(intent_info, tool_results.get("dep_employee_delay", {}), lang)
    elif intent == "operational_event_summary":
        return _summary_operational_event(intent_info, tool_results.get("operational_event", {}), lang)
    elif intent == "shift_report_summary":
        return _summary_shift_report(intent_info, tool_results.get("shift_report", {}), lang)
    elif intent == "airline_flight_stats":
        return _summary_airline_flight_stats(intent_info, tool_results.get("airline_flight_stats", {}), lang)
    
    return "Data fetched from the database but the intent type is not recognized for summary."


# =========================
#   مرحلة 3: توليد الرد
# =========================

def generate_answer_with_llm(
    message: str,
    lang: str,
    intent: str,
    intent_info: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> str:
    data_summary = build_data_summary(intent, intent_info, tool_results, lang)
    history_text = history_as_text()
    lang_label = "العربية" if lang == "ar" else "English"

    prompt = (
        SYSTEM_INSTRUCTION_ANSWER
        + "\n\n"
        + f"lang_code المطلوب للإجابة = {lang} ({lang_label})\n"
        + "\n"
        + "سجل المحادثة السابق (مختصر):\n"
        + (history_text if history_text else "(لا يوجد تاريخ سابق)")
        + "\n\n"
        + "سؤال المستخدم الحالي:\n"
        + message
        + "\n\n"
        + "intent_info (لوصف نوع الطلب فقط، لا تعرضه للمستخدم):\n"
        + json.dumps(intent_info, ensure_ascii=False)
        + "\n\n"
        + "data_summary (هذا النص يمثل النتائج الفعلية من قاعدة البيانات، لا تعرض كلمة data_summary للمستخدم):\n"
        + data_summary
        + "\n\n"
        + "تذكير صارم: أجب للمستخدم فقط بناءً على ما في data_summary، "
        "وبنفس لغة lang_code المذكورة أعلاه، بدون أي JSON أو كود أو أسماء أدوات أو تنسيق غليظ **."
    )

    text = _call_llm(prompt)

    if text.startswith("⚠️"):
        # في حالة فشل المحرك نرجع الملخص كما هو
        return data_summary
        
    return text


def generate_free_talk_answer(message: str, lang: str) -> str:
    history_text = history_as_text()
    lang_label = "العربية" if lang == "ar" else "English"
    system = (
        "أنت TCC AI • AirportOps Analytic.\n"
        "يمكنك التحدّث بشكل عام، شرح المفاهيم، أو مساعدة المستخدم في الأسئلة غير المرتبطة مباشرة بالاستعلام عن البيانات.\n"
        "في وضع free_talk لا تقدّم أرقاماً دقيقة من النظام أو تحاول تحليل بيانات، ولكن يمكنك استخدام سياق المحادثة السابق.\n"
        "استخدم نفس لغة المستخدم (lang_code) للإجابة."
    )
    prompt = (
        system
        + "\n\n"
        + f"lang_code المطلوب للإجابة = {lang} ({lang_label})\n"
        + "\n"
        + "سجل المحادثة السابق (مختصر):\n"
        + (history_text if history_text else "(لا يوجد تاريخ سابق)")
        + "\n\n"
        + "سؤال المستخدم الحالي:\n"
        + message
    )

    text = _call_llm(prompt)
    if text.startswith("⚠️"):
        return "⚠️ حدث خطأ في التواصل مع محرك TCC AI. سأحاول استخدام أدوات البيانات مباشرة بدلاً من ذلك."

    return text


# =========================
# الدماغ الرئيسي TCC AI
# =========================

def nxs_brain(message: str) -> Tuple[str, Dict[str, Any]]:
    """
    1) يستدعي TCC AI لتحديد النية (بدون ذكر Gemini للمستخدم).
    2) يستدعي أداة البيانات المناسبة لكل intent.
    3) يبني data_summary.
    4) يعيد إجابة جاهزة للمستخدم، مع meta بسيط للواجهة.
    """
    msg_clean = (message or "").strip()
    lang = detect_lang(msg_clean)
    logging.info("📥 سؤال جديد إلى TCC AI: %s (lang=%s)", msg_clean, lang)
    add_to_history("user", msg_clean)

    # 1) تحليل النية
    intent_info = classify_intent_with_llm(msg_clean, lang)
    intent = intent_info.get("intent", "free_talk")
    logging.info("🎯 intent = %s | info = %s", intent, intent_info)

    tool_results: Dict[str, Any] = {}
    tools_used: List[str] = []

    # 2) استدعاء الأدوات حسب intent
    if intent == "employee_profile":
        emp_id = intent_info.get("employee_id")
        if emp_id:
            tool_results["employee_profile"] = tool_employee_profile(emp_id)
            tool_results["employee_overtime"] = tool_employee_overtime_summary(employee_id=emp_id)
            tool_results["employee_sick_leave"] = tool_employee_sick_leave_summary(employee_id=emp_id)
            tool_results["employee_absence"] = tool_employee_absence_summary(employee_id=emp_id)
            tool_results["employee_delay"] = tool_employee_delay_summary(employee_id=emp_id)
            tool_results["dep_employee_delay"] = tool_dep_employee_delay_summary(employee_id=emp_id)
            tool_results["operational_event"] = tool_operational_event_summary(employee_id=emp_id)
            tools_used.extend(
                [
                    "employee_profile",
                    "employee_overtime_summary",
                    "employee_sick_leave_summary",
                    "employee_absence_summary",
                    "employee_delay_summary",
                    "dep_employee_delay_summary",
                    "operational_event_summary",
                ]
            )

    elif intent == "employee_absence_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        s_date = intent_info.get("start_date")
        e_date = intent_info.get("end_date")
        if emp_id or dept:
            tool_results["employee_absence"] = tool_employee_absence_summary(
                employee_id=emp_id, department=dept, start_date=s_date, end_date=e_date
            )
            tools_used.append("employee_absence_summary")

    elif intent == "employee_delay_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        s_date = intent_info.get("start_date")
        e_date = intent_info.get("end_date")
        if emp_id or dept:
            tool_results["employee_delay"] = tool_employee_delay_summary(
                employee_id=emp_id, department=dept, start_date=s_date, end_date=e_date
            )
            tools_used.append("employee_delay_summary")

    elif intent == "employee_overtime_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        if emp_id or dept:
            tool_results["employee_overtime"] = tool_employee_overtime_summary(
                employee_id=emp_id, department=dept
            )
            tools_used.append("employee_overtime_summary")

    elif intent == "employee_sickleave_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        if emp_id or dept:
            tool_results["employee_sick_leave"] = tool_employee_sick_leave_summary(
                employee_id=emp_id, department=dept
            )
            tools_used.append("employee_sick_leave_summary")

    elif intent == "flight_delay_summary":
        f_num = intent_info.get("flight_number")
        airline = intent_info.get("airline")
        s_date = intent_info.get("start_date")
        e_date = intent_info.get("end_date")
        if f_num or airline:
            tool_results["flight_delay"] = tool_flight_delay_summary(
                flight_number=f_num, airline=airline, start_date=s_date, end_date=e_date
            )
            tools_used.append("flight_delay_summary")

    elif intent == "dep_employee_delay_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        airline = intent_info.get("airline")
        if emp_id or dept:
            tool_results["dep_employee_delay"] = tool_dep_employee_delay_summary(
                employee_id=emp_id, department=dept, airline=airline
            )
            tools_used.append("dep_employee_delay_summary")

    elif intent == "operational_event_summary":
        emp_id = intent_info.get("employee_id")
        dept = intent_info.get("department")
        s_date = intent_info.get("start_date")
        e_date = intent_info.get("end_date")
        if emp_id or dept:
            tool_results["operational_event"] = tool_operational_event_summary(
                employee_id=emp_id, department=dept, start_date=s_date, end_date=e_date
            )
            tools_used.append("operational_event_summary")

    elif intent == "shift_report_summary":
        dept = intent_info.get("department")
        s_date = intent_info.get("start_date")
        e_date = intent_info.get("end_date")
        if dept:
            tool_results["shift_report"] = tool_shift_report_summary(
                department=dept, start_date=s_date, end_date=e_date
            )
            tools_used.append("shift_report_summary")

    elif intent == "airline_flight_stats":
        tool_results["airline_flight_stats"] = tool_airline_flight_stats()
        tools_used.append("airline_flight_stats")

    # 3) توليد الرد النهائي
    if intent == "free_talk" or not tool_results:
        # إذا كانت النية محادثة عامة أو لم يتم استدعاء أي أداة بنجاح
        reply = generate_free_talk_answer(msg_clean, lang)
    else:
        # إذا تم استدعاء أداة بيانات بنجاح
        reply = generate_answer_with_llm(
            message=msg_clean,
            lang=lang,
            intent=intent,
            intent_info=intent_info,
            tool_results=tool_results,
        )

    add_to_history("assistant", reply)
    meta: Dict[str, Any] = {
        "lang": lang,
        "intent": intent_info,
        "tools_used": tools_used,
    }

    return reply, meta


# =========================
# المسارات (API)
# =========================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": "TCC AI • AirportOps Analytic",
        "version": "2.6.2",
        "description": "LLM backend + Supabase with tools-style orchestration, chat history, and safe answers (no tool code exposed).",
        "endpoints": ["/health", "/chat"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_key_present": bool(SUPABASE_KEY),
        "gemini_key_present": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL_NAME,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    msg = (req.message or "").strip()
    if not msg:
        return {
            "reply": "⚠️ لم يتم استلام نص للسؤال.",
            "answer": "⚠️ لم يتم استلام نص للسؤال.",
            "meta": {},
        }
    try:
        reply, meta = nxs_brain(msg)
        return {
            "reply": reply,
            "answer": reply,
            "meta": meta,
        }
    except Exception as e:
        logging.exception("❌ خطأ داخلي في /chat: %s", e)
        return {
            "reply": "❌ حدث خطأ داخلي أثناء معالجة السؤال.",
            "answer": "❌ حدث خطأ داخلي أثناء معالجة السؤال.",
            "meta": {},
        }
