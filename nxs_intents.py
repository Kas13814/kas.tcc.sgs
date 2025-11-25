# nxs_intents.py
# -----------------------------
# مسؤول عن فهم نية السؤال (Intent) واستخراج المعلومات المهمة
# مثل: رقم الموظف، رقم الرحلة، نوع الطلب، الفترة الزمنية...

from typing import Dict, Any
import re
from datetime import datetime

EMP_ID_PATTERN = re.compile(r"(?:الموظف|رقمه الوظيفي|employee)\s*(\d{6,8})")
FLIGHT_PATTERN = re.compile(r"(?:الرحلة|رحلة|flight)\s*([A-Z]{2}\d+|\d{3,5})")

DATE_RANGE_PATTERN = re.compile(
    r"(من|from)\s*(\d{4}-\d{2}-\d{2})\s*(?:إلى|الى|to)\s*(\d{4}-\d{2}-\d{2})"
)

def _extract_employee_id(text: str):
    m = EMP_ID_PATTERN.search(text)
    return m.group(1) if m else None

def _extract_flight(text: str):
    m = FLIGHT_PATTERN.search(text.upper())
    return m.group(1) if m else None

def _extract_date_range(text: str):
    m = DATE_RANGE_PATTERN.search(text)
    if not m:
        return None, None
    start, end = m.group(2), m.group(3)
    return start, end

def classify_intent(message: str) -> Dict[str, Any]:
    """
    يرجّع قاموس فيه:
    - intent  : نوع الطلب
    - employee_id / flight_number / start_date / end_date (لو موجودة)
    - raw_text: النص الأصلي
    """

    text = message.strip()
    text_lower = text.lower()

    employee_id = _extract_employee_id(text)
    flight_number = _extract_flight(text)
    start_date, end_date = _extract_date_range(text)

    intent: str = "general_chat"

    # أسئلة عن الموظف
    if "من هو الموظف" in text or "بطاقة الموظف" in text or "profile" in text_lower:
        intent = "employee_profile"
    elif "ساعات عمل إضافي" in text or "عمل اضافي" in text_lower or "overtime" in text_lower:
        intent = "employee_overtime"
    elif "تأخيرات الموظف" in text or ("تأخيرات" in text and employee_id):
        intent = "employee_delays"
    elif "غياب الموظف" in text or ("غياب" in text and employee_id):
        intent = "employee_absence"

    # أسئلة عن الرحلات / التأخيرات
    elif "تأخير الرحلة" in text or ("delay" in text_lower and flight_number):
        intent = "flight_delay_detail"
    elif "أكثر سبب للتأخير" in text or "أكثر شركة تأخير" in text:
        intent = "delay_statistics"

    # تحليلات عامة
    elif "تحليل" in text or "dashboard" in text_lower:
        intent = "analytics_request"

    return {
        "intent": intent,
        "employee_id": employee_id,
        "flight_number": flight_number,
        "start_date": start_date,
        "end_date": end_date,
        "raw_text": text,
    }

if __name__ == "__main__":
    # اختبار سريع
    tests = [
        "من هو الموظف الذي رقمه الوظيفي 15013814؟",
        "كم لديه ساعات عمل إضافي 15013814؟",
        "اعرض تأخيرات الموظف 15013814 من 2024-12-31 إلى 2025-01-31",
        "ما سبب تأخير الرحلة SV123 أمس؟",
        "أكثر سبب للتأخير خلال الشهر الماضي؟",
    ]
    for t in tests:
        print(t, "→", classify_intent(t))
