# nxs_supabase_test.py

from nxs_supabase_client import (
    get_employee_by_id,
    list_all_flight_delays,
    get_employee_delays,
)
from nxs_analytics import summarize_employee_delays
from nxs_brain import answer_question_from_data


def test_employee():
    # غيّر هذا الرقم لرقم موظف موجود فعلياً في جدول Employee Master Db (لاحقاً)
    employee_id = "15013814"

    emp = get_employee_by_id(employee_id)
    if emp:
        print("✅ تم جلب بيانات الموظف:")
        for k, v in emp.items():
            print(f"  {k}: {v}")
    else:
        print(f"⚠️ لم يتم العثور على موظف بالرقم: {employee_id}")


def test_delays():
    delays = list_all_flight_delays(limit=5)
    print(f"✅ تم جلب {len(delays)} تأخيرات (أول 5):")

    if not delays:
        print("⚠️ لا توجد سجلات تأخير في الجدول dep_flight_delay.")
        return

    for row in delays:
        date = row.get("Date")
        shift = row.get("Shift")
        emp_id = row.get("Employee ID")
        emp_name = row.get("Employee Name")
        airline = row.get("Airlines")
        arr_flt = row.get("Arrival Flight Number")
        dep_flt = row.get("Departure Flight Number")
        arr_viol = row.get("Arrival Violations")
        dep_viol = row.get("Departure Violations")

        print(
            f"- {date} | {shift} | {airline} | "
            f"ARR {arr_flt} / DEP {dep_flt} | "
            f"الموظف {emp_id} - {emp_name} | "
            f"مخالفة وصول: {arr_viol} | مخالفة مغادرة: {dep_viol}"
        )


def test_employee_delays_raw():
    """
    اختبار: جلب تأخيرات موظف معيّن في فترة محددة (بيانات خام)
    """
    employee_id = "15013814"
    start_date = "2024-12-31"
    end_date = "2025-01-31"

    rows = get_employee_delays(employee_id, start_date, end_date)
    print(
        f"✅ تم جلب {len(rows)} تأخيرات للموظف {employee_id} "
        f"بين {start_date} و {end_date}"
    )

    for row in rows:
        print(
            f"- {row.get('Date')} | {row.get('Shift')} | {row.get('Airlines')} | "
            f"ARR {row.get('Arrival Flight Number')} / DEP {row.get('Departure Flight Number')} | "
            f"Arrival Violations: {row.get('Arrival Violations')} | "
            f"Departure Violations: {row.get('Departure Violations')}"
        )


def test_employee_delays_summary():
    """
    اختبار: ملخّص تحليلي لتأخيرات موظف معيّن
    """
    employee_id = "15013814"
    start_date = "2024-12-31"
    end_date = "2025-01-31"

    print("=== ملخّص تأخيرات موظف (تحليلي) ===")
    summary = summarize_employee_delays(employee_id, start_date, end_date, max_rows=5)
    print(summary)


def test_question_like_nxs():
    """
    اختبار: سؤال عربي واحد كما لو أنه قادم من NXS
    """
    message = "اعرض تأخيرات الموظف 15013814 من 2024-12-31 إلى 2025-01-31"
    print("=== سؤال بالعربي → جواب من البيانات ===")
    print(f"🗨️ السؤال: {message}\n")
    answer = answer_question_from_data(message)
    print("🤖 الإجابة:\n")
    print(answer)


if __name__ == "__main__":
    print("=== اختبار الموظف ===")
    test_employee()

    print("\n=== اختبار التأخيرات (أول 5) ===")
    test_delays()

    print("\n=== اختبار تأخيرات موظف خلال فترة (RAW) ===")
    test_employee_delays_raw()

    print("\n=== اختبار تأخيرات موظف خلال فترة (SUMMARY) ===")
    test_employee_delays_summary()

    print("\n=== اختبار سؤال عربي يشبه NXS ===")
    test_question_like_nxs()
