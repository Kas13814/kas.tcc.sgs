# nxs_gemini_test.py
import google.generativeai as genai

# ⚠️ ضع هنا المفتاح الجديد من مشروع NXS-Dv-AI
api_key = "AIzaSyBtaHq6QQS5fmyGFqWUMzeM1qbcs4-1TFk"

print("Using API key length:", len(api_key))

if not api_key:
    raise SystemExit("❌ لا يوجد مفتاح API، تأكد أنك وضعته في الكود.")

# تهيئة Gemini
genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content("اختبار من nxs_gemini_test.py بالمفتاح المباشر")
    print("✅ نجح الاتصال. نص الرد:")
    print(resp.text)
except Exception as e:
    print("❌ فشل الاتصال بـ Gemini:")
    print(repr(e))
