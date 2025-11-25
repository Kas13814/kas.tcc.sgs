# nxs_app_turbo.py — NXS • AirportOps AI (Stable Turbo Edition)
# -------------------------------------------------------------
# هذه النسخة مصممة لتكون:
# - سريعة ⚡
# - مستقرة 🛡️ (لا ترجع 500 أبداً للمستخدم، بل رسالة مفهومة)
# - متوافقة مع nxs_brain (أي نسخة عندك الآن)
# - بدون أي ذكر لـ Gemini في الـ API
#
# ملاحظة:
# - إذا حدث أي خطأ داخل nxs_brain، سيتم التقاطه وإرجاع رسالة نصية للمستخدم مع meta توضح الخطأ.
# - لا نستخدم HTTPException 500 حتى لا تظهر لك رسالة "Error: empty reply from server" في الواجهة.

import time
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nxs_brain import nxs_brain

app = FastAPI(
    title="NXS • AirportOps AI",
    version="2.0-stable-turbo",
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------- نماذج الطلب / الرد -------------
class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "ar"  # احتياطي للمستقبل إذا أحببنا تمرير اللغة من الواجهة


class ChatResponse(BaseModel):
    reply: str
    meta: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None


# ------------- كاش بسيط لتسريع الأسئلة المكررة -------------
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 20  # ثانية واحدة لعمر الكاش (قصير حتى نبقى أقرب للبيانات الحية)


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = CACHE.get(key)
    if not item:
        return None
    if time.time() - item["time"] > CACHE_TTL:
        return None
    return item["value"]


def cache_set(key: str, value: Dict[str, Any]) -> None:
    CACHE[key] = {"value": value, "time": time.time()}


# ------------- نقطة /chat الرئيسية -------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    نقطة المحادثة الرئيسية:
    - تقرأ message من المستخدم.
    - تستدعي nxs_brain(message).
    - ترجع الرد + meta + زمن التنفيذ.
    - في حالة أي خطأ، ترجع رسالة نصية للمستخدم، وليس 500.
    """
    start = time.time()

    # 1) فحص الكاش (إذا نفس السؤال تكرر خلال الفترة القصيرة)
    cached = cache_get(req.message)
    if cached is not None:
        return ChatResponse(
            reply=cached["reply"],
            meta=cached.get("meta"),
            latency_ms=0.5,  # لأن الرد من الكاش شبه فوري
        )

    # 2) استدعاء nxs_brain مع حماية كاملة من الأخطاء
    try:
        reply, meta = nxs_brain(req.message)
        # حفظ في الكاش
        cache_set(req.message, {"reply": reply, "meta": meta})
        latency = round((time.time() - start) * 1000.0, 2)
        return ChatResponse(reply=reply, meta=meta, latency_ms=latency)

    except Exception as e:
        # ⚠️ لا نرمي 500 أبداً للمستخدم، بل نرجع رسالة مفهومة
        # مع تسجيل الخطأ في meta فقط.
        fallback_reply = (
            "⚠️ حدث خطأ داخلي داخل NXS • AirportOps AI.\n"
            "يمكن مراجعة سجل الخادم (logs) لمعرفة التفاصيل.\n"
        )
        latency = round((time.time() - start) * 1000.0, 2)
        return ChatResponse(
            reply=fallback_reply,
            meta={
                "ok": False,
                "error": str(e),
                "source": "nxs_app_turbo_chat_handler",
            },
            latency_ms=latency,
        )


# ------------- نقطة فحص الصحة / -------------
@app.get("/")
async def home() -> Dict[str, Any]:
    return {
        "status": "running",
        "engine": "NXS • AirportOps AI",
        "mode": "Stable Turbo",
    }
