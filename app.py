import os
import re
import json
import html
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from datetime import datetime, timezone, timedelta
import feedparser
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="임장 에이전트 대시보드",
    page_icon="🏢",
    layout="wide",
)

KST = timezone(timedelta(hours=9))
MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

PERIOD_OPTIONS = {
    "최근 1일": 1,
    "최근 7일": 7,
    "최근 30일": 30,
    "최근 90일": 90,
    "전체 기간": None,
}


def get_gemini_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_google_news(query: str, days: int | None = 30, limit: int = 10) -> list[dict]:
    q = query.strip()
    if not q:
        return []

    search_q = f"{q} when:{days}d" if days else q
    url = f"https://news.google.com/rss/search?q={quote(search_q)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=days) if days else None

    seen_titles = set()
    articles = []
    for entry in feed.entries:
        title = clean_html(entry.get("title", ""))
        if not title or title in seen_titles:
            continue

        pub_str = entry.get("published", "")
        pub_display = ""
        if pub_str:
            try:
                pub_dt = parsedate_to_datetime(pub_str)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if cutoff and pub_dt < cutoff:
                    continue
                pub_display = pub_dt.astimezone(KST).strftime("%Y-%m-%d")
            except Exception:
                pub_display = pub_str[:16]

        seen_titles.add(title)
        articles.append({
            "title": title,
            "published_at": pub_display,
            "summary": clean_html(entry.get("summary", ""))[:180],
            "link": entry.get("link", ""),
        })
        if len(articles) >= limit:
            break

    return articles


def extract_json_block(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                cleaned = part
                break
    try:
        return json.loads(cleaned)
    except Exception:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e + 1])
            except Exception:
                return None
    return None


def call_gemini_text(api_key: str, prompt: str, temperature: float = 0.2) -> str:
    client = genai.Client(api_key=api_key)
    last_err = ""
    for model_name in MODEL_CANDIDATES:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            if res and res.text:
                return res.text
        except Exception as e:
            last_err = f"{model_name}: {str(e)[:220]}"
    raise RuntimeError(last_err or "Gemini 호출 실패")


def summarize_news_with_gemini(api_key: str, query: str, raw_articles: list[dict]) -> list[dict]:
    prompt = f"""
대한민국 부동산 입지·시세 분석가로서 '{query}' 검색 결과 중 투자·임장 가치가 있는 기사만 골라 순수 JSON으로만 답하세요.

출력 형식:
{{
  "articles": [
    {{
      "title": "기사 제목",
      "published_at": "발행일 (YYYY-MM-DD)",
      "link": "원문 URL",
      "category": "정비사업 | 교통호재 | 공급·분양 | 규제·정책 | 학군·환경",
      "sentiment": "호재 | 중립 | 악재",
      "summary_3lines": ["핵심 1줄", "핵심 2줄", "핵심 3줄"],
      "impact_analysis": "해당 지역·단지 시세에 미치는 영향 (1~2문장)",
      "imjang_checkpoint": "전임·임장 시 부동산 사장님께 확인할 질문 1가지"
    }}
  ]
}}

규칙: 단순 광고·분양홍보 기사 제외. 코드블록 없이 순수 JSON만 출력.

[뉴스 목록]
{json.dumps(raw_articles, ensure_ascii=False)}
"""
    raw_text = call_gemini_text(api_key, prompt, temperature=0.2)
    parsed = extract_json_block(raw_text)
    if parsed and isinstance(parsed.get("articles"), list):
        return parsed["articles"]
    return []


def analyze_phone_note_with_gemini(api_key: str, region: str, complex_name: str, agency: str, raw_note: str) -> dict:
    prompt = f"""
부동산 실전 투자자(월급쟁이부자들) 관점의 전임·매임 정리 비서입니다.
통화 메모를 구조화해 코드블록 없이 순수 JSON으로만 응답하세요.

- 지역: {region} / 단지: {complex_name} / 부동산: {agency}
- 메모: {raw_note}

출력 형식:
{{
  "listings": [
    {{
      "구분": "매매/전세",
      "평형": "평형",
      "동·층·향": "동/층/향",
      "호가": "호가",
      "전세/갭": "전세 또는 갭",
      "수리·상태": "수리상태",
      "입주·세입자": "입주/세입자",
      "매도사유·특이사항": "특이사항"
    }}
  ],
  "market_mood": "단지 분위기 및 급매 동향 요약 (2문장)",
  "next_questions": ["후속 질문 1", "후속 질문 2"]
}}
"""
    raw_text = call_gemini_text(api_key, prompt, temperature=0.2)
    parsed = extract_json_block(raw_text)
    if parsed and isinstance(parsed, dict):
        return parsed
    raise RuntimeError("AI 응답 정리에 실패했습니다. 다시 눌러주세요.")


if "news_results" not in st.session_state:
    st.session_state["news_results"] = []
if "phone_logs" not in st.session_state:
    st.session_state["phone_logs"] = []

st.title("🏢 AI 임장 에이전트 대시보드")
st.caption("지역·단지 뉴스 3줄 요약 · 전임/매임 통화 정리")

api_key = get_gemini_key()

with st.sidebar:
    st.header("🔑 API 연결 상태")
    if api_key:
        st.success("Gemini API 키 연결됨")
    else:
        st.warning("Secrets에 키가 없어 아래에 임시 입력할 수 있습니다.")
        api_key = st.text_input("Gemini API Key 입력", type="password").strip()

    st.divider()
    st.write(f"📰 조회된 뉴스: **{len(st.session_state['news_results'])}건**")
    st.write(f"📞 정리된 전임 메모: **{len(st.session_state['phone_logs'])}건**")

tab1, tab2 = st.tabs(["📰 뉴스 검색·요약", "📞 전임·매임 통화 정리"])

with tab1:
    st.subheader("뉴스 검색 및 AI 핵심 요약")
    col1, col2 = st.columns([3, 1])
    search_query = col1.text_input(
        "검색어",
        value="평촌 선도지구 재건축",
        placeholder="예: 평촌 관악성원, 울산 중구 트램, 인덕원 동탄선...",
    )
    selected_period = col2.selectbox(
        "검색 기간",
        options=list(PERIOD_OPTIONS.keys()),
        index=2,
    )

    if st.button("🔍 뉴스 검색해서 요약하기", type="primary"):
        if not api_key:
            st.error("Gemini API 키를 설정해 주세요.")
        elif not search_query.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            days = PERIOD_OPTIONS[selected_period]
            with st.spinner(f"'{search_query}' ({selected_period}) 뉴스를 검색하고 요약 중입니다..."):
                try:
                    raw_news = fetch_google_news(search_query, days=days, limit=10)
                    if not raw_news:
                        st.warning(f"{selected_period} 내에 검색된 뉴스가 없습니다. 검색 기간을 넓히거나 검색어를 바꿔 보세요.")
                    else:
                        summarized = summarize_news_with_gemini(api_key, search_query, raw_news)
                        st.session_state["news_results"] = summarized
                        st.success(f"총 {len(summarized)}건의 핵심 부동산 뉴스를 정리했습니다!")
                except Exception as e:
                    st.error(f"뉴스 분석 중 오류: {e}")

    st.divider()
    for art in st.session_state["news_results"]:
        sentiment = art.get("sentiment", "중립")
        badge = "🟢 호재" if sentiment == "호재" else ("🔴 악재" if sentiment == "악재" else "⚪ 중립")
        pub_date = f" ({art['published_at']})" if art.get("published_at") else ""
        with st.expander(f"{badge} [{art.get('category', '이슈')}] {art.get('title', '')}{pub_date}", expanded=True):
            for line in art.get("summary_3lines", []):
                st.markdown(f"- {line}")
            st.info(f"**입지·시세 영향:** {art.get('impact_analysis', '')}")
            st.warning(f"**전임·임장 확인 질문:** {art.get('imjang_checkpoint', '')}")
