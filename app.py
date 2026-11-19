import os
import re
import json
import html
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
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash"]


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


def fetch_google_news(queries: list[str], limit_per_query: int = 5) -> list[dict]:
    seen_titles = set()
    articles = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit_per_query]:
            title = clean_html(entry.get("title", ""))
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            articles.append({
                "query": q,
                "title": title,
                "summary": clean_html(entry.get("summary", "")),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
    return articles[:12]


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


def summarize_news_with_gemini(api_key: str, region: str, complexes: str, raw_articles: list[dict]) -> list[dict]:
    prompt = f"""
당신은 대한민국 부동산 입지·단지 분석 전문가입니다.
아래 [{region}] 지역 뉴스 목록을 분석해 투자 가치가 있는 기사만 골라 순수 JSON으로만 응답하세요.
관심 단지: {complexes}

출력 형식:
{{
  "articles": [
    {{
      "title": "정제된 기사 제목",
      "link": "원문 URL",
      "category": "정비사업 | 교통호재 | 공급·분양 | 규제·정책 | 학군·환경",
      "sentiment": "호재 | 중립 | 악재",
      "summary_3lines": ["핵심 1줄", "핵심 2줄", "핵심 3줄"],
      "impact_analysis": "관심 단지 및 주변 시세·전세에 미치는 영향 (2문장 이내)",
      "imjang_checkpoint": "전임 또는 현장 임장 시 부동산 사장님께 물어볼 질문 1가지"
    }}
  ]
}}

규칙:
- 단순 분양 홍보, 광고성, 타 지역, 중복 기사는 제외하세요.
- 코드블록 없이 순수 JSON만 출력하세요.

[뉴스 목록]
{json.dumps(raw_articles, ensure_ascii=False, indent=2)}
"""
    raw_text = call_gemini_text(api_key, prompt, temperature=0.2)
    parsed = extract_json_block(raw_text)
    if parsed and isinstance(parsed.get("articles"), list):
        return parsed["articles"]
    return []


def analyze_phone_note_with_gemini(api_key: str, region: str, complex_name: str, agency: str, raw_note: str) -> dict:
    prompt = f"""
당신은 부동산 실전 투자자(월급쟁이부자들) 관점의 전임(전화임장)·매임(현장임장) 정리 비서입니다.
아래 통화/임장 메모를 구조화해 순수 JSON으로만 응답하세요.

- 지역: {region}
- 단지명: {complex_name}
- 부동산명: {agency}
- 메모 원문:
{raw_note}

출력 형식:
{{
  "listings": [
    {{
      "구분": "매매 또는 전세",
      "평형": "예: 33평 (84㎡)",
      "동·층·향": "예: 102동 중층 남향",
      "호가": "예: 8억 5,000",
      "전세/갭": "예: 전세 5억 2,000 / 갭 3억 3,000",
      "수리·상태": "예: 올수리",
      "입주·세입자": "예: 주인거주 입주협의",
      "매도사유·특이사항": "예: 이사 일정으로 급매"
    }}
  ],
  "market_mood": "단지 분위기, 거래량, 급매 소진 여부 요약 (2~3문장)",
  "next_questions": ["다음 통화/방문 시 사장님께 확인할 질문 1", "질문 2"]
}}

규칙:
- 코드블록 없이 순수 JSON만 출력하세요.
"""
    raw_text = call_gemini_text(api_key, prompt, temperature=0.2)
    parsed = extract_json_block(raw_text)
    if parsed and isinstance(parsed, dict):
        return parsed
    raise RuntimeError("AI 응답을 JSON으로 정리하지 못했습니다. 다시 한 번 눌러주세요.")


if "news_results" not in st.session_state:
    st.session_state["news_results"] = []
if "phone_logs" not in st.session_state:
    st.session_state["phone_logs"] = []

st.title("🏢 AI 임장 에이전트 대시보드")
st.caption("버튼 한 번으로 지역 뉴스 3줄 요약 · 전임/매임 통화 정리 · 임장보고서 초안 생성")

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

tab1, tab2, tab3 = st.tabs(["📰 뉴스 검색·요약", "📞 전임·매임 통화 정리", "📑 임장보고서 초안"])

with tab1:
    st.subheader("버튼 눌러 최신 지역 뉴스 수집 및 AI 요약")
    col1, col2 = st.columns([1, 2])
    region_input = col1.text_input("임장 지역", value="안양 평촌")
    complexes_input = col2.text_input("주요 관심 단지 (쉼표 구분)", value="관악성원, 꿈마을, 귀인마을, 목련마을")
    queries_input = st.text_input(
        "검색 키워드 (쉼표 구분)",
        value="평촌 1기신도시 선도지구, 안양 동안구 재건축, 인덕원 동탄선 GTX-C 안양",
    )

    if st.button("🔍 최신 뉴스 가져와서 요약하기", type="primary"):
        if not api_key:
            st.error("사이드바 또는 Streamlit Secrets에 GEMINI_API_KEY를 입력해 주세요.")
        else:
            queries = [q.strip() for q in queries_input.split(",") if q.strip()]
            with st.spinner("Google 뉴스를 검색하고 Gemini가 핵심만 요약 중입니다..."):
                try:
                    raw_news = fetch_google_news(queries, limit_per_query=5)
                    if not raw_news:
                        st.warning("검색된 뉴스가 없습니다. 검색 키워드를 조금 넓혀 보세요.")
                    else:
                        summarized = summarize_news_with_gemini(api_key, region_input, complexes_input, raw_news)
                        for item in summarized:
                            item["region"] = region_input.strip()
                        st.session_state["news_results"] = summarized
                        st.success(f"총 {len(summarized)}건의 핵심 부동산 뉴스를 정리했습니다!")
                except Exception as e:
                    st.error(f"뉴스 분석 중 오류: {e}")

    st.divider()
    for art in st.session_state["news_results"]:
        sentiment = art.get("sentiment", "중립")
        badge = "🟢 호재" if sentiment == "호재" else ("🔴 악재" if sentiment == "악재" else "⚪ 중립")
        with st.expander(f"{badge} [{art.get('category', '이슈')}] {art.get('title', '')}", expanded=True):
            for line in art.get("summary_3lines", []):
                st.markdown(f"- {line}")
            st.info(f"**입지·시세 영향:** {art.get('impact_analysis', '')}")
            st.warning(f"**전임·임장 확인 질문:** {art.get('imjang_checkpoint', '')}")
            if art.get("link"):
                st.markdown(f"[원문 기사 보기]({art['link']})")

with tab2:
    st.subheader("전화임장 / 현장임장 메모 AI 정리")
    with st.form("phone_form"):
        c1, c2, c3 = st.columns(3)
        p_region = c1.text_input("지역", value=region_input)
        p_complex = c2.text_input("단지명", placeholder="예: 관악성원")
        p_agency = c3.text_input("부동산명", placeholder="예: OO공인중개사")
        p_note = st.text_area(
            "통화 메모 또는 녹취 내용 붙여넣기",
            height=150,
            placeholder="예: 33평 102동 중층 남향 8억5천 급매 1개 있음. 올수리, 주인거주 입주협의 가능. 전세는 5억2천선...",
        )
        run_phone = st.form_submit_button("✨ AI로 매물표·후속질문 정리하기", type="primary")

    if run_phone:
        if not api_key:
            st.error("GEMINI_API_KEY를 입력해 주세요.")
        elif not p_note.strip():
            st.warning("통화 메모를 입력해 주세요.")
        else:
            with st.spinner("통화 내용을 매물표와 후속 질문으로 정리 중입니다..."):
                try:
                    result = analyze_phone_note_with_gemini(api_key, p_region, p_complex, p_agency, p_note)
                    st.session_state["phone_logs"].insert(0, {
                        "time": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                        "region": p_region.strip(),
                        "complex": p_complex.strip(),
                        "agency": p_agency.strip(),
                        "raw": p_note.strip(),
                        "result": result,
                    })
                    st.success("정리 완료!")
                except Exception as e:
                    st.error(f"정리 실패: {e}")

    st.divider()
    for log in st.session_state["phone_logs"]:
        st.markdown(f"#### {log['time']} | [{log['region']}] {log['complex']} ({log['agency']})")
        res = log["result"]
        if res.get("listings"):
            st.table(res["listings"])
        if res.get("market_mood"):
            st.info(f"**단지 분위기:** {res['market_mood']}")
        if res.get("next_questions"):
            st.warning("**다음 확인 질문:** " + " / ".join(res["next_questions"]))
        with st.expander("원문 메모 보기"):
            st.text(log["raw"])

with tab3:
    st.subheader("임장보고서 핵심 초안 생성")
    if st.button("📑 현재 뉴스 + 전임 내용으로 보고서 초안 만들기", type="primary"):
        if not api_key:
            st.error("GEMINI_API_KEY를 입력해 주세요.")
        elif not st.session_state["news_results"] and not st.session_state["phone_logs"]:
            st.warning("먼저 1번 탭에서 뉴스를 요약하거나 2번 탭에서 전임 메모를 정리해 주세요.")
        else:
            with st.spinner("임장보고서 초안을 작성 중입니다..."):
                prompt = f"""
당신은 부동산 임장보고서 작성 코치입니다.
아래 [{region_input}]의 요약된 뉴스와 전임/매임 기록을 종합해 깔끔한 마크다운 임장보고서 초안을 작성하세요.

[요약 뉴스]
{json.dumps(st.session_state['news_results'], ensure_ascii=False, indent=2)}

[전임/매임 기록]
{json.dumps(st.session_state['phone_logs'], ensure_ascii=False, indent=2)}

목차:
1. 지역 핵심 이슈 및 호재·악재 요약
2. 단지별 매물·시세·현장 분위기 (전임 기반)
3. 관심 단지 체크포인트 및 우선순위
4. 다음 현장 임장(매임) 실행 계획
"""
                try:
                    report = call_gemini_text(api_key, prompt, temperature=0.3)
                    st.markdown(report)
                    st.download_button(
                        "마크다운(.md) 파일로 저장",
                        data=report.encode("utf-8"),
                        file_name=f"{region_input}_임장보고서_초안.md",
                        mime="text/markdown",
                    )
                except Exception as e:
                    st.error(f"보고서 생성 오류: {e}")
