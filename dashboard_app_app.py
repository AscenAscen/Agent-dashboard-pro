import os
import json
import re
from datetime import datetime, timezone, timedelta
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="임장 에이전트 대시보드",
    page_icon="🏢",
    layout="wide",
)

KST = timezone(timedelta(hours=9))
NEWS_PATH = "data/news_briefing.json"
IMJANG_LOG_PATH = "data/imjang_notes.json"


def get_gemini_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def load_json(path: str, default_data: dict) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return default_data


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def analyze_phone_note(api_key: str, region: str, complex_name: str, agency_name: str, raw_note: str) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"""
당신은 월급쟁이부자들 실전 투자자 관점의 부동산 전화임장(전임) 및 현장임장(매임) 정리 비서입니다.
아래 통화/임장 메모를 구조화해 순수 JSON으로만 응답하세요.

- 지역: {region}
- 단지명: {complex_name}
- 부동산명: {agency_name}
- 메모 원문:
{raw_note}

출력 JSON 형식:
{{
  "listings": [
    {{
      "type": "매매 또는 전세",
      "area": "평형 (예: 84㎡ / 33평)",
      "floor_dir": "동/층/향",
      "price": "호가 (예: 8억 5,000)",
      "jeonse_or_gap": "전세가 또는 갭 금액 (파악된 경우)",
      "repair_status": "수리/상태",
      "tenant_status": "세입자/입주가능일",
      "owner_situation": "매도자 사유/네고 여지"
    }}
  ],
  "market_mood": "단지 분위기, 거래량, 급매 소진 여부 요약 (2~3문장)",
  "key_insights": ["핵심 인사이트 1", "핵심 인사이트 2"],
  "next_questions": ["다음 통화/방문 시 사장님께 물어볼 후속 질문 1", "후속 질문 2"]
}}
"""
    last_err = ""
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            parsed = extract_json_block(res.text if res else "")
            if parsed and isinstance(parsed, dict):
                return parsed
            last_err = f"{model_name}: JSON 파싱 실패"
        except Exception as e:
            last_err = f"{model_name}: {str(e)[:150]}"
    raise RuntimeError(last_err)


def generate_imjang_report(api_key: str, region: str, news_items: list[dict], phone_notes: list[dict]) -> str:
    client = genai.Client(api_key=api_key)
    prompt = f"""
당신은 부동산 임장보고서 작성 코치입니다.
아래 [{region}]의 최신 뉴스 브리핑과 전임/매임 기록을 종합해 임장보고서 핵심 파트를 작성하세요.

[최신 뉴스 브리핑]
{json.dumps(news_items[:15], ensure_ascii=False, indent=2)}

[전임/매임 기록]
{json.dumps(phone_notes[:15], ensure_ascii=False, indent=2)}

다음 목차로 깔끔한 마크다운 보고서를 작성하세요:
1. 지역 핵심 이슈 및 호재·악재 요약
2. 현장 분위기 및 시세·급매 동향 (전임 기반)
3. 관심 단지별 체크포인트 및 우선순위
4. 다음 현장 임장(매임) 실행 액션 플랜
"""
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            if res and res.text:
                return res.text
        except Exception:
            continue
    return "보고서 생성 중 오류가 발생했습니다. API 키와 네트워크 상태를 확인해 주세요."


st.title("🏢 AI 임장 에이전트 대시보드")
st.caption("자동 수집 뉴스 브리핑 · 전임/매임 통화 AI 정리 · 임장보고서 초안")

api_key = get_gemini_key()
news_data = load_json(NEWS_PATH, {"last_updated": "", "total_count": 0, "status": "", "items": []})
notes_data = load_json(IMJANG_LOG_PATH, {"items": []})

with st.sidebar:
    st.header("⚙️ 상태 및 필터")
    st.write(f"**뉴스 최근 업데이트:**  
{news_data.get('last_updated') or '아직 없음'}")
    st.write(f"**누적 뉴스:** {len(news_data.get('items', []))}건")
    st.write(f"**전임/매임 기록:** {len(notes_data.get('items', []))}건")
    if api_key:
        st.success("Gemini API 키 연결됨")
    else:
        st.warning("Streamlit Secrets에 GEMINI_API_KEY를 등록하면 전임 AI 정리와 보고서 생성이 활성화됩니다.")

tab1, tab2, tab3 = st.tabs(["📰 지역 뉴스 브리핑", "📞 전임·매임 통화 정리", "📑 임장보고서 초안"])

with tab1:
    items = news_data.get("items", [])
    if not items:
        st.info("수집된 뉴스가 없습니다. GitHub Actions에서 'Daily Real Estate News Briefing'을 실행해 주세요.")
        if news_data.get("status"):
            st.caption(f"상태 메시지: {news_data.get('status')}")
    else:
        regions = ["전체"] + sorted(list({x.get("region", "기타") for x in items}))
        sentiments = ["전체", "호재", "중립", "악재"]
        c1, c2, c3 = st.columns([1, 1, 2])
        sel_region = c1.selectbox("지역 선택", regions)
        sel_sentiment = c2.selectbox("영향 구분", sentiments)
        keyword = c3.text_input("단지/키워드 검색", placeholder="예: 선도지구, 트램, 관악성원...")

        filtered = items
        if sel_region != "전체":
            filtered = [x for x in filtered if x.get("region") == sel_region]
        if sel_sentiment != "전체":
            filtered = [x for x in filtered if x.get("sentiment") == sel_sentiment]
        if keyword.strip():
            kw = keyword.strip()
            filtered = [
                x for x in filtered
                if kw in x.get("title", "")
                or kw in x.get("impact_analysis", "")
                or any(kw in c for c in x.get("impacted_complexes", []))
            ]

        st.subheader(f"필터링된 브리핑 ({len(filtered)}건)")
        for art in filtered:
            badge = "🟢 호재" if art.get("sentiment") == "호재" else ("🔴 악재" if art.get("sentiment") == "악재" else "⚪ 중립")
            with st.expander(f"[{art.get('region', '')}] {badge} | {art.get('title', '')}", expanded=True):
                st.caption(f"분류: {art.get('category', '-')} · 수집일: {art.get('collected_at', '-')} · 영향 단지: {', '.join(art.get('impacted_complexes', [])) or '-'}")
                for line in art.get("summary_3lines", []):
                    st.markdown(f"- {line}")
                st.info(f"**입지·시세 영향:** {art.get('impact_analysis', '')}")
                st.warning(f"**전임/임장 확인 질문:** {art.get('imjang_checkpoint', '')}")
                if art.get("link"):
                    st.markdown(f"[원문 기사 보기]({art['link']})")

with tab2:
    st.subheader("📞 전화임장 / 현장임장 메모 AI 구조화")
    with st.form("phone_note_form"):
        col1, col2, col3 = st.columns(3)
        n_region = col1.text_input("지역", value="안양 평촌")
        n_complex = col2.text_input("단지명", placeholder="예: 관악성원")
        n_agency = col3.text_input("부동산명", placeholder="예: OO공인중개사")
        raw_note = st.text_area(
            "통화 메모 또는 녹취 텍스트 붙여넣기",
            height=160,
            placeholder="예: 33평 102동 중층 남향 8억5천 급매 하나 있음. 올수리 되어있고 주인거주라 입주협의 가능. 전세는 5억2천선..."
        )
        submitted = st.form_submit_button("AI로 매물·시세 정리하기", type="primary")

    if submitted:
        if not api_key:
            st.error("GEMINI_API_KEY가 설정되어 있지 않습니다.")
        elif not raw_note.strip():
            st.warning("통화 메모를 입력해 주세요.")
        else:
            with st.spinner("Gemini가 매물 정보와 후속 질문을 정리하고 있습니다..."):
                try:
                    structured = analyze_phone_note(api_key, n_region, n_complex, n_agency, raw_note)
                    entry = {
                        "created_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                        "region": n_region.strip(),
                        "complex": n_complex.strip(),
                        "agency": n_agency.strip(),
                        "raw_note": raw_note.strip(),
                        "structured": structured,
                    }
                    notes_data["items"].insert(0, entry)
                    save_json(IMJANG_LOG_PATH, notes_data)
                    st.success("정리 및 저장 완료!")
                except Exception as e:
                    st.error(f"분석 실패: {e}")

    st.divider()
    st.subheader("📋 누적 전임·매임 기록")
    if not notes_data.get("items"):
        st.caption("아직 저장된 전임 기록이 없습니다.")
    else:
        for idx, item in enumerate(notes_data["items"]):
            st.markdown(f"#### {item.get('created_at')} | [{item.get('region')}] {item.get('complex')} ({item.get('agency')})")
            st_data = item.get("structured", {})
            listings = st_data.get("listings", [])
            if listings:
                st.table(listings)
            if st_data.get("market_mood"):
                st.info(f"**단지 분위기:** {st_data['market_mood']}")
            if st_data.get("next_questions"):
                st.warning("**다음 확인 질문:** " + " / ".join(st_data["next_questions"]))
            with st.expander("원문 메모 보기"):
                st.text(item.get("raw_note", ""))

with tab3:
    st.subheader("📑 임장보고서 핵심 초안 자동 생성")
    all_regions = sorted(list(
        {x.get("region", "") for x in news_data.get("items", []) if x.get("region")}
        | {x.get("region", "") for x in notes_data.get("items", []) if x.get("region")}
    ))
    target_reg = st.selectbox("보고서 작성 대상 지역", all_regions if all_regions else ["안양 평촌", "울산 중구"])
    if st.button("임장보고서 초안 생성", type="primary"):
        if not api_key:
            st.error("GEMINI_API_KEY가 설정되어 있지 않습니다.")
        else:
            r_news = [x for x in news_data.get("items", []) if x.get("region") == target_reg]
            r_notes = [x for x in notes_data.get("items", []) if x.get("region") == target_reg]
            with st.spinner("수집된 뉴스와 전임 기록을 종합해 보고서 초안을 작성 중입니다..."):
                report_md = generate_imjang_report(api_key, target_reg, r_news, r_notes)
                st.markdown(report_md)
                st.download_button(
                    "마크다운(.md) 파일로 다운로드",
                    data=report_md.encode("utf-8"),
                    file_name=f"{target_reg}_임장보고서_초안.md",
                    mime="text/markdown",
                )
