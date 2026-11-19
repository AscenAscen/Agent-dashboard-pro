import os
import re
import sys
import json
import html
import hashlib
from urllib.parse import quote
from datetime import datetime, timezone, timedelta
import feedparser
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

TARGET_REGIONS = {
    "안양 평촌": {
        "queries": ["평촌 1기신도시 선도지구", "안양 동안구 재건축", "인덕원 동탄선 GTX-C 안양"],
        "key_complexes": ["관악성원", "꿈마을", "귀인마을", "목련마을", "평촌더샵센트럴시티"]
    },
    "울산 중구": {
        "queries": ["울산 중구 재개발", "울산 트램 1호선 중구", "우정혁신도시 부동산"],
        "key_complexes": ["우정아이파크", "에일린의뜰", "번영로서한이다음", "태화강유보라팰라티움"]
    }
}

OUTPUT_PATH = "data/news_briefing.json"
KST = timezone(timedelta(hours=9))


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_google_news_rss(query: str, limit: int = 6) -> list[dict]:
    encoded_q = quote(query)
    url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){encoded_q}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)

    cleaned = []
    for entry in feed.entries[:limit]:
        cleaned.append({
            "title": clean_html(entry.get("title", "")),
            "description": clean_html(entry.get("summary", "")),
            "link": entry.get("link", ""),
            "pubDate": entry.get("published", ""),
        })
    return cleaned


def extract_json_block(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                cleaned = part
                break
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                return None
    return None


def analyze_with_gemini(region: str, key_complexes: list[str], raw_articles: list[dict]):
    if not raw_articles:
        return [], ""

    if not GEMINI_API_KEY:
        return [], "GEMINI_API_KEY가 비어 있습니다. GitHub Secrets에 GEMINI_API_KEY를 등록해 주세요."

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
당신은 대한민국 부동산 입지 및 단지 가치 분석 전문가입니다.
아래는 [{region}] 지역과 관련해 최근 수집된 뉴스 목록입니다.
주요 관심 단지 목록: {', '.join(key_complexes)}

각 기사를 분석해 다음 JSON 형식으로만 답하세요:
{{
  "articles": [
    {{
      "title": "정제된 기사 제목",
      "link": "원문 기사 URL",
      "pub_date": "YYYY-MM-DD",
      "is_relevant": true,
      "category": "정비사업 또는 교통호재 또는 공급·분양 또는 규제·정책 또는 학군·환경",
      "sentiment": "호재 또는 중립 또는 악재",
      "summary_3lines": ["요약 1줄", "요약 2줄", "요약 3줄"],
      "impacted_complexes": ["관련 단지명"],
      "impact_analysis": "시세·전세·수요에 미치는 영향 (2문장 이내)",
      "imjang_checkpoint": "전임 또는 현장 임장 시 확인 질문 1가지"
    }}
  ]
}}

규칙:
- 단순 광고/분양홍보/타지역/중복 기사는 is_relevant를 false로 두세요.
- 마크다운 코드블록 없이 순수 JSON만 출력하세요.

[수집된 기사 목록]
{json.dumps(raw_articles, ensure_ascii=False, indent=2)}
"""

    errors = []
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            parsed = extract_json_block(response.text if response else "")
            if parsed and isinstance(parsed.get("articles"), list):
                valid = [art for art in parsed["articles"] if isinstance(art, dict) and art.get("is_relevant")]
                return valid, ""
            errors.append(f"{model_name}: JSON 파싱 실패")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)[:180]}")

    return [], " / ".join(errors)


def make_article_id(link: str, title: str) -> str:
    base = title.strip() or link.strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def load_existing_data(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("items"), list):
                    return data
        except Exception as e:
            print(f"기존 JSON 읽기 경고: {e}", file=sys.stderr)
    return {"last_updated": "", "total_count": 0, "status": "", "items": []}


def main():
    existing = load_existing_data(OUTPUT_PATH)
    existing_ids = {item["id"] for item in existing.get("items", []) if "id" in item}
    now_kst = datetime.now(KST)
    today_str = now_kst.strftime("%Y-%m-%d")

    newly_added = []
    status_messages = []

    for region, cfg in TARGET_REGIONS.items():
        seen_titles = set()
        candidates = []

        for q in cfg["queries"]:
            try:
                arts = fetch_google_news_rss(q, limit=6)
                print(f"[{region}] '{q}' 검색 결과 {len(arts)}건")
            except Exception as e:
                print(f"[{region}] '{q}' RSS 오류: {e}", file=sys.stderr)
                continue

            for art in arts:
                art_id = make_article_id(art["link"], art["title"])
                if art_id in existing_ids or art["title"] in seen_titles:
                    continue
                seen_titles.add(art["title"])
                candidates.append(art)

        candidates = candidates[:8]
        if not candidates:
            continue

        analyzed, err_msg = analyze_with_gemini(region, cfg["key_complexes"], candidates)
        if err_msg:
            print(f"[{region}] Gemini 분석 안내: {err_msg}", file=sys.stderr)
            status_messages.append(f"{region}: {err_msg}")

        for item in analyzed:
            art_id = make_article_id(item.get("link", ""), item.get("title", ""))
            if art_id in existing_ids:
                continue
            existing_ids.add(art_id)
            item["id"] = art_id
            item["region"] = region
            item["collected_at"] = today_str
            newly_added.append(item)

    merged_items = (newly_added + existing.get("items", []))[:300]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_payload = {
        "last_updated": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "total_count": len(merged_items),
        "status": " | ".join(status_messages) if status_messages else "OK",
        "items": merged_items,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"완료: 신규 기사 {len(newly_added)}건 추가 (총 {len(merged_items)}건), 상태: {result_payload['status']}")


if __name__ == "__main__":
    main()
