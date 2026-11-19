import os
import re
import json
import html
import hashlib
from urllib.parse import quote
from datetime import datetime, timezone, timedelta
import feedparser
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

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


class AnalyzedArticle(BaseModel):
    title: str = Field(description="정제된 기사 제목")
    link: str = Field(description="원문 기사 URL")
    pub_date: str = Field(description="발행일 (YYYY-MM-DD)")
    is_relevant: bool = Field(description="실제 부동산 가치, 정비사업, 교통, 학군, 공급에 영향을 주는 유의미한 뉴스인지 여부")
    category: str = Field(description="정비사업 | 교통호재 | 공급·분양 | 규제·정책 | 학군·환경 중 택1")
    sentiment: str = Field(description="호재 | 중립 | 악재 중 택1")
    summary_3lines: list[str] = Field(description="핵심 내용 3줄 요약")
    impacted_complexes: list[str] = Field(description="직간접적 영향이 예상되는 단지명 또는 생활권")
    impact_analysis: str = Field(description="해당 지역 및 단지 시세·전세·수요에 미치는 구체적 영향 (2문장 이내)")
    imjang_checkpoint: str = Field(description="전임 또는 현장 임장 시 부동산 사장님께 확인해볼 질문 1가지")


class RegionBriefingResult(BaseModel):
    articles: list[AnalyzedArticle]


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_google_news_rss(query: str, limit: int = 8) -> list[dict]:
    encoded_q = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_q}&hl=ko&gl=KR&ceid=KR:ko"
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


def analyze_with_gemini(region: str, key_complexes: list[str], raw_articles: list[dict]) -> list[dict]:
    if not raw_articles:
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
당신은 대한민국 부동산 입지 및 단지 가치 분석 전문가입니다.
아래는 [{region}] 지역과 관련해 최근 수집된 뉴스 목록입니다.
주요 관심 단지 목록: {', '.join(key_complexes)}

각 기사를 분석해 다음 기준에 따라 구조화하세요:
1. 단순 분양 홍보성 기사, 타 지역 기사, 중복 내용은 `is_relevant: false`로 표시하세요.
2. 유효한 기사는 핵심 팩트 위주로 3줄 요약하고, 관심 단지 및 주변 생활권에 미치는 실질적 영향을 분석하세요.
3. 투자자가 다음 전화임장(전임)이나 현장 방문 시 바로 물어볼 수 있는 실전 질문 1가지를 작성하세요.

[수집된 기사 목록]
{json.dumps(raw_articles, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RegionBriefingResult,
            temperature=0.2,
        ),
    )

    parsed = json.loads(response.text)
    return [art for art in parsed.get("articles", []) if art.get("is_relevant")]


def make_article_id(link: str, title: str) -> str:
    base = title.strip() or link.strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def load_existing_data(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": "", "items": []}


def main():
    existing = load_existing_data(OUTPUT_PATH)
    existing_ids = {item["id"] for item in existing.get("items", [])}
    now_kst = datetime.now(KST)
    today_str = now_kst.strftime("%Y-%m-%d")

    newly_added = []

    for region, cfg in TARGET_REGIONS.items():
        seen_titles = set()
        candidates = []

        for q in cfg["queries"]:
            for art in fetch_google_news_rss(q, limit=8):
                art_id = make_article_id(art["link"], art["title"])
                if art_id in existing_ids or art["title"] in seen_titles:
                    continue
                seen_titles.add(art["title"])
                candidates.append(art)

        candidates = candidates[:15]
        if not candidates:
            continue

        analyzed = analyze_with_gemini(region, cfg["key_complexes"], candidates)
        for item in analyzed:
            art_id = make_article_id(item["link"], item["title"])
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
        "items": merged_items,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"완료: 신규 기사 {len(newly_added)}건 추가 (총 {len(merged_items)}건)")


if __name__ == "__main__":
    main()
