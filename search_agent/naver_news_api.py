import os
import re
from datetime import datetime
import requests
from dotenv import load_dotenv
from html import unescape

# .env 파일에서 환경 변수 로드
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

def clean_html(raw_html):
    """
    HTML 태그(<b> 등)를 제거하는 함수
    """
    if not raw_html:
        return ""
    # 정규표현식을 사용하여 HTML 태그 제거
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner, '', raw_html)
    # HTML 특수문자(&quot;, &amp; 등) 처리
    return unescape(cleantext).strip()

def parse_date(date_str):
    """
    API 응답의 날짜 문자열을 YYYY-MM-DD HH:mm:ss 포맷으로 변환.
    네이버 뉴스 API 응답 날짜 형식 예: 'Tue, 06 Apr 2026 14:30:10 +0900'
    """
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # 형식 변환에 실패할 경우 원본을 그대로 반환
        return date_str

DOMAIN_TO_PRESS = {
    "chosun.com":       "조선일보",
    "joongang.co.kr":   "중앙일보",
    "donga.com":        "동아일보",
    "hani.co.kr":       "한겨레",
    "khan.co.kr":       "경향신문",
    "ohmynews.com":     "오마이뉴스",
    "yonhapnews.co.kr": "연합뉴스",
    "yna.co.kr":        "연합뉴스",
    "news.kbs.co.kr":   "KBS뉴스",
    "imnews.imbc.com":  "MBC뉴스",
    "news.sbs.co.kr":   "SBS뉴스",
    "ytn.co.kr":        "YTN",
    "mbn.co.kr":        "MBN",
    "hankyung.com":     "한국경제",
    "mt.co.kr":         "머니투데이",
    "etnews.com":       "전자신문",
    "zdnet.co.kr":      "ZDNet Korea",
    "mk.co.kr":         "매일경제",
    "sedaily.com":      "서울경제",
}

def extract_press_from_url(url):
    """URL 도메인에서 언론사명을 추출합니다."""
    if not url:
        return "알 수 없음"
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip("www.")
        for key, name in DOMAIN_TO_PRESS.items():
            if key in domain:
                return name
        # 매핑 없으면 도메인 자체를 반환
        return domain or "알 수 없음"
    except Exception:
        return "알 수 없음"

def fetch_naver_news(query, display=10):
    """
    네이버 뉴스 API를 통해 검색 결과를 가져오는 함수.
    
    Args:
        query (str): 검색할 키워드
        display (int): 검색 개수 (기본 10개, 매개변수로 조절 가능)
        
    Returns:
        list: 뉴스 기사 정보 리스트 [제목, 언론사, URL, 본문 요약, 게시일]
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise RuntimeError("Naver credentials missing")

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "sort": "sim" # 'sim': 유사도순, 'date': 최신순
    }

    try:
        # API 요청 보내기
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status() # 4xx, 5xx 에러 발생 시 HTTPError 던짐
        
        data = response.json()
        items = data.get("items", [])
        
        results = []
        for item in items:
            title = clean_html(item.get("title", ""))
            description = clean_html(item.get("description", ""))
            
            # originallink 도메인에서 언론사명 추출
            original_url = item.get("originallink", "")
            press = extract_press_from_url(original_url)

            # originallink: 언론사 본문 링크, link: 네이버 뉴스 링크
            # (차후 Phase 3 내용 크롤링 시, 네이버 뉴스 링크가 파싱하기 쉽습니다)
            url_link = item.get("link", "")
            
            pub_date = parse_date(item.get("pubDate", ""))
            
            results.append({
                "title": title,
                "press": press,
                "url": url_link,
                "original_url": original_url,
                "published_at": item.get("pubDate") or None,  # Preserve the source timezone.
                "pub_date": pub_date,
                "description": description,
                "source_type": "naver_api",
            })
            
        return results

    except Exception as err:
        raise RuntimeError(f"Naver collection failed: {type(err).__name__}") from None

if __name__ == "__main__":
    # 간단한 테스트 시나리오
    test_query = "인공지능"
    print(f"'{test_query}'(으)로 뉴스 검색 데이터 테스트...\n")
    news_list = fetch_naver_news(test_query, display=10)
    
    for i, news in enumerate(news_list, 1):
        print(f"[{i}] {news['title']}")
        print(f"    - 언론사: {news['press']}")
        print(f"    - 게시일: {news['pub_date']}")
        print(f"    - URL: {news['url']}")
        print(f"    - 요약: {news['description']}\n")
