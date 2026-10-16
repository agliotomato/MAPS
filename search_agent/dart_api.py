import io
import os
import zipfile
import xml.etree.ElementTree as ET
import requests
from dotenv import load_dotenv

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")
BASE_URL = "https://opendart.fss.or.kr/api"

# 기업 목록 캐시 (프로세스 실행 중 재사용)
_corp_list_cache = None


def _load_corp_list():
    """DART 전체 기업 목록 ZIP을 다운로드하여 파싱합니다. (캐시 적용)"""
    global _corp_list_cache
    if _corp_list_cache is not None:
        return _corp_list_cache
    if not DART_API_KEY:
        raise RuntimeError("DART credentials missing")

    url = f"{BASE_URL}/corpCode.xml"
    params = {"crtfc_key": DART_API_KEY}
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            with z.open("CORPCODE.xml") as f:
                tree = ET.parse(f)

        corps = {}
        for item in tree.getroot().findall("list"):
            corp_code  = item.findtext("corp_code", "").strip()
            corp_name  = item.findtext("corp_name", "").strip()
            stock_code = item.findtext("stock_code", "").strip()
            if corp_code:
                corps[corp_name] = {
                    "corp_code":  corp_code,
                    "corp_name":  corp_name,
                    "stock_code": stock_code,
                }

        _corp_list_cache = corps
        print(f"[DART] 기업 목록 로드 완료: {len(corps)}개")
        return corps

    except Exception as e:
        raise RuntimeError(f"DART company lookup failed: {type(e).__name__}") from None


def fetch_corp_code(corp_name):
    """
    기업명으로 DART corp_code를 검색합니다. (부분 일치 지원)

    Args:
        corp_name (str): 기업명 (예: '삼성전자')

    Returns:
        dict | None: {'corp_code', 'corp_name', 'stock_code'} 또는 None
    """
    corps = _load_corp_list()

    # 완전 일치 우선
    if corp_name in corps:
        return corps[corp_name]

    # 부분 일치
    matches = [v for k, v in corps.items() if corp_name in k]
    if matches:
        return matches[0]

    print(f"[DART] '{corp_name}' 기업을 찾을 수 없습니다.")
    return None


def fetch_disclosures(corp_code=None, bgn_de=None, end_de=None, display=20):
    """
    DART 공시 목록을 조회합니다.

    Args:
        corp_code (str|None): DART 기업 고유코드. None이면 전체 조회.
        bgn_de (str|None):    조회 시작일 (YYYYMMDD). None이면 오늘 기준 최근.
        end_de (str|None):    조회 종료일 (YYYYMMDD).
        display (int):        최대 수집 건수 (최대 100)

    Returns:
        list: [{'rcept_no', 'corp_code', 'corp_name', 'report_type', 'title', 'filed_at'}, ...]
    """
    url = f"{BASE_URL}/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "page_count": min(display, 100),
        "sort": "date",
        "sort_mth": "desc",
    }
    if corp_code:
        params["corp_code"] = corp_code
    if bgn_de:
        params["bgn_de"] = bgn_de
    if end_de:
        params["end_de"] = end_de

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if data.get("status") not in ("000", "013"):
            raise RuntimeError("DART returned a failure status")

        items = data.get("list", [])
        results = []
        for item in items:
            results.append({
                "rcept_no":    item.get("rcept_no", ""),
                "corp_code":   item.get("corp_code", ""),
                "corp_name":   item.get("corp_name", ""),
                "report_type": item.get("pblntf_detail_ty", ""),
                "title":       item.get("report_nm", ""),
                "filed_at":    item.get("rcept_dt", ""),
                "url":         ("https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + item["rcept_no"])
                               if item.get("rcept_no") else None,
            })

        print(f"[DART] {len(results)}건 공시 수집 완료")
        return results

    except Exception as e:
        raise RuntimeError(f"DART disclosure lookup failed: {type(e).__name__}") from None


if __name__ == "__main__":
    # 테스트: 삼성전자 기업코드 조회 후 최신 공시 5건
    corp = fetch_corp_code("삼성전자")
    if corp:
        print(f"기업코드: {corp['corp_code']} / 종목코드: {corp['stock_code']}")
        disclosures = fetch_disclosures(corp_code=corp["corp_code"], display=5)
        for d in disclosures:
            print(f"[{d['filed_at']}] {d['title']} ({d['report_type']})")
