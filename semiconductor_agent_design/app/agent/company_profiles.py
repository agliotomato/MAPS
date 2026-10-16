from __future__ import annotations

from typing import Any


COMPANY_PROFILES: dict[str, dict[str, Any]] = {
    "005930": {
        "code": "005930",
        "name_en": "Samsung Electronics",
        "name_ko": "삼성전자",
        "role": "IDM",
        "role_ko": "종합 반도체 기업",
        "core_businesses": [
            "DRAM / NAND memory",
            "System LSI",
            "Foundry",
            "Advanced packaging",
        ],
        "core_businesses_ko": [
            "디램 / 낸드 메모리",
            "시스템 LSI",
            "파운드리",
            "첨단 패키징",
        ],
        "value_chain_position": [
            "memory manufacturer",
            "logic/foundry manufacturer",
            "advanced node process developer",
        ],
        "value_chain_position_ko": [
            "메모리 제조사",
            "로직/파운드리 제조사",
            "선단 공정 개발사",
        ],
        "key_tech_exposures": [
            "HBM",
            "DRAM",
            "V-NAND",
            "GAA",
            "EUV",
            "advanced packaging",
        ],
        "key_tech_exposures_ko": [
            "HBM",
            "DRAM",
            "V-NAND",
            "GAA",
            "EUV",
            "첨단 패키징",
        ],
        "why_it_moves": [
            "memory pricing and mix",
            "HBM competitiveness",
            "foundry yield / node progress",
            "capex and advanced packaging execution",
        ],
        "why_it_moves_ko": [
            "메모리 가격과 제품 믹스",
            "HBM 경쟁력",
            "파운드리 수율 / 공정 진전",
            "설비투자와 첨단 패키징 실행력",
        ],
    },
    "000660": {
        "code": "000660",
        "name_en": "SK hynix",
        "name_ko": "SK하이닉스",
        "role": "IDM",
        "role_ko": "메모리 중심 종합 반도체 기업",
        "core_businesses": [
            "DRAM",
            "NAND",
            "HBM",
            "memory packaging / integration",
        ],
        "core_businesses_ko": [
            "DRAM",
            "NAND",
            "HBM",
            "메모리 패키징 / 적층",
        ],
        "value_chain_position": [
            "memory manufacturer",
            "HBM supply leader",
            "advanced memory integration company",
        ],
        "value_chain_position_ko": [
            "메모리 제조사",
            "HBM 공급 핵심 업체",
            "첨단 메모리 적층/통합 업체",
        ],
        "key_tech_exposures": [
            "HBM",
            "DRAM",
            "TSV",
            "hybrid bonding",
            "CXL",
            "PIM",
        ],
        "key_tech_exposures_ko": [
            "HBM",
            "DRAM",
            "TSV",
            "하이브리드 본딩",
            "CXL",
            "PIM",
        ],
        "why_it_moves": [
            "HBM shipment / qualification milestones",
            "memory pricing and cycle turns",
            "yield / stack complexity execution",
            "AI accelerator customer demand",
        ],
        "why_it_moves_ko": [
            "HBM 출하 / 고객 승인 이정표",
            "메모리 가격과 업황 전환",
            "수율 / 적층 난이도 실행력",
            "AI 가속기 고객 수요",
        ],
    },
    "042700": {
        "code": "042700",
        "name_en": "Hanmi Semiconductor",
        "name_ko": "한미반도체",
        "role": "Semiconductor equipment supplier",
        "role_ko": "반도체 장비 업체",
        "core_businesses": [
            "semiconductor packaging equipment",
            "TC bonder / hybrid bonding-related equipment",
            "inspection / process equipment for advanced packaging",
        ],
        "core_businesses_ko": [
            "반도체 후공정 / 패키징 장비",
            "TC 본더 / 하이브리드 본딩 관련 장비",
            "첨단 패키징용 검사 / 공정 장비",
        ],
        "value_chain_position": [
            "back-end equipment vendor",
            "beneficiary of HBM and advanced packaging capex",
            "supplier linked to OSAT / memory packaging transitions",
        ],
        "value_chain_position_ko": [
            "후공정 장비 공급사",
            "HBM / 첨단 패키징 설비투자 수혜 가능 업체",
            "OSAT / 메모리 패키징 전환과 연결된 공급사",
        ],
        "key_tech_exposures": [
            "advanced packaging",
            "hybrid bonding",
            "TSV",
            "interposer",
            "inspection / metrology",
        ],
        "key_tech_exposures_ko": [
            "첨단 패키징",
            "하이브리드 본딩",
            "TSV",
            "인터포저",
            "검사 / 계측",
        ],
        "why_it_moves": [
            "HBM packaging capex cycle",
            "customer equipment orders",
            "hybrid bonding adoption speed",
            "OSAT / memory customer expansion",
        ],
        "why_it_moves_ko": [
            "HBM 패키징 설비투자 사이클",
            "고객 장비 발주",
            "하이브리드 본딩 채택 속도",
            "OSAT / 메모리 고객 확대",
        ],
    },
}


def get_company_profile(company_code: str) -> dict[str, Any]:
    code = str(company_code or "").strip().upper()
    return dict(COMPANY_PROFILES.get(code, {}))
