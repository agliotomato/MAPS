# MAPS

뉴스·거시경제·시장·반도체 기술의 근거를 역할별로 분석하고, 원문·기업·시점·기간을 검사해 최종 보고서를 만드는 프로젝트입니다. Python 3.12를 기준으로 정리했습니다.

**2026-10-16 현재:** 운영 분석·보고서 검사와 선택적 상호 검토가 구현돼 있습니다. 상호 검토의 오류 감소나 멀티 에이전트의 예측 우위는 입증되지 않았습니다. 기본 경로는 독립 분석 후 통합이며 `--peer-review`가 뉴스↔반도체 검토1라운드를 추가합니다. [검증 결과와 한계](docs/STATUS.md)를 확인하세요.

## 시작과 오프라인 검증

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B -X utf8 -m unittest discover -s tests -v
```

회귀 검사는 합성 자료와 보존된 작은 입력/응답을 사용하며 실제 모델·DB 호출 없이 실행합니다. 과거 프로젝트 폴더나 API 키가 필요하지 않습니다.

가격 기준선·단일/멀티 어댑터의 연결을 모의 실행하려면:

```powershell
.venv\Scripts\python -B -X utf8 benchmarks/contribution.py --input benchmarks/contribution_fixture.json --output outputs/comparison_smoke
```

출력 폴더는 새 경로를 사용합니다. 이 명령의 응답은 모의 출력이며 실제 예측 성능으로 세지 않습니다.

## 실제 분석 실행

실제 수집·DB 분석에는 추가 의존성과 외부 데이터 서비스가 필요합니다.

```powershell
.venv\Scripts\python -m pip install -r requirements-runtime.txt
Copy-Item .env.example .env
# .env에 실제 API 키와 DB 연결을 로컬에서 설정
.venv\Scripts\python -B -X utf8 main.py "삼성전자 2026-10-16 기준 7/14/30일(달력일) 전망을 분석해줘"
# 선택 기능:
.venv\Scripts\python -B -X utf8 main.py --peer-review "삼성전자 2026-10-16 기준 7/14/30일(달력일) 전망을 분석해줘"
```

`.env`, 뉴스DB, 가격 캐시, PostgreSQL/벡터DB 데이터는 저장소에 포함하지 않습니다. 키만 설정해도 과거 수집 데이터가 자동 복원되지는 않습니다. 서비스가 없거나 자료가 없으면 해당 상태를 표시합니다. 모델 가중치·추가 공급자 라이브러리는 사용하는 기능에 따라 설치합니다.

**`main.py` 자체는 전체 달러 비용 상한을 제공하지 않습니다.** 반도체 내부 호출 수 제한과 전체 비용 제한은 다릅니다. 이전 실험의 비용 장부와 잔여는 [비용 인계](docs/results/budget_handoff.json)에 보존했으며 새 폴더가 예산을 초기화하지 않습니다. 새 유료 비교는 원본 장부 검증과 비용 제한 실행기 연결 후 진행합니다. 이번 이전에서는 실제 API·DB 실행을 하지 않았습니다.

## 핵심 파일

| 경로 | 역할 |
|---|---|
| `main.py`, `agent_team.py` | 실행과 역할 순서 |
| `integration_tools.py` | 네 분야 수집/분석 도구와 보고서 저장 |
| `independent_reasoning.py`, `role_prompts.py` | 독립 가설과 역할 규칙 |
| `peer_review.py` | 원문 기반 상호 검토·작성자 답변 |
| `evidence_*.py`, `financial_claims.py` | 원문·시각·주장·최종 근거 검사 |
| `search_agent/`, `economy_agent/src/`, `market_agent/`, `semiconductor_agent_design/app/` | 네 분야의 운영 코드 |
| `benchmarks/contribution.py` | 고정 입력 비교 어댑터와 모의 CLI |
| `tests/`, `tests/fixtures/` | 원본 프로젝트 없이 실행하는 핵심 회귀 검사 |
| `docs/`, `reports/` | 현재 상태·이전 내역·작업 기록 |

## Git 작업

비밀값·DB·캐시·자동 생성 보고서는 `.gitignore`로 제외합니다. 작은 테스트 입력, 검증 결과 요약, 일일 보고서는 추적합니다. 기존 `origin` 연결은 유지하며 원격 푸시는 별도 실행합니다.

```powershell
git status
git switch -c codex/작업이름
# 수정과 회귀 검사 후
git add 수정한파일
git commit -m "변경 내용"
```

원본 프로젝트와 과거 실험은 삭제하지 않았습니다. 선택/제외 기준과 파일별 원본 해시는 [이전 명세](docs/migration_manifest.json), 변경된 경계와 재개 지점은 [이전 안내](docs/MIGRATION.md), 매일 작업은 [보고서 목록](reports/README.md)에 있습니다.
