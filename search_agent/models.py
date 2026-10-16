from sqlalchemy import Column, Integer, String, Text, ForeignKey, BigInteger, Float, DateTime
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)

Base = declarative_base()

class Press(Base):
    __tablename__ = 'press'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    # 신뢰도 점수는 에이전트가 참고하는 핵심 데이터. 기본값 설정.
    reliability_score = Column(Integer, default=50)  
    
    reporters = relationship("Reporter", back_populates="press")
    news_articles = relationship("News", back_populates="press")


class Reporter(Base):
    __tablename__ = 'reporter'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    
    # 소속 언론사 외래키
    press_id = Column(Integer, ForeignKey('press.id'), nullable=True) 
    
    # 에이전트가 참고할 판단 근거 정보들
    correction_count = Column(Integer, default=0)
    reporter_score = Column(Integer, default=50)   # 기자 신뢰도 점수 (0~100)
    history_details = Column(Text, nullable=True)  # 구체적인 사유나 이력 저장 (JSON 문자열 저장 권장)
    email = Column(String(100), nullable=True)     # Phase 4: 기자 이메일 주소 추가

    press = relationship("Press", back_populates="reporters")
    news_articles = relationship("News", back_populates="reporter")


class News(Base):
    __tablename__ = 'news'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    url = Column(String(500), unique=True, nullable=False) # 동일 기사 중복 저장 방지
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)  # 차후 Phase 3(크롤링) 단계에서 채워질 본문
    summary = Column(Text, nullable=True)  # Phase 4: 본문 기반 첫 3문장 간이 요약
    pub_date = Column(String(50), nullable=True)
    source_type = Column(String(50), nullable=True)   # 수집 출처: naver_api / rss_naver / rss_google
    sentiment_score = Column(Integer, nullable=True)  # 감성 점수 (-100 ~ 100, KoBERT 결과)
    
    # 외래키 설정
    press_id = Column(Integer, ForeignKey('press.id'), nullable=True)
    reporter_id = Column(Integer, ForeignKey('reporter.id'), nullable=True)
    
    press = relationship("Press", back_populates="news_articles")
    reporter = relationship("Reporter", back_populates="news_articles")


# ── DART 공시 ──────────────────────────────────────────────

class Company(Base):
    __tablename__ = 'company'

    id = Column(Integer, primary_key=True, autoincrement=True)
    corp_code = Column(String(20), unique=True, nullable=False)  # DART 기업 고유코드
    corp_name = Column(String(100), nullable=False)              # 기업명
    stock_code = Column(String(20), nullable=True)               # 종목코드

    disclosures = relationship("Disclosure", back_populates="company")


class Disclosure(Base):
    __tablename__ = 'disclosure'

    id = Column(Integer, primary_key=True, autoincrement=True)
    rcept_no = Column(String(20), unique=True, nullable=False)   # 공시번호
    report_type = Column(String(100), nullable=True)             # 공시유형 (사업보고서 등)
    title = Column(String(300), nullable=False)
    content = Column(Text, nullable=True)                        # 크롤링한 원문
    filed_at = Column(String(50), nullable=True)                 # 공시일시

    company_id = Column(Integer, ForeignKey('company.id'), nullable=False)
    company = relationship("Company", back_populates="disclosures")


# ── Threads ────────────────────────────────────────────────

class ThreadsAccount(Base):
    __tablename__ = 'threads_account'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(100), unique=True, nullable=False)  # Threads 계정 ID
    username = Column(String(100), nullable=False)
    followers = Column(BigInteger, default=0)
    is_verified = Column(Integer, default=0)                       # 1: 인증, 0: 미인증
    reliability_score = Column(Integer, default=50)                # 계정 신뢰도 점수

    posts = relationship("ThreadsPost", back_populates="account")


class ThreadsPost(Base):
    __tablename__ = 'threads_post'

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(100), unique=True, nullable=False)     # Threads 포스트 ID
    content = Column(Text, nullable=True)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    repost_count = Column(Integer, default=0)
    posted_at = Column(String(50), nullable=True)

    account_id = Column(Integer, ForeignKey('threads_account.id'), nullable=False)
    account = relationship("ThreadsAccount", back_populates="posts")


# ── LLM 답변 저장 ────────────────────────────────────────────

class AgentResponse(Base):
    __tablename__ = 'agent_response'

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)                        # 사용자 질문
    model_name = Column(String(100), nullable=False)            # gpt-4o, claude-sonnet 등
    response = Column(Text, nullable=True)                      # LLM 답변
    created_at = Column(DateTime, default=utcnow)


# ── 신뢰도 계산 이력 ──────────────────────────────────────────

class ReliabilityLog(Base):
    __tablename__ = 'reliability_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(Integer, ForeignKey('news.id'), nullable=False)
    final_score = Column(Float, nullable=True)                  # 최종 신뢰도 점수
    sentiment_score = Column(Float, nullable=True)              # 감성 점수
    press_score = Column(Float, nullable=True)                  # 언론사 점수
    reporter_score = Column(Float, nullable=True)               # 기자 점수
    freshness = Column(Float, nullable=True)                    # 최신성 점수
    created_at = Column(DateTime, default=utcnow)

    news = relationship("News")
