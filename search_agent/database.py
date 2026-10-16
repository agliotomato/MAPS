from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Press, Reporter, News, Company, Disclosure, ThreadsAccount, ThreadsPost

DATABASE_URL = "sqlite:///news_data.db"

# Engine 생성 및 SessionLocal 설정
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """데이터베이스 파일과 내부 테이블들을 자동으로 생성합니다."""
    Base.metadata.create_all(bind=engine)

def save_news_to_db(news_list):
    """
    Phase 1에서 가져온 뉴스 리스트를 DB에 한꺼번에 밀어 넣는 함수.
    
    Args:
        news_list: [{'title': ..., 'url': ..., 'press': ..., 'pub_date': ..., 'description': ...}, ...]
    """
    db = SessionLocal()
    try:
        saved_count = 0
        duplicate_count = 0
        seen_urls = set()  # 같은 배치 내 중복 방지

        for data in news_list:
            url = data.get('url', '')
            if not url:
                continue

            if url in seen_urls:
                duplicate_count += 1
                continue
            seen_urls.add(url)

            # 중복 URL 체크 (이미 존재하는 기사면 스킵 - UPSERT 방어 로직)
            existing_news = db.query(News).filter(News.url == url).first()
            if existing_news:
                duplicate_count += 1
                continue

            # 1. 언론사(Press) 매칭 및 생성
            # (API가 언론사를 안주면 기본으로 '알 수 없음' 사용)
            press_name = data.get('press', '알 수 없음')
            press_obj = db.query(Press).filter(Press.name == press_name).first()
            if not press_obj:
                press_obj = Press(name=press_name, reliability_score=50)
                db.add(press_obj)
                db.commit() # 외래키로 사용하기 위해 커밋
                
            # 2. 기자(Reporter) 매칭 및 생성
            # (현재 뉴스 데이터에 없으면 'Unknown'으로 처리)
            reporter_name = data.get('reporter', 'Unknown')
            reporter_obj = db.query(Reporter).filter(Reporter.name == reporter_name, Reporter.press_id == press_obj.id).first()
            if not reporter_obj:
                reporter_obj = Reporter(name=reporter_name, press_id=press_obj.id, correction_count=0)
                db.add(reporter_obj)
                db.commit()

            title = data.get('title', '')
            description = data.get('description', '')
            
            # 신뢰도 임시 스코어링
            # 내용에 "정정", "사과", "오보" 가 있으면 기자의 correction_count 증가 처리
            combined_text = title + " " + description
            if any(keyword in combined_text for keyword in ["정정", "사과", "오보"]):
                reporter_obj.correction_count += 1
                db.commit()

            # 3. News 객체 생성
            new_news = News(
                title=title,
                url=url,
                description=description,
                pub_date=data.get('pub_date', ''),
                source_type=data.get('source_type'),
                press_id=press_obj.id,
                reporter_id=reporter_obj.id
            )
            db.add(new_news)
            saved_count += 1
            
        db.commit() # 전체 커밋
        print(f"[OK] DB 처리 완료: {saved_count}개 신규 저장, {duplicate_count}개 중복 스킵됨.")
        
    except Exception as e:
        db.rollback()
        print(f"[ERROR] DB 저장 중 에러 발생: {e}")
    finally:
        db.close()

def save_disclosures_to_db(disclosures):
    """
    DART 공시 리스트를 DB에 저장합니다.

    Args:
        disclosures: [{'rcept_no', 'corp_code', 'corp_name', 'stock_code', 'report_type', 'title', 'filed_at'}, ...]
    """
    db = SessionLocal()
    try:
        saved_count = 0
        duplicate_count = 0

        for data in disclosures:
            rcept_no = data.get("rcept_no", "")
            if not rcept_no:
                continue

            # 중복 체크
            if db.query(Disclosure).filter(Disclosure.rcept_no == rcept_no).first():
                duplicate_count += 1
                continue

            # 기업 매칭 및 생성
            corp_code = data.get("corp_code", "")
            corp_obj = db.query(Company).filter(Company.corp_code == corp_code).first()
            if not corp_obj:
                corp_obj = Company(
                    corp_code=corp_code,
                    corp_name=data.get("corp_name", ""),
                    stock_code=data.get("stock_code", ""),
                )
                db.add(corp_obj)
                db.commit()

            disclosure = Disclosure(
                rcept_no=rcept_no,
                report_type=data.get("report_type", ""),
                title=data.get("title", ""),
                filed_at=data.get("filed_at", ""),
                company_id=corp_obj.id,
            )
            db.add(disclosure)
            saved_count += 1

        db.commit()
        print(f"[DART] DB 처리 완료: {saved_count}개 저장, {duplicate_count}개 중복 스킵")

    except Exception as e:
        db.rollback()
        print(f"[DART] DB 저장 오류: {e}")
    finally:
        db.close()


def save_threads_accounts_to_db(accounts: list) -> None:
    """
    Threads 인플루언서 계정 정보를 DB에 저장합니다.

    Args:
        accounts: [{'account_id', 'username', 'followers', 'is_verified', 'reliability_score'}, ...]
    """
    db = SessionLocal()
    try:
        saved_count = 0
        updated_count = 0

        for data in accounts:
            account_id = data.get("account_id", "")
            if not account_id:
                continue

            existing = db.query(ThreadsAccount).filter(
                ThreadsAccount.account_id == account_id
            ).first()

            if existing:
                existing.followers = data.get("followers", existing.followers)
                existing.reliability_score = data.get("reliability_score", existing.reliability_score)
                updated_count += 1
            else:
                obj = ThreadsAccount(
                    account_id=account_id,
                    username=data.get("username", ""),
                    followers=data.get("followers", 0),
                    is_verified=data.get("is_verified", 0),
                    reliability_score=data.get("reliability_score", 50),
                )
                db.add(obj)
                saved_count += 1

        db.commit()
        print(f"[Threads] 계정 DB: {saved_count}개 신규, {updated_count}개 업데이트")

    except Exception as e:
        db.rollback()
        print(f"[Threads] 계정 저장 오류: {e}")
    finally:
        db.close()


def save_threads_posts_to_db(posts: list) -> None:
    """
    Threads 포스트 정보를 DB에 저장합니다.

    Args:
        posts: [{'post_id', 'content', 'like_count', 'reply_count', 'repost_count', 'posted_at', 'account_id'}, ...]
    """
    db = SessionLocal()
    try:
        saved_count = 0
        duplicate_count = 0

        for data in posts:
            post_id = data.get("post_id", "")
            if not post_id:
                continue

            if db.query(ThreadsPost).filter(ThreadsPost.post_id == post_id).first():
                duplicate_count += 1
                continue

            # account_id(문자열) → threads_account.id(정수 FK) 변환
            account = db.query(ThreadsAccount).filter(
                ThreadsAccount.account_id == data.get("account_id", "")
            ).first()
            if not account:
                print(f"[Threads] 미등록 계정 스킵: {data.get('account_id')}")
                continue

            obj = ThreadsPost(
                post_id=post_id,
                content=data.get("content", ""),
                like_count=data.get("like_count", 0),
                reply_count=data.get("reply_count", 0),
                repost_count=data.get("repost_count", 0),
                posted_at=data.get("posted_at", ""),
                account_id=account.id,
            )
            db.add(obj)
            saved_count += 1

        db.commit()
        print(f"[Threads] 포스트 DB: {saved_count}개 저장, {duplicate_count}개 중복 스킵")

    except Exception as e:
        db.rollback()
        print(f"[Threads] 포스트 저장 오류: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("DB 테이블 셋업...")
    init_db()
    print("[OK] 완료")
