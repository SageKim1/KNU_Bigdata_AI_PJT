import pandas as pd
from sqlalchemy import create_engine

# ===== DB 연결 정보 =====
MYSQL_USER = "ajin"
MYSQL_PW   = "agin1234"
MYSQL_HOST = "database-1.c3asgoye8svw.ap-northeast-2.rds.amazonaws.com"
MYSQL_DB   = "AJIN_newDB"
TABLE_NAME = "bottleneck_overview"

# ===== SQLAlchemy 엔진 생성 =====
engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PW}@{MYSQL_HOST}/{MYSQL_DB}"
)

# ===== CSV 파일 경로 =====
file = r"C:\Users\KDT6\Downloads\모델링_UI\ai_data\병목\results_block1_bottleneck_predictions.csv"

try:
    # CSV 읽기 (UTF-8 우선, 안되면 CP949)
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="cp949")

    # 컬럼 이름 정리 (BOM 제거 등)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # 날짜/시간 변환
    df["Time_Now"] = pd.to_datetime(df["Time_Now"], errors="coerce")

    # ===== DB 적재 =====
    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="append",   # 기존 데이터 유지 + 새 데이터 추가
        index=False,
        method="multi",       # 대량 insert 최적화
        chunksize=1000        # 1000행 단위로 나눠서 insert
    )

    print(f"✅ 전체 데이터 삽입 완료: {file} (rows={len(df)})")

except Exception as e:
    print(f"❌ 삽입 실패: {file} -> {e}")
