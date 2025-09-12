import pandas as pd
from sqlalchemy import create_engine

# ===== DB 연결 정보 =====
MYSQL_USER = "ajin"
MYSQL_PW   = "agin1234"
MYSQL_HOST = "database-1.c3asgoye8svw.ap-northeast-2.rds.amazonaws.com"
MYSQL_DB   = "AJIN_newDB"
TABLE_NAME = "daily_production"

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PW}@{MYSQL_HOST}/{MYSQL_DB}"
)

# ===== 처리할 CSV 파일 =====
files = [
    r"C:\Users\KDT6\Downloads\모델링_UI\ai_data\생산량\SKU별_일자별_데이터(24.01-25.06) - 수정.csv",
]

# ===== CSV → MySQL 적재 =====
for file in files:
    try:
        # CSV 읽기 (UTF-8 우선, 안되면 CP949)
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="cp949")

        # 컬럼 이름 정리 (BOM 제거 등)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # ✅ % 제거 후 숫자로 변환
        for col in df.columns:
            if "pct_error" in col.lower():
                df[col] = df[col].astype(str).str.replace("%", "").astype(float)

        # MySQL 테이블에 적재
        df.to_sql(
            name=TABLE_NAME,
            con=engine,
            if_exists="append",  # append = 기존 데이터 유지 + 뒤에 추가
            index=False,
            method="multi",
            chunksize=1000
        )

        print(f"✅ 삽입 성공: {file} (rows={len(df)})")

    except Exception as e:
        print(f"❌ 삽입 실패: {file} -> {e}")
