import pandas as pd
from sqlalchemy import create_engine, text

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
file = r"C:\Users\KDT6\Downloads\모델링_UI\ai_data\병목\results_block1_bottleneck_predictions_fixed.csv"

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

    # 필요한 컬럼만 추출
    update_df = df[["Time_Now", "Bottleneck_actual_Cell", "Bottleneck_pred_Cell"]].dropna()

    # ===== UPDATE 실행 =====
    with engine.begin() as conn:
        for _, row in update_df.iterrows():
            conn.execute(
                text(f"""
                    UPDATE {TABLE_NAME}
                    SET Bottleneck_actual_Cell = :actual,
                        Bottleneck_pred_Cell   = :pred
                    WHERE Time_Now = :time_now
                """),
                {
                    "actual": row["Bottleneck_actual_Cell"],
                    "pred": row["Bottleneck_pred_Cell"],
                    "time_now": row["Time_Now"]
                }
            )

    print(f"✅ 두 컬럼 업데이트 완료 (rows={len(update_df)})")

except Exception as e:
    print(f"❌ 업데이트 실패: {file} -> {e}")
