from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 폴더 경로 설정
folder_path = Path(r'C:\Users\kj100\Desktop\data_project\code\prepare\keywords')

# CSV 파일 목록
keywords_files = list(folder_path.glob('*.csv'))

# 키워드별 가중치 딕셔너리
weights = {
    'china export control': 1.5,
    'chip war': 1.2,
    'economic sanctions china': 1.4,
    'export ban': 1.3,
    'retaliatory tariffs': 1.2,
    'section 301 tariffs': 1.1,
    'tariff escalation': 1.3,
    'tariff war': 1.4,
    'trade policy uncertainty': 1.0
}

# 결과 저장용 리스트
score_list = []

for file in keywords_files:
    keyword_name = file.stem.lower()
    
    # 가중치 없는 키워드는 무시
    if keyword_name not in weights:
        continue

    weight = weights[keyword_name]
    
    # CSV 파일 읽기 (첫 줄 스킵)
    df = pd.read_csv(file, skiprows=1, header=None, names=['date', 'value'])

    # 날짜 변환 & 결측 제거
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date']) 

    # 숫자형으로 변환 
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    # 가중치 반영 점수 계산
    df['weighted_score'] = df['value'] * weight
    df['keyword'] = keyword_name

    score_list.append(df[['date', 'weighted_score']])

# 빈 DF 제거
score_list = [df for df in score_list if not df.empty]

# 모든 데이터 비었을 경우 예외 처리
if not score_list:
    raise ValueError("모든 키워드 파일이 비어있습니다.")

# 일자별 가중 점수 합산
all_scores = pd.concat(score_list, ignore_index=True)
daily_tension = all_scores.groupby('date', as_index=False)['weighted_score'].sum()

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(daily_tension['date'], daily_tension['weighted_score'], marker='o', color='blue', label='무역 긴장도 지수')
plt.title('Google Trends 기반 무역 긴장도 지수')
plt.xlabel('Date')
plt.ylabel('Weighted Tension Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# CSV 저장
output_path = folder_path / 'google_trends_tension_index.csv'
daily_tension.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"최종 무역 긴장도 지수 파일 저장 완료 : {output_path}")