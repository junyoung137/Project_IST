import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

#  한글 폰트 경로 지정
font_path = "C:/Windows/Fonts/malgun.ttf" 
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

# 폰트 설정
plt.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False

# 무역 긴장도 지수 불러오기 및 준비
tension_df = pd.read_csv('google_trends_tension_index.csv')
tension_df['date'] = pd.to_datetime(tension_df['date'])
tension_df = tension_df.rename(columns={'weighted_score': 'tension'})

# SCFI 데이터 불러오기 및 전처리
file_path = r"C:\Users\kj100\Desktop\data_project\code\prepare\china_scfi.xls"
df = pd.read_excel(file_path)
df = df[['구분', '운임지수', '등록일']]
scfi_df = df[df['구분'] == 'SCFI'].copy()
scfi_df['등록일'] = pd.to_datetime(scfi_df['등록일'])
scfi_df = scfi_df[(scfi_df['등록일'] >= '2023-11-01') & (scfi_df['등록일'] <= '2025-05-31')]
scfi_df['운임지수'] = scfi_df['운임지수'].replace(',', '', regex=True).astype(float)
scfi_df = scfi_df.rename(columns={'등록일': 'date', '운임지수': 'scfi'})
scfi_df = scfi_df[['date', 'scfi']]

# 주 단위로 변환 및 병합
tension_df['week'] = tension_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
scfi_df['week'] = scfi_df['date'].dt.to_period('W').apply(lambda r: r.start_time)

weekly_tension = tension_df.groupby('week')['tension'].mean().reset_index()
weekly_scfi = scfi_df.groupby('week')['scfi'].mean().reset_index()

merged_df = pd.merge(weekly_tension, weekly_scfi, on='week', how='inner')

# 정규화
scaler = MinMaxScaler()
merged_df[['tension_scaled', 'scfi_scaled']] = scaler.fit_transform(merged_df[['tension', 'scfi']])

# 상관계수 계산
if len(merged_df) >= 2:
    pearson_corr, pearson_pval = pearsonr(merged_df['tension'], merged_df['scfi'])
    
    print(f"피어슨 상관계수: {pearson_corr:.4f}, p-value: {pearson_pval:.4f}")
else:
    print("병합된 데이터가 2개 미만입니다.")

# 시각화 꺾은선 그래프

import seaborn as sns

sns.set_theme(style="whitegrid", font='Malgun Gothic', rc={"axes.unicode_minus": False})

fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = 'royalblue'
line1 = sns.lineplot(x=merged_df['week'], y=merged_df['tension_scaled'], ax=ax1,
                     color=color1, linewidth=2, label='무역 긴장도 지수')

ax1.set_ylabel('무역 긴장도 지수', color=color1, fontsize=12)
ax1.set_xlabel('') 
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
color2 = 'firebrick'
line2 = sns.lineplot(x=merged_df['week'], y=merged_df['scfi_scaled'], ax=ax2,
                     color=color2, linewidth=2, label='SCFI 운임지수')
ax2.set_ylabel('SCFI 운임지수', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)

# 오른쪽 위 Seaborn 범례제거
if ax2.legend_:
    ax2.legend_.remove()

# 통합 범례 (왼쪽 위)
lines = [line1.lines[0], line2.lines[0]]
labels = ['무역 긴장도 지수', 'SCFI 운임지수']
ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True)

plt.title('무역 긴장도 지수 vs SCFI 운임지수', fontsize=15, weight='bold', pad=20)

fig.tight_layout()
plt.show()

# 산점도 분포 시각화

fig, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(
    data=merged_df,
    x='tension',   
    y='scfi',      
    s=70,
    color='royalblue',
    edgecolor='black',
    alpha=0.7,
    ax=ax
)

ax.set_title('무역 긴장도 지수 vs SCFI 운임지수', fontsize=15, weight='bold', pad=20)
ax.set_xlabel('무역 긴장도 지수', fontsize=12)
ax.set_ylabel('SCFI 운임지수', fontsize=12)

plt.tight_layout()
plt.show()

#################### 부산항 물돌량 #####################

import os
import re

folder_path_c = r"C:\Users\kj100\Desktop\data_project\code\prepare\busan_all"
compare_data = []

# 모든 .csv 파일 반복
for filename in os.listdir(folder_path_c):
    if filename.endswith(".csv") and filename.startswith("busan_"):
        # 날짜 추출
        match = re.search(r'busan_(\d{6})\.csv', filename)
        if match:
            date_str = match.group(1)
            year = int('20' + date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            formatted_date = f"{year}-{month:02}-{day:02}"

            file_path = os.path.join(folder_path_c, filename)

            try:
                with open(file_path, 'r', encoding='euc-kr') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip().startswith("합계,"):
                            parts = line.strip().split(",")
                            if len(parts) >= 3:
                                val1 = parts[1].replace('"', '').strip()
                                val2 = parts[2].replace('"', '').strip()
                                value = f"{val1},{val2}"
                                compare_data.append({
                                    'date': formatted_date,
                                    '합계': value
                                })
                            break  
            except Exception as e:
                print(f"에러 - 파일: {filename}, 내용: {e}")

# 결과 확인
print(f"총 {len(compare_data)}개의 합계 데이터를 수집했습니다.")

###########  부산항 합계 데이터를 DataFrame으로 변환 ######

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

busan_df = pd.DataFrame(compare_data)
busan_df['date'] = pd.to_datetime(busan_df['date'])

# 합계값 분할 및 정리
busan_df[['val1', 'val2']] = busan_df['합계'].str.split(",", expand=True)
busan_df['val1'] = busan_df['val1'].str.replace(',', '', regex=False).astype(float)
busan_df['val2'] = busan_df['val2'].str.replace(',', '', regex=False).astype(float)

# 전체 합계 계산
busan_df['total'] = busan_df['val1'] + busan_df['val2']

# 주간 단위로 변환
busan_df['week'] = busan_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_busan = busan_df.groupby('week')['total'].mean().reset_index()

#################### 무역 긴장도 & SCFI 데이터 준비 ####################

merged_df = pd.merge(weekly_tension, weekly_scfi, on='week', how='inner')
merged_df = pd.merge(merged_df, weekly_busan, on='week', how='inner') 

#################### 상관관계 분석 ####################
if len(merged_df) >= 2:
    # 피어슨 상관계수 계산
    pearson_t_b, pearson_t_p = pearsonr(merged_df['tension'], merged_df['total'])
    pearson_s_b, pearson_s_p = pearsonr(merged_df['scfi'], merged_df['total'])

    print(f"무역 긴장도 vs 부산항 물동량 - 피어슨 상관계수: {pearson_t_b:.4f}, 유의확률: {pearson_t_p:.4f}")
    print(f"SCFI vs 부산항 물동량 - 피어슨 상관계수: {pearson_s_b:.4f}, 유의확률: {pearson_s_p:.4f}")
else:
    print("병합된 데이터가 2개 미만입니다. 상관계수 계산 불가.")

##################시각화 ####################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font='Malgun Gothic', rc={"axes.unicode_minus": False})

fig, ax1 = plt.subplots(figsize=(14, 6))

# 부산항 물동량
bar = ax1.bar(merged_df['week'], merged_df['total'], 
              color='royalblue', width=5, alpha=0.8, 
              edgecolor='navy', label='부산항 물동량')

ax1.set_ylabel('부산항 물동량 (TEU)', color='royalblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1.tick_params(axis='x', rotation=30)

# x축 레이블 간격 줄이기
ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax1.set_xticks(merged_df['week'][::4])
ax1.set_xticklabels(merged_df['week'].dt.strftime('%Y-%m-%d')[::4])

# 두 번째 y축
ax2 = ax1.twinx()

# 긴장도지수
line1 = ax2.plot(merged_df['week'], merged_df['tension'], 
                 color='crimson', linewidth=2, linestyle='--', label='긴장도지수')[0]

# SCFI
line2 = ax2.plot(merged_df['week'], merged_df['scfi'], 
                 color='forestgreen', linewidth=2, linestyle='-.', label='SCFI')[0]

ax2.set_ylabel('긴장도지수 / SCFI 지수', fontsize=12)
ax2.tick_params(axis='y')

# 범례 위치 조정 및 순서 변경
lines = [bar, line2, line1]
labels = ['부산항 물동량', 'SCFI', '무역 긴장도']
ax2.legend(lines, labels, loc='upper left', fontsize=11, frameon=True)

# 제목
plt.title('주간 부산항 물동량 vs 긴장도지수 & SCFI 지수', fontsize=15, weight='bold', pad=20)

fig.tight_layout()
plt.show()

######## 산점도 시각화 #########
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 전역 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 산점도 함수
def draw_correlation_scatter(x, y, xlabel, ylabel, title):
    # 함수 내에서 새 그림 생성 시 폰트 설정 유지
    plt.figure(figsize=(8, 6))
    # seaborn 스타일 조정 (폰트 설정 유지)
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.scatterplot(x=x, y=y, s=80, color='steelblue', edgecolor='k')
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()

# 시각화
draw_correlation_scatter(
    x=merged_df['scfi'],
    y=merged_df['total'],
    xlabel='SCFI',
    ylabel='부산항 주간 물동량',
    title='SCFI vs 부산항 주간 물동량'
)

draw_correlation_scatter(
    x=merged_df['tension'],
    y=merged_df['total'],
    xlabel='긴장도지수',
    ylabel='부산항 주간 물동량',
    title='긴장도지수 vs 부산항 주간 물동량'
)
merged_df.to_csv(r'C:\Users\kj100\Desktop\data_project\code\prepare\merged_df.csv', index=False, encoding='utf-8-sig')
print("[완료] 최종 merged_df.csv 저장됨!")

############ 대시보드 구현 ############
import streamlit as st

#  라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
merged_df = pd.read_csv('data/merged_df.csv', encoding='utf-8-sig')
merged_df['week'] = pd.to_datetime(merged_df['week'])

# 정규화
scaler = MinMaxScaler()
merged_df[['tension_norm', 'scfi_norm']] = scaler.fit_transform(merged_df[['tension', 'scfi']])

# 사이드바 메뉴
st.sidebar.title("📊 대시보드")
st.sidebar.markdown("**긴장도지수와 물동량간 상관관계 분석**")
menu = st.sidebar.radio(
    "시각화 항목 선택",
    (
        '1. 긴장도지수 & SCFI',
        '2. 긴장도지수 vs SCFI',
        '3. 부산항 물동량 vs 주요 지표',
        '4. 부산항 물동량 vs SCFI',
        '5. 부산항 물동량 vs 긴장도지수'
    )
)

# 공통 figure 사이즈
FIGSIZE = (5.5, 3.5)

# 시각화 분기 처리
if menu == '1. 긴장도지수 & SCFI':
    st.subheader("긴장도지수 & SCFI")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(merged_df['week'], merged_df['tension_norm'], marker='o', markersize=3, label='긴장도지수', color='red')
    ax.plot(merged_df['week'], merged_df['scfi_norm'], marker='s', markersize=3, label='SCFI', color='blue')
    ax.set_xlabel('연도', fontsize=7)
    ax.legend(loc='upper left', fontsize=8)
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif menu == '2. 긴장도지수 vs SCFI':
    st.subheader("긴장도지수 vs SCFI 산점도")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=merged_df, x='tension', y='scfi',
                    color='steelblue', edgecolor='k', s=25, alpha=0.7, ax=ax)
    ax.set_xlabel('긴장도지수', fontsize=7)
    ax.set_ylabel('SCFI', fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif menu == '3. 부산항 물동량 vs 주요 지표':
    st.subheader("부산항 물동량 및 주요 지표")
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    bar = ax1.bar(merged_df['week'], merged_df['total'], width=5, color='royalblue', alpha=0.6, label='부산항 물동량')
    ax1.set_ylabel('물동량 (TEU)', fontsize=7, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=7)
    ax1.set_xlabel('연도', fontsize=7)
    ax1.tick_params(axis='x', rotation=45, labelsize=7)

    ax2 = ax1.twinx()
    line1 = ax2.plot(merged_df['week'], merged_df['tension'], color='crimson', linestyle='--', label='무역긴장도')[0]
    line2 = ax2.plot(merged_df['week'], merged_df['scfi'], color='forestgreen', linestyle='-.', label='SCFI')[0]
    ax2.tick_params(labelsize=7)

    lines = [bar, line1, line2]
    labels = ['부산항 물동량', '긴장도지수', 'SCFI']
    ax2.legend(lines, labels, loc='upper left', fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif menu == '4. 부산항 물동량 vs SCFI':
    st.subheader("부산항 물동량 vs SCFI 산점도")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=merged_df, x='scfi', y='total',
                    color='orange', edgecolor='k', s=25, alpha=0.7, ax=ax)
    ax.set_xlabel('SCFI', fontsize=7)
    ax.set_ylabel('물동량 (TEU)', fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif menu == '5. 부산항 물동량 vs 긴장도지수':
    st.subheader("부산항 물동량 vs 긴장도지수 산점도")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=merged_df, x='tension', y='total',
                    color='green', edgecolor='k', s=25, alpha=0.7, ax=ax)
    ax.set_xlabel('무역긴장도', fontsize=7)
    ax.set_ylabel('물동량 (TEU)', fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)