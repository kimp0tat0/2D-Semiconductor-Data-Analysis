import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터베이스 로드 및 전처리 (Data Engineering)
# DB 연결 및 쪼개진 데이터 로드
conn = sqlite3.connect('c2db.db')
df_num = pd.read_sql_query("SELECT id, key, value FROM number_key_values", conn)
df_text = pd.read_sql_query("SELECT id, key, value FROM text_key_values", conn)
conn.close()

# 파이썬 Pandas를 이용해 데이터를 보기 좋은 표 형태로 병합 (Merge)
df_num_pivot = df_num.pivot(index='id', columns='key', values='value').reset_index()
df_text_pivot = df_text.pivot(index='id', columns='key', values='value').reset_index()
df_raw = pd.merge(df_text_pivot, df_num_pivot, on='id', how='outer')

# 화학식(Formula) 추출
df_raw['Formula'] = df_raw['folder'].apply(lambda x: str(x).split('/')[-1].split('-')[0])

# 결측치 보완: 초정밀 데이터(gap_hse)가 비어있으면 일반 데이터(gap)로 채우기
df_raw['gap_final'] = df_raw['gap_hse'].fillna(df_raw['gap'])
df_raw['efermi_final'] = df_raw['efermi_hse'].fillna(df_raw['efermi'])

# 전체 데이터 개수 추적
count_raw = len(df_raw)
print(f"전처리 완료! 확보된 전체 Raw Data: {count_raw:,}개")

# 2. 절대 조건 필터링 (1단계 & 2단계 판별 알고리즘)
# ==========================================
# 1단계: 밴드갭 1.0 ~ 2.0 eV 필터링
cond_bandgap = (df_raw['gap_final'] >= 1.0) & (df_raw['gap_final'] <= 2.0)
df_step1 = df_raw[cond_bandgap].copy()

# 2단계: 2D 층상 구조를 나타내는 특정 공간군 번호 필터링
target_space_groups = [162, 164, 166, 187, 191, 194] 
cond_spacegroup = df_step1['number'].isin(target_space_groups)
df_step2 = df_step1[cond_spacegroup].copy()

# 3. 스코어링 알고리즘 (3단계 & 4단계)
# 3단계: 형성 에너지 0 이하(안정) 조건 및 100점 만점 스코어링
df_step3 = df_step2[df_step2['hform'] <= 0].copy() 
E_min = df_step3['hform'].min()
df_step3['Score_Stability'] = 100 * (df_step3['hform'] / E_min)

# 4단계: 상용 전극(Au) 일함수 5.1eV 기준 가우시안 스코어링
df_step3['Work_Function'] = df_step3['efermi_final'].abs()
target_wf = 5.1 
sigma = 0.5     
df_step3['Score_Contact'] = 100 * np.exp(-((df_step3['Work_Function'] - target_wf)**2) / (2 * sigma**2))

# 최종 총점 합산 및 내림차순 정렬
df_step3['Total_Score'] = df_step3['Score_Stability'] + df_step3['Score_Contact']
df_final = df_step3.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)

print("\n=== 차세대 2D 반도체 최우수 후보물질 Top 5 ===")
print(df_final[['Formula', 'Score_Stability', 'Score_Contact', 'Total_Score']].head(5))

# 4. 분석 결과 시각화 (최종 Top 10 바 차트 단독 출력)
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6)) # 단일 그래프에 맞게 비율 크기 조정

top_10 = df_final.head(10)
bars = plt.bar(top_10['Formula'], top_10['Total_Score'], color='#4C72B0', edgecolor='black', width=0.5)

# 축 및 타이틀 세팅 
plt.title('Top 10 Neuromorphic 2D Candidates', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Material Formula', fontsize=13, fontweight='bold', labelpad=10)
plt.ylabel('Total Score (Max 200)', fontsize=13, fontweight='bold', labelpad=10)
plt.ylim(0, 220) # 점수 텍스트가 잘리지 않도록 천장 여백 확보
plt.xticks(rotation=45, fontsize=12, fontweight='bold') 
plt.yticks(fontsize=11)

# 막대 위에 정확한 Total Score 수치 텍스트 얹기
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval:.1f}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

