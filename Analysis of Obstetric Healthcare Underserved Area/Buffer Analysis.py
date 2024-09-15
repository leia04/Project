#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 필요할 라이브러리 설치

import geopandas as gpd
import pandas as pd
import time
import numpy as np
from tqdm import tqdm, trange
from shapely.geometry import Point, MultiPoint
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


# # 전남의 취약 시군구 분석을 위한 호남지역, 경남지역 버퍼분석

# In[3]:


# 전남, 전북, 광주 그리고 경남 지역 shp 파일

Jeonam = gpd.read_file('./LARD_ADM_SECT_SGG_46_202405.shp')
Jeonbuk = gpd.read_file('./LARD_ADM_SECT_SGG_52_202405.shp')
Gyeongnam = gpd.read_file('./LARD_ADM_SECT_SGG_48_202405.shp')
Gwangju = gpd.read_file('./LARD_ADM_SECT_SGG_29_202405.shp')


# - 전남 지도 출력

# In[4]:


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
Jeonam.plot(ax=ax, color='white', edgecolor='black')
plt.show()


# - 호남권과 경남 병상수 1이상을 가진 병원의 shp파일
# - 네 개 지역 병합후 시각화

# In[5]:


# 호남권이랑 경남수 병상 인

Honamhospital = gpd.read_file('./호남권+경남 산부인과 shp파일.shp')


# In[6]:


# 네개 지역 병합

combined_gdf = gpd.GeoDataFrame(pd.concat([Jeonam, Jeonbuk, Gyeongnam, Gwangju], ignore_index=True))


# In[7]:


# 네개 지역 한 번에 병합 후 시각화

Honamhospital = Honamhospital.to_crs(combined_gdf.crs)

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 지역별 지도 그리기
combined_gdf.plot(ax=ax, color='white', edgecolor='black')

# 병원 위치 빨간색 마커로 표시
Honamhospital.plot(ax=ax, marker='o', color='red', markersize=5)

ax.set_title('Jeonam, Jeonbuk, Gyeongnam, and Gwangju with Hospitals')
plt.show()


# In[8]:


combined_gdf.head()


# # 버퍼 분석
# - 호남 및 경남 병원지역 버퍼
# - 버퍼를 통해 전남 인근 지역에서 전남이 아닌 다른 지역으로부터 의료서비스를 받는 곳 파악

# In[9]:


dist = 20000
Honamhospital_buffer = Honamhospital.buffer(dist)


# In[10]:


# 버퍼 다른 종류

fig, ax = plt.subplots(figsize=(7, 10))

Honamhospital_buffer.boundary.plot(ax=ax, color='blue', lw=0.5, zorder=1)
Honamhospital.plot(ax=ax, markersize=10, legend=True, zorder=2)
combined_gdf.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=1)

plt.show()


# - 전남은 붉은색으로 표시

# In[11]:


fig, ax = plt.subplots(figsize=(7, 10))

Honamhospital_buffer.boundary.plot(ax=ax, color='blue', lw=0.5, zorder=1)
Honamhospital.plot(ax=ax, markersize=10, legend=True, zorder=2)
combined_gdf.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=1)

Jeonam.boundary.plot(ax=ax, color='lightcoral', linewidth=0.5)

plt.show()


# # 전북 및 경남 병원의 20km 버퍼가 전남 지역과 겹치는 부분 분석 및 시각화
# 
# ## 버퍼 거리 설정
# 분석에 사용할 거리 반경을 20km로 설정
# 
# ## 병원 데이터 버퍼 생성
# 모든 병원의 20km 버퍼를 생성
# 
# ## 전남 지역 병원과 다른 지역 병원 분리
# 전남 지역 병원과 다른 지역 병원을 분리
# 
# ## 전북 및 경남 지역 병원만 추출
# 다른 지역 병원 중 전북 및 경남 지역 병원만 추출
# 
# ## 전북 및 경남 지역 병원의 버퍼 생성
# 전북 및 경남 지역 병원의 20km 버퍼를 생성
# 
# ## 전북 및 경남 병원의 버퍼가 전남 지역과 겹치는 부분 찾기
# 전북 및 경남 병원의 버퍼가 전남 지역과 겹치는 부분을 찾음
# 
# ## 겹치는 영역의 전체 면적 계산
# 겹치는 영역의 전체 면적을 계산
# 
# ## 전체 전남 지역의 면적 계산
# 전체 전남 지역의 면적을 계산
# 
# ## 전체 면적 대비 겹치는 면적 비율 계산
# 전체 전남 지역 면적 대비 겹치는 면적의 비율을 계산
# 
# ## 결과 출력
# 계산된 겹치는 면적, 전체 전남 지역 면적, 그리고 면적 비율을 출력
# 
# ## 시각화
# 결과를 시각화
# - 전북 및 경남 병원과 겹치는 전남 지역 부분을 색상으로 표시
# - 병원 버퍼 경계를 파란색으로 표시
# - 전북 및 경남 병원의 위치를 빨간색으로 표시
# - 전체 지도 경계를 점선으로 표시
# - 전남 지역 경계를 연한 산호색으로 표시

# In[27]:


# 버퍼 거리 설정
buffer_distance = 20000  # 20km

# 병원 데이터 버퍼 생성
Honamhospital_buffer = Honamhospital.buffer(buffer_distance)

# 전남 지역 병원과 다른 지역 병원 분리
Jeonam_hospitals = Honamhospital[Honamhospital.within(Jeonam.unary_union)]
non_Jeonam_hospitals = Honamhospital[~Honamhospital.within(Jeonam.unary_union)]

# 전북 및 경남 지역 병원만 추출
Jeonbuk_Gyeongnam_hospitals = non_Jeonam_hospitals[non_Jeonam_hospitals['Province'].isin(['전북', '경남'])]

# 전북 및 경남 지역 병원의 버퍼 생성
Jeonbuk_Gyeongnam_hospitals_buffer = Jeonbuk_Gyeongnam_hospitals.buffer(buffer_distance)

# 전북 및 경남 병원의 버퍼가 전남 지역과 겹치는 부분 찾기
intersection = Jeonbuk_Gyeongnam_hospitals_buffer.intersection(Jeonam.unary_union)

# 겹치는 영역의 전체 면적 계산
intersection_area = intersection.area.sum()

# 전체 전남 지역의 면적 계산
total_jeonam_area = Jeonam.unary_union.area

# 전체 면적 대비 겹치는 면적 비율 계산
intersection_ratio = (intersection_area / total_jeonam_area) * 100

# 결과 출력
print(f'전북 및 경남 병원과 겹치는 면적: {intersection_area} 평방미터')
print(f'전체 전남 지역 면적: {total_jeonam_area} 평방미터')
print(f'전체 면적 대비 겹치는 면적 비율: {intersection_ratio:.2f}%')

# 시각화
fig, ax = plt.subplots(figsize=(7, 10))

# 전북 및 경남 병원과 겹치는 전남 지역 부분을 색상으로 표시
intersection.plot(ax=ax, color='blue', alpha=0.5, zorder=1)

# 병원 버퍼 경계 표시
Honamhospital_buffer.boundary.plot(ax=ax, color='blue', lw=0.5, zorder=2)

# 전북 및 경남 병원 위치 표시
Jeonbuk_Gyeongnam_hospitals.plot(ax=ax, markersize=10, color='red', zorder=3)

# 전체 지도 경계 표시
combined_gdf.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=4)

# 전남 지역 경계 표시
Jeonam.boundary.plot(ax=ax, color='lightcoral', linewidth=0.5, zorder=5)

ax.set_title('Regions within 20km Buffer of Jeonbuk and Gyeongnam Hospitals')
plt.show()


# In[18]:


# 라벨링을 위해 전라남도 도시가 담긴 칼럼명을 알기 위해 Jeonam 칼럼명 확인

Jeonam.head()


# In[20]:


## 코랩 한글깨짐현상 해결깨짐현상 해결

get_ipython().system('sudo apt-get install -y fonts-nanum')
get_ipython().system('sudo fc-cache -fv')
get_ipython().system('rm ~/.cache/matplotlib -rf')


# In[25]:


plt.rc('font', family='NanumBarunGothic')


# # 전남 지역 병원과 다른 지역 병원의 20km 버퍼 분석 및 시각화
# 
# 
# ## 버퍼 거리 설정
# 분석에 사용할 거리 반경을 20km로 설정
# 
# 
# ## 병원 데이터 버퍼 생성
# 모든 병원의 20km 버퍼를 생성
# 
# 
# ## 전남 지역 병원과 다른 지역 병원 분리
# 전남 지역 병원과 다른 지역 병원을 분리
# 
# 
# ## 다른 지역 병원의 버퍼 생성
# 다른 지역 병원의 20km 버퍼를 생성
# 
# 
# ## 전남 지역과 다른 지역 병원의 버퍼가 겹치는 부분 찾기
# 전남 지역과 다른 지역 병원의 버퍼가 겹치는 부분을 찾음
# 
# 
# ## 전남 지역 전체에서 버퍼가 겹치지 않는 부분 찾기
# 전남 지역 내에서 병원 버퍼가 겹치지 않는 부분을 찾음
# 
# 
# ## 시각화
# 결과를 시각화
# - 전남 지역에서 버퍼가 겹치지 않는 부분을 연한 초록색으로 표시
# - 전남 지역과 겹치는 다른 지역 병원 버퍼 부분을 흰색으로 표시
# - 병원 버퍼 경계를 파란색으로 표시
# - 병원 위치를 빨간색으로 표시
# - 전체 지도 경계를 점선으로 표시
# - 전남 지역 경계를 연한 산호색으로 표시
# - 전남 각 군의 라벨을 추가

# In[26]:


buffer_distance = 20000  # 20km

Honamhospital_buffer = Honamhospital.buffer(buffer_distance)

Jeonam_hospitals = Honamhospital[Honamhospital.within(Jeonam.unary_union)]
non_Jeonam_hospitals = Honamhospital[~Honamhospital.within(Jeonam.unary_union)]

non_Jeonam_hospitals_buffer = non_Jeonam_hospitals.buffer(buffer_distance)

intersection = non_Jeonam_hospitals_buffer.intersection(Jeonam.unary_union)

buffered_area_within_jeonam = Jeonam.unary_union.intersection(Honamhospital_buffer.unary_union)
non_buffered_jeonam = Jeonam.unary_union.difference(buffered_area_within_jeonam)

fig, ax = plt.subplots(figsize=(7, 10))
gpd.GeoSeries(non_buffered_jeonam).plot(ax=ax, color='lightgreen', alpha=0.5, zorder=1)
gpd.GeoSeries(intersection.unary_union).plot(ax=ax, color='white', alpha=0.5, zorder=2)
gpd.GeoSeries(Honamhospital_buffer.unary_union).boundary.plot(ax=ax, color='blue', lw=0.5, zorder=3)

Honamhospital.plot(ax=ax, markersize=10, color='red', zorder=4)

combined_gdf.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=5)

Jeonam.boundary.plot(ax=ax, color='lightcoral', linewidth=0.5, zorder=6)

for idx, row in Jeonam.iterrows():
    plt.annotate(text=row['SGG_NM'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                 horizontalalignment='center', fontsize=5, color='black')

ax.set_title('Regions within 20km Buffer of Non-Jeonam Hospitals')
plt.show()


# In[ ]:


# 가임여성인구 수 파악

all_childbearing_age = pd.read_csv('./2022_시군구별 25-39세 분만실 현황.csv')


# In[ ]:


all_childbearing_age


# In[ ]:




