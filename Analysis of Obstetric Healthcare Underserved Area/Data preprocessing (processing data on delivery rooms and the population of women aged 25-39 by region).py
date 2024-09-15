#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# # 시도별 25-39세가임인구 및 분만실 데이터 가공
# - 20222 인구데이터 부른다음 geo_hospital 데이터에 붙이기
# - 분만실을 1개 이상 가진 의원 및 병원 병합 후 중복값 제거
# - 데이터 값 시도군별로 표현하기 위해 매핑
# - 데이터프레임의 인덱스를 매핑된 값으로 변환 (같은 시도명으로 통일하기 위)
# 

# In[40]:


sido_sigungo_pop = pd.read_csv('/content/2022_시군구별_가임여성인구_재재재가공.csv', encoding='cp949')
sido_sigungo_pop.head(2)


# In[12]:


hospital1 = pd.read_excel("/content/drive/MyDrive/2024수업/1학기/데이터액티비즘/data/1.병원정보서비스 2022.12.xlsx")
hospital2 = pd.read_excel("/content/drive/MyDrive/2024수업/1학기/데이터액티비즘/data/3.의료기관별상세정보서비스_01_시설정보_2022.12.xlsx")
hospital3 = hospital1.merge(hospital2, how='outer',
                            on=['암호화요양기호', '요양기관명', '종별코드','종별코드명','시도코드', '시도코드명', '시군구코드', '시군구코드명', '읍면동','주소'])
print(hospital3.shape)


# In[13]:


hospital4 = hospital3[['요양기관명','시도코드', '시도코드명', '시군구코드', '시군구코드명','주소','총의사수','조산사 인원수','일반입원실상급병상수','일반입원실일반병상수','분만실병상수','수술실병상수','좌표(X)', '좌표(Y)']]
hospital6 = hospital4[(hospital4['분만실병상수']>=1)]
geo_hosputal = gpd.GeoDataFrame(hospital6, geometry=gpd.points_from_xy(hospital6['좌표(X)'], hospital6['좌표(Y)']))


# In[14]:


geo_hosputal


# In[41]:


mapping = {
    '강원도': '강원',
    '세종특별자치시': '세종시',
    '충청남도': '충남',
    '제주특별자치도': '제주',
    '전라북도': '전북',
    '전라남도': '전남',
    '인천광역시': '인천',
    '울산광역시': '울산',
    '서울특별시': '서울',
    '경기도': '경기',
    '부산광역시': '부산',
    '대전광역시': '대전',
    '대구광역시': '대구',
    '광주광역시': '광주',
    '경상북도': '경북',
    '경상남도': '경남',
    '충청북도': '충북'
}
sido_sigungo_pop.loc[:, '시도'] = sido_sigungo_pop['시도'].map(mapping)


# In[42]:


sido_25to39pop = sido_sigungo_pop.groupby(['시도'])['가임여성인구수(25-39세)'].sum().reset_index().copy()
sido_25to39pop


# In[43]:


sido_hospital = geo_hosputal.groupby(['시도코드명'])['분만실병상수'].sum().reset_index().copy()
sido_hospital


# In[44]:


merged_df = pd.merge(sido_hospital, sido_25to39pop, left_on='시도코드명', right_on='시도', how='left').copy()
sido_merged_df = merged_df[['시도코드명', '분만실병상수', '가임여성인구수(25-39세)']]
sido_merged_df


# In[ ]:


sido_merged_df.to_csv('/content/2022_시도별_분만실_25_39세인구.csv')


# # 시군구별 25-39세 가임인구 및 분만실 데이터 가공

# In[45]:


sido_sigungo_pop.head()


# In[47]:


sigungo_25_39pop = sido_sigungo_pop[['행정구역(시군구)별', '가임여성인구수(25-39세)']].copy()
sigungo_25_39pop


# In[48]:


sigungo_hospital = geo_hosputal.groupby(['시군구코드명'])['분만실병상수'].sum().reset_index().copy()
sigungo_hospital


# In[49]:


merged_df = pd.merge(sigungo_hospital, sigungo_25_39pop, left_on='시군구코드명', right_on='행정구역(시군구)별', how='left').copy()
sigungo_merged_df = merged_df[['시군구코드명', '분만실병상수', '가임여성인구수(25-39세)']]
sigungo_merged_df


# In[50]:


sido_merged_df.to_csv('/content/2022_시군구별_분만실_25_39세인구.csv')

