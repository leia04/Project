#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # 출산율 대비 분만실 개수 분석
# - 시군구별 및 연령별 출산율데이터와 전처리 작업을 한 분만실 병상수 데이터를 사용
# - 산부인과와 분만실을 가진 의원 및 병원 병합 후 중복값 제거
# - "주소" 열을 띄어쓰기를 기준으로 슬라이싱하여 시도와 행정구역 추출
# - 행정구역별 분만실 병상수 합계 계산(좌표없이)
# - 비율이 1보다 작은 것을 분만 취약지로 선정
# - 전남이 4곳으로 가장 많았음

# In[2]:


new = pd.read_csv("시도_합계출산율__모의_연령별_출산율_20240610010220.csv", encoding='utf8', header=1)


# In[3]:


df = pd.read_csv("시군구별 및 연령별 출산율_2022.csv", encoding='cp949',header=1)
delivery_room = pd.read_csv("분만실병상수_2022.csv", encoding='cp949')


# In[4]:


delivery_room.head()


# In[5]:


df.head()


# In[6]:


df_b = df.iloc[:,:2]
df_b


# In[7]:


delivery_room.head()


# In[8]:


delivery_room['시도'] = delivery_room['주소'].apply(lambda x: x.split()[0])
delivery_room['행정구역'] = delivery_room['주소'].apply(lambda x: x.split()[1])


# In[9]:


new_delivery_room_grouped = delivery_room.groupby(['시도', '행정구역'])['분만실병상수'].sum().reset_index()
new_delivery_room_grouped


# In[10]:


new_merged_df = pd.merge(new_delivery_room_grouped, df_b, left_on='행정구역', right_on='행정구역별', how='inner')
new_merged_df = new_merged_df.drop(columns=['행정구역별'])


# In[11]:


new_merged_df['출산율 대비 분만실 병상수'] = new_merged_df['분만실병상수'] / new_merged_df['합계출산율 (가임여성 1명당 명)']
new_merged_df = new_merged_df.sort_values(by='출산율 대비 분만실 병상수', ascending=True)
new_merged_df


# In[12]:


new_merged_df.describe()


# In[13]:


new_under_1 = new_merged_df[new_merged_df['출산율 대비 분만실 병상수'] < 1]
new_under_1


# In[14]:


sido_counts = new_under_1.groupby('시도').size().reset_index(name='count')
sido_counts



# In[ ]:




