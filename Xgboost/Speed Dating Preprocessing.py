#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


data = pd.read_csv("speed_dating.csv")


# In[37]:


data['interests_correlate']


# In[38]:


data.head()


# In[39]:


data['expected_num_interested_in_me']


# In[40]:


data[data['expected_num_interested_in_me']== -99]


# In[41]:


data.describe()


# In[44]:


data.isna().sum()


# In[45]:


# # 일부 변수에서 결측치 제거
# data = data.dropna(subset=['pref_o_attractive', 'pref_o_ambitious', 'pref_o_shared_interests',
#                            'attractive_important', 'sincere_important', 'intellicence_important',
#                            'funny_important', 'ambtition_important', 'shared_interests_important'])

# 남은 결측치는 -99로 대체
data = data.fillna(-99)


# In[10]:


# 나이 데이터 관련 피처 엔지니어링

def age_gap(x):
    if x['age'] == -99:
        return -99
    elif x['age_o'] == -99:
        return -99
    elif x['gender'] == 'female':
        return x['age_o'] - x['age']
    else:
        return x['age'] - x['age_o']

data['age_gap'] = data.apply(age_gap, axis=1)
# 남녀 중 한 명이라도 나이가 -99이면 -99 반환
# 그렇지 않은 경우 남자가 연상이면 플러스값, 여자가 연상이면 마이너스값 반환

data['age_gap_abs'] = abs(data['age_gap'])
# 어느 쪽 나이가 많은지와 관계 없이 나이 자체가 중요한 변수가 될 수 있으므로
# age_gap 변수에 절댓값을 취한 변수도 추가


# In[ ]:





# In[11]:


# 인종 데이터 관련 피처 엔지니어링
# 본인과 상태방 인종이 같으면 1, 다르면 -1로 처리
# 결측치는 -99를 반환

def same_race(x):
    if x['race'] == -99:
        return -99
    elif x['race_o'] == -99:
        return -99
    elif x['race'] == x['race_o']:
        return 1
    else:
        return -1

data['same_race'] = data.apply(same_race, axis=1)


# In[12]:


# importance_same_race 변수 이용 : 동일 인종 여부가 얼마나 중요한지를 나타냄
# 동일 인종이면 양수, 아니면 음수이며 중요할수록 절댓값이 크게 나타남
# 결측치가 있는 값은 -99 반환

def same_race_point(x):
    if x['same_race'] == -99:
        return -99
    else:
        return x['same_race'] * x['importance_same_race']

data['same_race_point'] = data.apply(same_race_point, axis=1)


# In[13]:


# attractive, sincere 등에 대한 평가/중요도 변수
# 평가 점수 * 중요도로 계산
# 결측치는 -99 반환

def rating(data, importance, score):
    if data[importance] == -99:
        return -99
    elif data[score] == -99:
        return -99
    else:
        return data[importance] * data[score]


# In[14]:


data.columns[9:15]


# In[15]:


data.columns[15:21]


# In[16]:


data.columns[21:27]


# In[17]:


data.columns[27:33]


# In[18]:


partner_imp = data.columns[9:15]
partner_rate_me = data.columns[15:21]
my_imp = data.columns[21:27]
my_rate_partner = data.columns[27:33]


# In[19]:


# 상대방 관련 새 변수 이름을 저장하는 리스트
new_label_partner = ['attractive_p', 'sincere_partner_p', 'intelligence_p',
                     'funny_p', 'ambition_p', 'shared_interests_p']

# 본인 관련 새 변수 이름을 저장하는 리스트
new_label_me = ['attractive_m', 'sincere_partner_m', 'intelligence_m',
                     'funny_m', 'ambition_m', 'shared_interests_m']


# In[20]:


data['interests_correlate']


# In[21]:


for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):
    print(i, " & ", j, " & ", k)


# In[22]:


data['interests_correlate']


# In[23]:


for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)


# In[24]:


data['interests_correlate']


# In[25]:


for i, j, k in zip(new_label_me, my_imp, my_rate_partner):
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)


# In[26]:


data['interests_correlate'] 


# In[27]:


print(data.columns)


# In[28]:


data = pd.get_dummies(data,columns=['gender', 'race', 'race_o'], drop_first=True)
bool_columns = data.select_dtypes(include='bool').columns
data[bool_columns] = data[bool_columns].astype(int)


# In[29]:


data['interests_correlate']


# In[30]:


data


# In[31]:


data['interests_correlate']


# In[32]:


data.to_csv("speed_dating_data1.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




