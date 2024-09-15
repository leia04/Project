#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import pandas as pd
import osmnx as ox
import time
import numpy as np
from tqdm import tqdm, trange
from shapely.geometry import Point, MultiPoint
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


# # 전남의 취약 시군구 분석을 위한 2SFCA
# 

# - 엑셀 파일 경로지정 후 쉐이프 파일을 저장할 디렉토리와 파일 경로 지정
# - 지오메트리를 생성하고 좌표계 설정 후 파일 저장 

# In[2]:


import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

file_path = 'C:/Users/USER/Desktop/파이썬 팀플/전남+주변 산부인과.xlsx'
output_dir = 'C:/Users/USER/Desktop/파이썬 팀플'
shapefile_path = os.path.join(output_dir, '전남+주변 산부인과.shp')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 엑셀 파일 읽기
data = pd.read_excel(file_path)

# 컬럼명 변환 (한글 그대로 유지, ASCII 문자로 변환)
column_mapping = {
    '요양기관명': 'Institution_Name',
    '종별코드': 'Type_Code',
    '종별코드명': 'Type_Code_Name',
    '시도코드명': 'Province',
    '시군구코드명': 'City_District',
    '주소': 'Address',
    '총의사수': 'Total_Doctors',
    '조산사 인원수': 'Midwives_Count',
    '일반입원실상급병상수': 'General_Room_Premium_Beds',
    '일반입원실일반병상수': 'General_Room_Standard_Beds',
    '분만실병상수': 'Delivery_Room_Beds',
    '수술실병상수': 'Surgery_Room_Beds',
    '좌표(X)': 'Longitude',
    '좌표(Y)': 'Latitude'
}

# 컬럼명 변경
data_renamed = data.rename(columns=column_mapping)

# 지오메트리 생성
geometry = [Point(xy) for xy in zip(data_renamed['Longitude'], data_renamed['Latitude'])]
geo_df = gpd.GeoDataFrame(data_renamed, geometry=geometry)

# 좌표계 설정 (WGS84)
geo_df.set_crs(epsg=4326, inplace=True)

# 쉐이프파일로 내보내기 (UTF-8 인코딩 설정)
geo_df.to_file(shapefile_path, driver='ESRI Shapefile', encoding='utf-8')

print(f'쉐이프파일이 {shapefile_path}에 저장되었습니다.')



# In[3]:


hospitals = gpd.read_file('./전남+주변 산부인과/전남+주변 산부인과.shp')


# In[4]:


hospitals[hospitals['Province']=='전북'].head()


# In[5]:


si = gpd.read_file('./호남권/전남.shp')
si = si.to_crs("epsg:4326")
si.head()


# - SHP 파일과 XLSX 파일 읽기
# - XLSX 파일의 열 이름 변경 (시군구 -> SGG_NM)하여 조인 준비
# - 필요한 열만 선택 (SGG_NM과 female_population)
# - 조인을 위해 키 값 설정
# - 인코딩을 UTF-8로 설정하여 SHP 파일로 저장

# In[6]:


csv_df = pd.read_excel("./시군구별 25-39세 가임여성인구 전남가공.xlsx")

csv_df.rename(columns={'행정구역(시군구)별': 'SGG_NM', '가임여성인구수(25-39세)': 'female_population'}, inplace=True)

csv_df = csv_df[['SGG_NM', 'female_population']]

merged_df = si.set_index('SGG_NM').join(csv_df.set_index('SGG_NM'))

merged_df.reset_index(inplace=True)  # 인덱스를 다시 열로 변환
merged_df.to_file('전남 25-39세 여성.shp', encoding='utf-8')

# 결과 확인 (가임여성인구수(15-49세) 컬럼이 추가된 것을 확인)
print(merged_df.head())


# In[6]:


jn = gpd.read_file('./전남 25-39세 여성.shp')
jn.head()


# - 전남과 전남의 경계 지역에 포함되는 곳에서 분만실 병상수가 있는 병원들의 포인트를 나타냄

# In[7]:


ax = jn.plot(figsize=(10, 10),color='white', edgecolor='black')
hospitals.plot(ax=ax, color='red', markersize=10)


# In[8]:


places = ['목포시, 대한민국',
          '여수시, 대한민국',
          '순천시, 대한민국',
          '나주시, 대한민국',
          '광양시, 대한민국',
          '담양군, 대한민국',
          '곡성군, 대한민국',
          '구례군, 대한민국',
          '고흥군, 대한민국',
          '보성군, 대한민국',
          '화순군, 대한민국',
          '장흥군, 대한민국',
          '강진군, 대한민국',
          '해남군, 대한민국',
          '영암군, 대한민국',
          '무안군, 대한민국',
          '함평군, 대한민국',
          '영광군, 대한민국',
          '장성군, 대한민국',
          '완도군, 대한민국',
          '진도군, 대한민국',
          '신안군, 대한민국',
          '고창군, 대한민국',
          '정읍시, 대한민국',
          '남원시, 대한민국',
          '순창군, 대한민국',
          '하동군, 대한민국',
          '동구, 광주광역시, 대한민국',
          '서구, 광주광역시, 대한민국',
          '남구, 광주광역시, 대한민국',
          '북구, 광주광역시, 대한민국',
          '광산구, 광주광역시, 대한민국',]


# In[9]:


G = ox.graph_from_place(places, network_type='drive')


# In[10]:


ox.plot_graph(G)


# In[11]:


G.remove_nodes_from(list(nx.isolates(G)))


# # OSM (OpenStreetMap) 네트워크에서 가장 가까운 노드 찾기 함수
# 
# ## 함수 정의
# - `find_nearest_osm(network, gdf)` 함수는 지정된 지오데이터프레임(gdf) 내 각 지점에 대해 OSM 네트워크에서 가장 가까운 노드를 찾음
# 
# ### 매개변수
# - `network`: OSM 네트워크 그래프 객체
# - `gdf`: 지오데이터프레임, 지오메트리 정보를 포함
# 
# ### 동작
# 1. 지오데이터프레임의 각 행을 반복 처리
# 2. 지오메트리 타입에 따라 가장 가까운 OSM 노드를 찾음
# - `Point` 타입: 지점의 x, y 좌표를 사용하여 가장 가까운 노드를 찾음
# - `Polygon` 또는 `MultiPolygon` 타입: 폴리곤의 중심점(centroid)의 x, y 좌표를 사용하여 가장 가까운 노드를 찾음
# - 다른 지오메트리 타입은 무시
# 3. 찾은 가장 가까운 OSM 노드를 `nearest_osm` 컬럼에 저장
# 4. 업데이트된 지오데이터프레임을 반환

# In[12]:


def find_nearest_osm(network, gdf):
    for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        if row.geometry.geom_type == 'Point':
            nearest_osm = ox.distance.nearest_nodes(network, 
                                                    X=row.geometry.x, 
                                                    Y=row.geometry.y
                                                   )
        elif row.geometry.geom_type == 'Polygon' or row.geometry.geom_type == 'MultiPolygon':
            nearest_osm = ox.distance.nearest_nodes(network, 
                                        X=row.geometry.centroid.x, 
                                        Y=row.geometry.centroid.y
                                       )
        else:
            print(row.geometry.geom_type)
            continue

        gdf.at[idx, 'nearest_osm'] = nearest_osm

    return gdf

supply = find_nearest_osm(G, hospitals)
demand = find_nearest_osm(G, jn)


# In[13]:


nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True)
nodes.head(2)


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# # 분만실 병상수 공급 분석
# 
# ## 거리 설정
# 분석에 사용할 거리 반경을 20km로 설정
# 
# ## 데이터 복사 및 초기화
# `supply` 데이터프레임을 깊은 복사하여 `supply_1`을 생성하고, `A` 컬럼을 0으로 초기화
# 
# ## 공급 분석 루프
# 공급 분석을 위한 루프를 수행
# - 주어진 위치로부터 접근 가능한 노드를 찾음
# - 접근 가능한 노드의 지오메트리를 사용해 컨벡스 헐(볼록 껍질)을 계산
# - 접근 가능 영역 내에 있는 인구수를 계산
# - 각 병원의 분만실 병상 수를 계산
# - 각 병원의 분만실 병상 수를 `A` 컬럼에 저장

# In[24]:


dist = 20000 # 20km

supply_1 = supply.copy(deep=True)
supply_1['A'] = 0

for i in tqdm(range(supply.shape[0])):
    
    temp_nodes = nx.single_source_dijkstra_path_length(G, supply.loc[i, 'nearest_osm'], dist, weight='length')
    access_nodes = nodes.loc[nodes.index.isin(temp_nodes.keys()), 'geometry']
    access_nodes = gpd.GeoSeries(access_nodes.unary_union.convex_hull, crs="EPSG:4326")
    
    temp_demand = demand.loc[demand['geometry'].centroid.within(access_nodes[0]), 'female_pop'].sum()
    temp_supply = supply.loc[i, 'Delivery_R']
    
    supply_1.at[i, 'A'] = temp_supply 
    


# # 분만실 병상수 공급 분석 결과 시각화
# 
# ## 지도 시각화 설정
# 20x20 크기의 그래프를 설정
# 
# ## 전남지역 여성 인구 밀도 시각화
# `female_pop` 컬럼을 기준으로 Fisher-Jenks 방법을 사용해 색상으로 표시
# 
# ## 지도 경계 및 특정 지역 강조
# - 지도 경계를 점선으로 표시
# - 특정 지역(여기서는 인덱스가 10인 지역)을 빨간색으로 강조
# 
# ## 접근 가능 지역 표시
# - 특정 위치(여기서는 인덱스가 10인 지역)로부터 접근 가능한 노드를 찾음
# - 접근 가능한 노드의 지오메트리를 사용해 컨벡스 헐(볼록 껍질)을 계산하고, 이를 녹색 경계선으로 표시
# 
# ## 접근 가능한 병원과 접근 불가능한 병원 시각화
# - 접근 가능한 병원을 검은색으로, 접근 불가능한 병원을 회색으로 표시
# - 병상의 수에 비례해 마커의 크기를 조절

# In[114]:


fig, ax = plt.subplots(figsize=(20, 20))

jn.plot('female_pop', ax=ax, scheme='FisherJenks', cmap='Greens', legend=True)
jn.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=1)
jn.loc[jn.index==10].plot(ax=ax, color='red')

# Plot catchment area
temp_nodes = nx.single_source_dijkstra_path_length(G, demand.loc[10, 'nearest_osm'], dist, weight='length')
access_nodes = nodes.loc[nodes.index.isin(temp_nodes.keys()), 'geometry']
access_nodes = gpd.GeoSeries(access_nodes.unary_union.convex_hull, crs="EPSG:4326")
access_nodes.boundary.plot(ax=ax, color='green', linewidth=1)

# Calculate accessible hospital from a dong
access_supply_1 = supply_1.loc[supply_1['geometry'].centroid.within(access_nodes[0])]
non_access_supply_1 = supply_1.loc[~supply_1['geometry'].centroid.within(access_nodes[0])]

access_supply_1.plot(ax=ax, markersize=access_supply_1['A']*7, color='black')
non_access_supply_1.plot(ax=ax, markersize=non_access_supply_1['A']*7, color='grey')


# # 접근 가능한 분만실 병상수 분석
# 
# ## 데이터 복사 및 초기화
# `demand` 데이터프레임을 깊은 복사하여 `demand_1`을 생성하고, `access_1` 컬럼을 0으로 초기화
# 
# ## 접근 가능한 병상수 계산 루프
# 각 수요 지점에 대해 접근 가능한 병상수를 계산
# - 주어진 위치로부터 접근 가능한 노드를 찾음
# - 접근 가능한 노드의 지오메트리를 사용해 컨벡스 헐(볼록 껍질)을 계산
# - 접근 가능 영역 내에 있는 병원의 병상수를 누적 계산
# - 누적 계산된 병상수를 `access_1` 컬럼에 저장
# 

# In[26]:


demand_1 = demand.copy(deep=True)
demand_1['access_1'] = 0

for j in trange(demand.shape[0]):
    temp_nodes = nx.single_source_dijkstra_path_length(G, demand.loc[j, 'nearest_osm'], dist, weight='length')
    access_nodes = nodes.loc[nodes.index.isin(temp_nodes.keys()), 'geometry']
    access_nodes = gpd.GeoSeries(access_nodes.unary_union.convex_hull, crs="EPSG:4326")
    
    accum_ratio_1 = supply_1.loc[supply_1['geometry'].within(access_nodes[0]), 'A']
    accum_ratio_1 = accum_ratio_1.replace([np.inf, -np.inf], np.nan).dropna(axis=0).sum()
    demand_1.at[j, 'access_1'] = accum_ratio_1
    
demand_1


# In[27]:


min_access_1 = demand_1['access_1'].min()
max_access_1 = demand_1['access_1'].max()
demand_1['norm_1'] = (demand_1['access_1']-min_access_1)/(max_access_1-min_access_1)


# In[125]:


demand_1


# In[36]:


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# # 전남 시군구별 분만실 접근성 시각화
# 
# ## 지도 시각화 설정
# 20x20 크기의 그래프를 설정
# 
# ## 전남 시군구별 분만실 접근성 시각화
# - `norm_1` 컬럼을 기준으로 Fisher-Jenks 방법을 사용해 색상으로 표시
# - 지도 경계를 점선으로 표시
# - 그래프 제목을 설정
# 
# ## 각 시군구 라벨링
# - 시군구 이름을 지도에 표시
# - 접근성 값에 따라 라벨의 색상을 흰색 또는 검은색으로 설정
# 
# ## 결과
# - 접근성이 0값인 시군구가 9개 -> 가임여성인구가 많은 순으로 나주시, 영암군, 완도군, 보성군, 진도군, 신안군, 함평군, 곡성군, 구례군 -> 똑같이 나옴
# - 접근성이 의미있게 높은 지역은 장성군(1), 담양군(0.764), 화순군(0.705). 이후 중간층은 0.52~0.59로 비슷비슷함

# In[126]:


fig, ax = plt.subplots(figsize=(20,20))

dㅁemand_1.plot('norm_1', ax=ax, figsize=(10,10), legend=True, cmap='Blues', scheme='FisherJenks')
demand_1.boundary.plot(ax=ax, linestyle='dotted', lw=0.5, color='black', zorder=1)
ax.set_title("전남 시군구별 분만실 접근성", fontsize=25, pad=20)


demand_1.apply(lambda x: ax.annotate(text=x['SGG_NM'].replace('전라남도 ', ''), xy=x.geometry.centroid.coords[0],ha='center', fontweight='bold',
                                     fontsize=13, color='white' if x['norm_1'] > 0.59 else 'black'),
               axis=1)

plt.show()


