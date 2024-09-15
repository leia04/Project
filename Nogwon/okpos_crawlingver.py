# -*- coding: utf-8 -*-

##########################사용자 다운로드 폴더 주소 입력###########################

usersdownload_path = 'User_path/'

#############################################################################

# 필요한 패키지 설치
#!pip install selenium
#!pip install webdriver_manager
#!pip install xlrd>=2.0.1

import pandas as pd
import numpy as np
from glob import glob
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.alert import Alert
import time
from datetime import datetime, timedelta
from selenium import webdriver

driver = webdriver.Chrome()

current_date = datetime.now()
days = 7
start_date = current_date - timedelta(days=days)   
yesterday = current_date - timedelta(days= 1)    


date_list = [start_date + timedelta(days=i) for i in range(days)]


year_list = [date.year for date in date_list]
month_list = [date.month for date in date_list]
day_list = [date.day for date in date_list]

date_range_str = f'{start_date.strftime("%Y-%m-%d")}_{yesterday.strftime("%Y-%m-%d")}'

ID = ''
PW = ''


# Chrome 드라이버 옵션 설정
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--start-maximized') 
#chrome_options.add_argument('--headless')  
#chrome_options.add_argument('--no-sandbox')  
chrome_options.add_argument('--disable-dev-shm-usage')  
chrome_options.add_argument("disable-gpu") 
chrome_options.add_argument("lang=ko_KR")  
chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')

driver = webdriver.Chrome(options=chrome_options)
url = 'https://kis.okpos.co.kr/login/login_form.jsp'

#OKpos 사이트로 이동
driver.get(url)
page_source = driver.page_source
time.sleep(2)

# ID 및 PW 입력
id_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,'#user_id')))
id_input.send_keys(ID)

pw_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,'#user_pwd')))
pw_input.send_keys(PW)

#로그인
driver.find_element(By.CSS_SELECTOR, '#loginForm > div:nth-child(4) > div:nth-child(5) > img').click()
time.sleep(3)

#매출관리 클릭
sales_management_button_selector  = '#cswmMenuButtonGroup_15'
sales_management_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sales_management_button_selector )))
sales_management_button.click()


#매출현황 조회
sales_status_button_selector = '#cswmItemGroup_15_6'
sales_status_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sales_status_button_selector)))
sales_status_button.click()


#영수증별매출상세현황 조회
sales_details_receipt_button_selector = '#cswmItem6_24'
sales_details_receipt_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sales_details_receipt_button_selector)))
sales_details_receipt_button.click()
time.sleep(1)

topFrameBody_page_source = driver.page_source

# top <iframe> 요소에 접근
iframe_element = driver.find_element(By.ID, 'MainFrm')
# 해당 <iframe>으로 전환
driver.switch_to.frame(iframe_element)
iframe_element_page_source = driver.page_source


formatted_date_list = [date.strftime("%Y-%m-%d") for date in date_list]
for i in range(len(date_list)):
    # day_list의 첫 번째 값으로 value 속성 값을 변경
    new_value = formatted_date_list[i]
    input_element_id = 'date1'
    script = f"document.getElementById('{input_element_id}').value = '{new_value}';"
    driver.execute_script(script)

    # 조회 클릭
    button_element = driver.find_element(By.XPATH, '//*[@id="form1"]/div/div[1]/div[4]/button[1]')
    button_element.click()
    time.sleep(1)
    #엑셀 클릭
    button_element = driver.find_element(By.XPATH, '//*[@id="form1"]/div/div[1]/div[4]/button[2]')
    button_element.click()
    time.sleep(1)

    try:
        alert = WebDriverWait(driver,1).until(EC.alert_is_present()) 
        alert.accept()
        print(f"{new_value}에는 출력할 자료가 없습니다.")
    except:
        pass

driver.quit()

####파이썬 시간과 날짜 엑셀 형식으로 변환하는 함수 ####
def convert_to_excel_time(time_str):
    time_format = "%H:%M:%S"
    time_dt = datetime.strptime(time_str, time_format)

    excel_base_date = datetime(1900, 1, 1)
    excel_time = (time_dt - excel_base_date).total_seconds() / (24 * 60 * 60)
    return excel_time

def convert_to_excel_date(dt):
    excel_base_date = datetime(1900, 1, 1)
    excel_days = (dt - excel_base_date).days + 2

    return excel_days

#전처리 함수
def preprocess_df(df):

    date_hint = df.iloc[1,0].split('조회일자 : ')[1].strip()
    df = df.drop([0, 2, 3]).reset_index(drop=True)
    updated_value = df.iloc[0, 0].replace('조회일자 : ', '')
    df.iloc[:, 0] = updated_value
    df = df.drop(0).reset_index(drop=True)
    df.iloc[0, 0] = '조회일자'
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)
    df = df.iloc[:-1, :]

    # '조회일자'를 기준으로 빠른 날짜부터 정렬하고, 조회일자가 같은 경우 '영수증번호'를 기준으로 정렬
    df = df.sort_values(by=['조회일자', '영수증번호']).reset_index(drop=True)
    # 'HOT', 'ICE', 'TOGO' 문자열 삭제 및 맨 끝 공백 제거
    df['상품명'] = df['상품명'].str.replace('HOT|ICED|ICE|TOGO', '', regex=True).str.rstrip()
    # '청귤 캐모마일' 또는 '청귤캐모마일'인 경우 '청캐'로 변경
    df['상품명'] = df['상품명'].replace(['청귤 캐모마일', '청귤캐모마일'], '청캐', regex=True)

    drop_clos = ['조회일자','테이블명','최초주문',
                 '상품코드','바코드', 'ERP 매핑코드',
                 '비고', '할인구분', '할인액', '실매출액',
                 '가액', '부가세'
                ]

    df = df.drop(drop_clos, axis=1)

    date_object = datetime.strptime(date_hint, '%Y-%m-%d')

    weekday = date_object.weekday()

    weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    day_of_week = weekday_names[weekday]

    df.rename(columns={'영수증번호': '주문번호'}, inplace=True)
    df['주문번호'] = df['주문번호'].astype(int)
    df.insert(0, '조회요일', day_of_week)
    df.insert(0, '조회일자', date_hint)
    df.insert(0, '결제채널', '오케이포스')

    df['결제시각'] = df['결제시각'].apply(convert_to_excel_time)

    df['조회일자'] = pd.to_datetime(df['조회일자'])  # '조회일자' 열의 값을 datetime 객체로 변환
    df['조회일자'] = df['조회일자'].apply(convert_to_excel_date)
    return df

#다운 받은 엑셀 파일 모두 읽어 전처리 후 하나의 엑셀파일로 변환
Downloads_pafh = usersdownload_path #사용자의 다운로드 파일 경로로 재설정 필요
folder_path = f"{Downloads_pafh}Crawling_data" # 최종 엑셀파일이 모일 폴더 경로 지정

os.makedirs(folder_path, exist_ok=True)


file_pattern = '*영수증별매출상세현황*'
file_paths_list = glob(os.path.join(Downloads_pafh, file_pattern))
file_paths_list = sorted(file_paths_list, key=os.path.getmtime, reverse=False) #가장 최근 날짜를 먼저 읽을 수 있게 정렬
folder_dfs = []


for file_paths in file_paths_list:
    df = pd.read_excel(file_paths)
    processed_df = preprocess_df(df)
    folder_dfs.append(processed_df)

combined_folder_df = pd.concat(folder_dfs, ignore_index=True)

# 디렉토리 경로와 파일 이름 결합
file_path = os.path.join(folder_path, f'okpos_{date_range_str}_data.xlsx')

# DataFrame을 지정한 경로에 엑셀로 저장
combined_folder_df.to_excel(file_path, index=False)

# 다운 받았던 일별 rowdata 삭제하기
files_to_delete = glob(os.path.join(Downloads_pafh, file_pattern))
for file_to_delete in files_to_delete:
     os.remove(file_to_delete)

print('==============================수집된 데이터의 결과 예시 입니다.=======================')
print()
combined_folder_df.head()
