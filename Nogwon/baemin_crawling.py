# -*- coding: utf-8 -*-

usersdownload_path = 'User_path'

# 필요한 패키지 설치
#!pip install selenium
#!pip install webdriver_manager

import os
import shutil
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import requests
import time
from datetime import datetime, timedelta
from selenium import webdriver

### 배달의 민족 데이터 크롤링 함수 ###
def baemin_crawler():

    driver = webdriver.Chrome()
    current_date = datetime.now()
    days = 7 
    start_date = current_date - timedelta(days=days)  
    yesterday = current_date - timedelta(days= 1)  
    date_range_str = f'{start_date.strftime("%Y-%m-%d")}_{yesterday.strftime("%Y-%m-%d")}'  

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--start-maximized') 
    #chrome_options.add_argument('--headless') 
    #chrome_options.add_argument('--no-sandbox')  
    chrome_options.add_argument('--disable-dev-shm-usage') 
    chrome_options.add_argument("disable-gpu") 
    chrome_options.add_argument("lang=ko_KR")  
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')

    driver = webdriver.Chrome(options=chrome_options)

    url = 'https://self.baemin.com/orders/history'

    ID = ''
    PW = ''

    driver.get(url)
    page_source = driver.page_source

    id_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,'#root > div.style__LoginWrap-sc-145yrm0-0.hKiYRl > div > div > form > div:nth-child(1) > span > input[type=text]')
    ))
    id_input.send_keys(ID)

    pw_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,'#root > div.style__LoginWrap-sc-145yrm0-0.hKiYRl > div > div > form > div.Input__InputWrap-sc-tapcpf-1.kjWnKT.mt-half-3 > span > input[type=password]')))
    pw_input.send_keys(PW)

    
    driver.find_element(By.CSS_SELECTOR, '#root > div.style__LoginWrap-sc-145yrm0-0.hKiYRl > div > div > form > button').click()

    time.sleep(5)

    #팝업 끄기(일시적인 팝업 같아서 try except로)
    #팝업은 종종 바뀌는 것 같아서 지속적인 모니터링과 수정이 필요
    try:
        X_button_selector = '/html/body/div[4]/div/section/header/div/button/div'
        X_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, X_button_selector))
        )
        X_button.click()
    except:
        pass

    time.sleep(2)

    duration_button_selector = '#root > div > div.frame-container > div.frame-wrap > div.frame-body > div.OrderHistoryPage-module__R0bB > div.FilterContainer-module___Rxt > button.FilterContainer-module__vSPY.FilterContainer-module__vOLM'
    duration_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, duration_button_selector))
    )
    duration_button.click()


    time.sleep(2)


    #일.주 메뉴 선택하기
    h_button_selector = '/html/body/div[4]/div/section/div/div[1]/div[1]/label[1]/span'
    h_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()

    time.sleep(1)

    #지난7일 클릭
    h_button_selector = '/html/body/div[4]/div/section/div/div[1]/div[2]/div/label[3]'
    h_button = WebDriverWait(driver, 10).until(
       EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()
    time.sleep(1)

    ##################################################################################################################
    ################################분기 메뉴 선택 할때 주석을 풀어서 사용#############################################
    ##################################################################################################################
    '''

    #분기 메뉴 선택하기
    h_button_selector = ' /html/body/div[4]/div/section/div/div[1]/div[1]/label[3]/span'
    h_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()

    time.sleep(1)

    #분기 선택 창 클릭 주소
    h_button_selector = '/html/body/div[4]/div/section/div/div[1]/div[2]/div'
    h_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()
    time.sleep(1)

    #2023년 분기 클릭
    h_button_selector = '/html/body/div[4]/div/section/div/div[1]/div[2]/div/select/option[2]'
    h_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()
    time.sleep(1)
    '''
    ##################################################################################################################
    ################################분기 메뉴 선택 할때 주석을 풀어서 사용#############################################
    ##################################################################################################################

    #선택한 기간 적용하기
    h_button_selector = '/html/body/div[4]/div/section/footer/button/div'
    h_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, h_button_selector))
    )
    h_button.click()
    time.sleep(1)


    # 조회할 데이터가 있으면 크롤링, 없으면 메시지 출력.
    order_dict = {
        '주문시간': [],
        '주문번호': [],
        '배달유형' : [],
        '주문상품': [],
        '매출': []
    }

    check = driver.find_elements(By.CSS_SELECTOR, 'div.frame-container div.frame-wrap div.frame-body div.OrderHistoryPage-module__R0bB div.Contents-module__b1bl button'
    )

    if check:

        while True:

            #모두 펼쳐보기 클릭(상품명, 수량, 상품별 가격은 상세내역에 있어서 펼쳐보기 후 크롤링 가능)
            detail_button_selector = 'div.frame-container div.frame-wrap div.frame-body div.OrderHistoryPage-module__R0bB div.Contents-module__b1bl button'

            detail_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, detail_button_selector))
            )
            detail_button.click()

            #각 주문요약 가져오고, 합치기(주문시간, 주문번호, 건별 매출액은 주문요약에 있음)
            a = driver.find_element(By.CLASS_NAME, "css-ibrff7")
            boxes = driver.find_elements(By.CLASS_NAME, "css-qequj0")
            boxes.insert(0, a)
            time.sleep(2)
            details = driver.find_elements(By.CLASS_NAME, "css-1vmeggj")

            #주문번호, 주문시각 가져오기
            for box, detail in zip(boxes, details):
                times = box.find_element(By.CLASS_NAME, "DesktopVersion-module__MRUd")
                num = box.find_element(By.CLASS_NAME, "DesktopVersion-module__AHoo")


                # 배달유형 가져오기 == td[4]에 해당하는 열의 정보 가져오기
                td_4_elements = box.find_elements(By.XPATH, ".//td[4]")

                # td[4]에 해당하는 열이 있는지 확인하고 정보 추출
                if len(td_4_elements) > 0:
                    delivery_type = td_4_elements[0]
                else:
                    delivery_type = "배달 유형을 찾을 수 없음"


                #상품명, 수량, 배달유형 추출하기, 상품별 가격 포함
                prods = detail.find_elements(By.CLASS_NAME, "DetailInfo-module__pC_2.FieldItem-module__gYJs")
                for prod in prods:
                    product = prod.find_element(By.CLASS_NAME, "DetailInfo-module__nV94").text
                    price = prod.find_element(By.CLASS_NAME, "FieldItem-module__rb57").text

                    order_dict['주문시간'].append(times.text)
                    order_dict['주문번호'].append(num.text)
                    order_dict['배달유형'].append(delivery_type.text)
                    order_dict['주문상품'].append(product)
                    order_dict['매출'].append(price)


            try:
                np_button_selector = '*[aria-label="다음 페이지로 이동"] .button-overlay.css-fowwyy'
                np_button = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, np_button_selector))
                )
                np_button.click()
            except:
                break
            time.sleep(2)

        #현재 작업 디랙토리에 csv파일로 저장
        df = pd.DataFrame(order_dict)

        csv_filename = f'baemin_{date_range_str}_rowdata.csv'
        df.to_csv(csv_filename, index=False, encoding='cp949')

        destination_directory = usersdownload_path
        destination_path = os.path.join(destination_directory, csv_filename)
        shutil.move(csv_filename, destination_path)

    else:
        print(f"{date_range_str}에 해당하는 데이터가 없습니다. 크롤링을 스킵합니다.")

    # 크롤링이 끝나면 브라우저 닫기
    driver.close()

####파이썬 시간과 날짜 엑셀 형식으로 변환하는 함수 ####

def convert_to_excel_time(time_str):
    time_format = "%H:%M:%S"
    time_dt = datetime.strptime(time_str, time_format)

    if time_dt.hour < 12:
        time_dt = time_dt + timedelta(hours=12)

    excel_base_date = datetime(1900, 1, 1)
    excel_time = (time_dt - excel_base_date).total_seconds() / (24 * 60 * 60)

    return excel_time


def convert_to_excel_date(dt):
    excel_base_date = datetime(1900, 1, 1)
    excel_days = (dt - excel_base_date).days + 2 #파이썬은 19899년 12월 30일을 기준으로 하기 때문에 엑셀 기준에 맞게 +2로 보정

    return excel_days

#### 크롤링 데이터 전처리 하는 함수 ####
def preprocess_data():

    # 크롤링된 csv파일 경로 지정
    current_date = datetime.now()                   
    start_date = current_date - timedelta(days=7)   
    yesterday = current_date - timedelta(days=1)   

    # 7일 기간 계산
    start_date = yesterday - timedelta(days=6)
    date_range_str = f'{start_date.strftime("%Y-%m-%d")}_{yesterday.strftime("%Y-%m-%d")}'
    file_name = f'baemin_{date_range_str}_rowdata.csv'
    file_path = f'{usersdownload_path}{file_name}'


    # 현재 디랙토리에 저장된 CSV 파일 읽어오기
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='cp949')

    else:
        print(f"파일 '{file_path}'을 찾을 수 없습니다.")
        print('BACS 학회장에게 연락해주세요.')
        return

    if df.empty:
        print("업데이트할 데이터가 없습니다.")
        return pd.DataFrame()  

    df[['조회일자', '조회요일', '결제시각']] = df['주문시간'].str.split('. \(|\n', expand=True)
    df.drop('주문시간', axis=1, inplace=True)
    df[['구분', '주문번호']] = df['주문번호'].str.split('\n', expand=True)
    df['수량'] = df['주문상품'].str.extract(r'(\d+개?$)')
    df['상품명'] = df['주문상품'].str.replace(r'\d+개?$', '', regex=True).str.strip()

    # df.loc[df['상품명'] == '쑥 꼬숩이(쑥 라떼)', '상품명'] = '쑥 꼬숩이 (쑥 라떼)'
    # df.loc[df['상품명'] == '흑임자 꼬숩이', '상품명'] = '흑임자 꼬숩이(흑임자 라떼)'
    df.drop('주문상품', axis=1, inplace=True)
    df.rename(columns={'매출': '총매출액'}, inplace=True)
    

    df = df[['조회일자', '조회요일', '주문번호', '배달유형', '결제시각', '상품명', '수량', '총매출액']]
    df.rename(columns={'배달유형': '구분'}, inplace=True)

    df.loc[:, '조회요일'] = df['조회요일'].str.replace('\)', '요일', regex=True)
    df.loc[:, '수량'] = df['수량'].str.replace(r'\D', '', regex=True)
    df.loc[:, '총매출액'] = df['총매출액'].str.replace(r'\D', '', regex=True)

    df['수량'] = pd.to_numeric(df['수량'], errors='coerce')  
    df['총매출액'] = pd.to_numeric(df['총매출액'], errors='coerce')
    df['조회일자'] = pd.to_datetime(df['조회일자'], format='%Y. %m. %d')
    df['결제시각'] = df['결제시각'].str.extract(r'.*오후(.*)')[0].str.strip()       #배민의 경우 오픈 시간이 오후12:00 ~ 09:30으로 설정되어 있음. 오픈 시간 변경으로 '오전'이 생길 경우 수정이 필요.

    #조회일자 엑셀 기준으로 변환
    df['조회일자'] = df['조회일자'].apply(convert_to_excel_date)

    #결제시각 엑셀 기준으로 변환
    df['결제시각'] = df['결제시각'].apply(convert_to_excel_time)

    #정렬
    df.sort_values(by=['조회일자', '결제시각'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)



    #전처리 된 데이터 csv파일로 변환
    processed_df = df
    processed_df.insert(0, '결제채널', '배달의민족')

    file_name = f'baemin_{date_range_str}_data.xlsx'
    processed_df.to_excel(file_name, index=False)



    folder_path = f"{usersdownload_path}Crawling_data" 

    #디렉토리가 존재하지 않으면 생성
    os.makedirs(folder_path, exist_ok=True)



    current_directory = os.getcwd()     # 현재 작업 디렉토리
    destination_directory = folder_path # 이동시킬 디렉토리

    # 파일 경로
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)

    # 파일 이동
    shutil.move(source_path, destination_path)

    # 현재 디랙토리 rowdata.csv 파일 식제하기
    if os.path.exists(file_path):
        os.remove(file_path)

    print('==================================수집된 데이터의 결과 예시 입니다.==============================')
    print()
    return processed_df

baemin_crawler()

processed_df = preprocess_data()
processed_df.head()

