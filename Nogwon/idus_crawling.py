# -*- coding: utf-8 -*-

##########################사용자 다운로드 폴더 주소 입력###########################

usersdownload_path = 'User_path'

#############################################################################

# 필요한 패키지 설치
#!pip install selenium
#!pip install webdriver_manager

### 아이디어스 데이터 크롤링 함수 ###
def idus_crawler():
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select
    import requests
    import time
    from datetime import datetime
    now = datetime.now()

    # Chrome 드라이버 옵션 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--start-maximized')  
    #chrome_options.add_argument('--headless')  
    chrome_options.add_argument('--no-sandbox')  
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("disable-gpu")   
    chrome_options.add_argument("lang=ko_KR")    
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
    # Chrome 웹 드라이버 생성
    driver = webdriver.Chrome(options=chrome_options)

    url = 'https://artist.idus.com/login?return=%252F'
    ID = ''
    PW = ''

    # 아이디어스 사이트로 이동
    driver.get(url)
    page_source = driver.page_source


    # 아이디 입력
    id_input_selector = '#login_email input[type="email"]'
    id_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, id_input_selector))
    )
    id_input.send_keys(ID)


    # 비밀번호 입력
    pw_input_selector = '#login_password input[type="password"]'
    pw_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, pw_input_selector))
    )
    pw_input.send_keys(PW)


    # 로그인 버튼 클릭
    login_button_selector = '#app > div > span > main > div > div > div > div.pb-11 > form > button'
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, login_button_selector))
    )
    login_button.click()


        # 팝업처리
    try:
        X_button_selector = '/html/body/div[1]/div/div/div[3]/div/div/div[1]/button'
        X_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, X_button_selector))
        )
        X_button.click()
    except:
        pass

    # 신규 주문 버튼 클릭
    order_button_selector = '/html/body/div[1]/div/div/div[1]/div/div[1]/main/div/div/div/div[2]/div[2]/div/div[2]/div[1]/div/div/div'
    order_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, order_button_selector))
    )
    order_button.click()

    # 조회기간 일주일(오늘) 선택
    today_button_selector = '/html/body/div[1]/div/div/div[1]/div/div[1]/main/div/div/div/div[2]/div[1]/span/div[1]/div[1]/div[2]/div[1]/div/div[2]/div/span[2]/span'
    #'/html/body/div[1]/div/div/div[1]/div/div[1]/main/div/div/div/div[2]/div[1]/span/div[1]/div[1]/div[2]/div[1]/div/div[2]/div/span[1]/span'
    today_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, today_button_selector))
    )
    today_button.click()

    # 전체 결제 내역 보기
    all_button_selector = '/html/body/div[1]/div/div/div[1]/div/div[1]/main/div/div/div/div[2]/div[1]/span/div[2]/div/div[2]/div/span[1]/span/div/div[1]/div'
    all_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, all_button_selector))
    )
    all_button.click()


    try:
        order_history_xpath = '/html/body/div[1]/div/div/div[1]/div/div[1]/main/div/div/div/div[2]/div[1]/div[1]/div/div/div/div/div[2]/div/div[1]/div[1]/div/div/div/label'
        order_history_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, order_history_xpath))
        )

        if order_history_element:
            download_button_selector = '#app > div.v-application--wrap > div > div.mx-auto.d-flex > main > div > div > div > div.grey_f5 > div:nth-child(1) > footer > div > button.BaseButton.px-2.mr-3.v-btn.v-btn--outlined.theme--light.v-size--small.grey_33--text.BaseButtonOutlined--grey_33 > span'
            download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, download_button_selector))
            )
            download_button.click()
            time.sleep(2)

        else:
            print(f"{now.date()}에 해당하는 데이터가 없습니다. 크롤링을 스킵합니다.")

    except Exception as e:
        print(f"에러 발생: {str(e)}")

    #드라이버 종료
    driver.quit()

### 이모지 없애는 함수 ###
def remove_emoji(inputString):
    return inputString.encode('cp949', 'ignore').decode('cp949')

### 전처리 함수 ###
def preprocess_idus_data():
    import pandas as pd
    import numpy as np
    import os
    import re
    from datetime import datetime, timedelta

    path = usersdownload_path

    # 지정된 경로의 파일 목록을 얻기
    files = os.listdir(path)

    # 파일이 존재하는 경우에만 진행
    if files:
        # 파일 목록을 최근 수정 시간을 기준으로 정렬
        sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
        # 가장 상단에 위치한 파일 선택
        top_file = sorted_files[0]
        # 선택된 파일의 전체 경로 출력
        full_path = os.path.join(path, top_file)


    xls = pd.ExcelFile(full_path)
    df1 = pd.read_excel(xls, sheet_name=0)  
    df2 = pd.read_excel(xls, sheet_name=1)  
    df2 = df2.drop([0, 1]).reset_index(drop=True)

    combined_df1 = df1
    combined_df2 = df2


    # 모든 행과 열을 출력하도록 설정
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  

    # 열 이름 변경
    combined_df2.columns = ['작품명', '주문수량']


    combined_df2['작품명'] = combined_df2['작품명'].apply(lambda x: re.sub(r'\/.*', '', x))
    combined_df2['작품명'] = combined_df2['작품명'].str.rstrip()
    combined_df2['작품명'] = combined_df2['작품명'].apply(remove_emoji)
    combined_df2['주문수량'] = combined_df2['주문수량'].astype(int)
    combined_df2 = combined_df2.groupby('작품명', as_index=False)['주문수량'].sum()
    product_list = combined_df2.copy()

    product_list['상품명'] = np.nan

    product_list.loc[product_list['작품명'].str.contains('블렌딩 티'), '상품명'] = '블렌딩 티 세트'
    product_list.loc[product_list['작품명'].str.contains('보은 대추고'), '상품명'] = '보은 대추고'
    product_list.loc[product_list['작품명'].str.contains('9개입|선물 세트|선물세트'), '상품명'] = '양갱 9개입 선물세트'
    product_list.loc[product_list['작품명'].str.contains('4구세트|4구 세트'), '상품명'] = '양갱 4구세트'
    product_list.loc[product_list['작품명'].str.contains('골라담기'), '상품명'] = '양갱 4구&티세트 골라담기'
    product_list.loc[product_list['작품명'].str.contains('못난이'), '상품명'] = '못난이 양갱'
    product_list.loc[product_list['작품명'].str.contains('한정판 초콜릿'), '상품명'] = '한정판 초콜릿양갱'
    product_list.loc[product_list['작품명'].str.contains('12시차'), '상품명'] = '오후 12시차(7개입)'
    product_list.loc[product_list['작품명'].str.contains('3시차'), '상품명'] = '오후 3시차(7개입)'
    product_list.loc[product_list['작품명'].str.contains('6시차'), '상품명'] = '오후 6시차(7개입)'


    combined_df1['작품명'] = combined_df1['작품명'].apply(lambda x: re.sub(r'\/.*', '', x))
    combined_df1['작품명'] = combined_df1['작품명'].str.rstrip()
    combined_df1['작품명'] = combined_df1['작품명'].apply(remove_emoji)

    merged_df = pd.merge(combined_df1, product_list[['작품명', '상품명']], on='작품명', how='left')

    merged_df.loc[merged_df['작품명'].str.contains('개인결제|개인 결제'), '상품명'] = '개인 결제'

    options_split = merged_df['옵션'].str.split(' / ', expand=True)

    if len(options_split.columns) < 8:
        num_new_columns = 8 - len(options_split.columns) 산

        for i in range(len(options_split.columns), 8):
            new_column_name = f'{i}'
            options_split[new_column_name] = np.nan

    options_split.columns = [f'옵션_{i}' for i in range(1, len(options_split.columns) + 1)]

    for col in options_split.columns:
        options_split[col] = options_split[col].astype(str)

        options_split.loc[~options_split[col].str.contains('양갱|제품선택|티 1|티2|티3', na=False), col] = np.nan


    merged_df = pd.concat([merged_df, options_split.iloc[:, :-1]], axis=1)

    merged_df.loc[merged_df['옵션_1'].str.contains('3시차', na=False), '상품명'] = '오후 3시차(7개입)'
    merged_df.loc[merged_df['옵션_1'].str.contains('6시차', na=False), '상품명'] = '오후 6시차(7개입)'
    merged_df.loc[merged_df['옵션_1'].str.contains('12시차', na=False), '상품명'] = '오후 12시차(7개입)'


    merged_df.loc[merged_df['옵션_1'].str.contains('제품선택', na=False), '옵션_1'] = np.nan

    merged_df.rename(columns={'옵션_1': '양갱1', '옵션_2': '양갱2', '옵션_3': '양갱3', '옵션_4': '양갱4', '옵션_5': '티1', '옵션_6': '티2', '옵션_7': '티3'}, inplace=True)


    merged_df.iloc[:, -7:] = merged_df.iloc[:, -7:].apply(lambda x: x.str.split(': ').str[1].str.strip())


    final_df = merged_df.drop(['옵션', '요청사항', '회원메모', '주문메모', '후원', '주문자', '주문자 전화번호', '받는분', '우편번호', '주소', '전화번호'], axis=1)

    final_df['주문일자'] = final_df['주문번호'].str.split('_').str[1].str[:8]


    final_df['주문일자'] = pd.to_datetime(final_df['주문일자'], format='%Y%m%d', errors='coerce')


    final_df.iloc[:, 8:14] = final_df.iloc[:, 8:14].apply(lambda x: x.astype(str).str.replace(r'\s+', '', regex=True))
    final_df['티3'] = final_df['티3'].astype(str).str.replace(r'\s+', '', regex=True) #'티3 열만 적용이 안되어서 따로 처리'



    column_order = [
        '주문일자', '주문번호', '주문상태', '작품명', '상품명', '양갱1', '양갱2',
        '양갱3', '양갱4', '티1', '티2', '티3', '수량', '결제금액'
    ]
    final_df = final_df[column_order]

    final_df.sort_values(by=['주문일자', '주문번호'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    day_of_week = final_df[['주문일자']].copy()
    day_of_week['주문요일'] = day_of_week['주문일자'].dt.strftime('%A')
    weekday_names_korean = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    day_of_week['주문요일'] = day_of_week['주문일자'].dt.weekday.map(lambda x: weekday_names_korean[x])
    final_df.insert(1, '주문요일', day_of_week.iloc[:,1])

    final_df.replace('nan', '', inplace=True)

    return final_df


### 함수 실행 및 파일 변환 함수 ###
def main_process():
    from datetime import datetime, timedelta
    import os
    import pandas as pd
    import re
    from glob import glob

    idus_crawler()
    processed_df = preprocess_idus_data()
    processed_df.head()

    current_date = datetime.now()
    start_date = current_date - timedelta(days=6)
    date_range_str = f'{start_date.strftime("%Y-%m-%d")}_{current_date.strftime("%Y-%m-%d")}'

    Downloads_pafh = usersdownload_path
    folder_path = f"{Downloads_pafh}Crawling_data" 
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'idus_{date_range_str}_data.xlsx')
    processed_df.to_excel(file_path, index=False)


    #다운 받은 엑셀파일 삭제
    today = current_date.strftime("%Y%m%d")
    file_pattern = f'order_list*{today}*'
    files_to_delete = glob(os.path.join(Downloads_pafh, file_pattern))
    file_paths_list = sorted(files_to_delete, key=os.path.getmtime, reverse=False)

    for file_to_delete in file_paths_list:
        os.remove(file_to_delete)


    print('=================================================================수집된 데이터의 결과 예시 입니다.==============================================================')
    print()
    return processed_df

processed_df = main_process()

