setwd("C:/R")
install.packages("extrafont")
library(extrafont)

font_import(pattern = "AppleGothic")
loadfonts()
par(family = "AppleGothic")




# row.names를 NULL로 설정하여 자동으로 행 인덱스를 생성하도록 함
df <- read.csv("CARD_SUBWAY_MONTH_202310 (3).csv", header = TRUE, row.names = NULL)
head(df)
# X.test 열을 삭제
df <- df[, -1]
# 수정된 데이터프레임 확인
head(df)



### 1. 호선별 승차총승객수
get_on <- aggregate(승차총승객수 ~ 호선명, data = df, sum)
#내림차순정렬
get_on <- get_on[order(-get_on$승차총승객수), ]
#데이터 확인
get_on


### 2. 호선별 하차총승객수
#호선별 승차총승객수
get_off <- aggregate(하차총승객수 ~ 호선명, data = df, sum)
#내림차순정렬
get_off <- get_off[order(-get_off$하차총승객수), ]
#데이터 확인
get_off

#승차와 하차 사이의 차이는 없음

### 3. 승하차
get <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 호선명, data = df, sum)
#승차총승객수와 하차총승객수를 더해서 '총승객수'열 새로 만들기
get$호선별총승객수 <- get$승차총승객수 + get$하차총승객수
#데이터 확인
get
#총승객수 값의 내림차순 정렬
get <- get[order(-get$호선별총승객수), ]
#데이터 확인
head(get)

### 1-1. 2호선별  상하위 역 3개 뽑아보기
#subset으로 2호선만 선택
second <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 역명, data = subset(df, 호선명 == '2호선'), sum)
second$승하차총승객수_2호선 <- second$승차총승객수 + second$하차총승객수
second <- second[order(-second$승하차총승객수_2호선), ]
#데이터 확인
head(second)
tail(second)

### 1-2. 5호선별 상하위 역 3개
#subset으로 5호선만 선택
fifth <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 역명, data = subset(df, 호선명 == '5호선'), sum)
fifth$승하차총승객수_5호선 <- fifth$승차총승객수 + fifth$하차총승객수
fifth <- fifth[order(-fifth$승하차총승객수_5호선), ]
#데이터 확인
head(fifth)
tail(fifth)

### 1-3. 7호선별 상하위 역 3개
seventh <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 역명, data = subset(df, 호선명 == '7호선'), sum)
seventh$승하차총승객수_7호선 <- seventh$승차총승객수 + seventh$하차총승객수
seventh <- seventh[order(-seventh$승하차총승객수_5호선), ]
#데이터 확인
head(seventh)
tail(seventh)

### 1-4. 3호선별 상하위 역 3개
third <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 역명, data = subset(df, 호선명 == '3호선'), sum)
third$승하차총승객수_7호선 <- third$승차총승객수 + third$하차총승객수
third <- third[order(-third$승하차총승객수_5호선), ]
#데이터 확인
head(third)
tail(third)


### 4. 역별 승차총승객수
st_on <- aggregate(승차총승객수 ~ 역명, data = df, sum)
#내림차순정렬
st_on <- st_on[order(-st_on$승차총승객수), ]
#데이터 확인
st_on_h10 = head(st_on,10)
st_on_t10 = tail(st_on,10)
st_on_h10

### 5. 역별 하차총승객수
st_off <- aggregate(하차총승객수 ~ 역명, data = df, sum)
#내림차순정렬
st_off <- st_off[order(-st_off$하차총승객수), ]
#데이터 확인
st_off_h10 <- head(st_off, 10)
st_off_t10 <- tail(st_off, 10)
st_off_t10

### 6. 역별 승하차
get_station <- aggregate(cbind(승차총승객수, 하차총승객수) ~ 역명, data = df, sum)
#승차총승객수와 하차총승객수를 더해서 '역별총승객수'열 새로 만들기
get_station$역별총승객수 <- get_station$승차총승객수 + get_station$하차총승객수
#데이터 확인
get_station
#총승객수 값의 내림차순 정렬
get_station <- get_station[order(-get_station$역별총승객수), ]

#서울 지역 필터링(하위)
bt_st_seoul <- c('구반포', '도림천', '남태령', '신답', '응봉', '삼양', '버티고개', '신내', '학여울', '개화')
# 역명이 일치하는 행 필터링
sb_t10 <- get_station[get_station$역명 %in% bt_st_seoul, ]
# 결과 확인

sb_t10
#데이터 확인
sb_h10 = head(get_station, 10)
sb_h10

#승차와 하차의 차이가 거의 없음 --> 승하차 합쳐서 사용가능
#이후의 회귀분석과 t-test를 위해 상위40 데이터 추출
sb_h40 <- head(get_station, 40)


#승차와 하차승객수 사이에 역별로 순위 차이가 있는지 확인
st_on_h10$승차총승객수 <- as.numeric(st_on_h10$승차총승객수)
st_off_h10$하차총승객수 <- as.numeric(st_off_h10$하차총승객수)
st_on_h10

par(mfrow = c(1, 2))

# Plot for 승차총승객수 상위10개
barplot(st_on_h10$승차총승객수, names.arg = st_on_h10$역명, las = 2,
        main = "Top10 by Station (승차)")

# Plot for 하차총승객수 상위10개
barplot(st_off_h10$하차총승객수, names.arg = st_off_h10$역명, las = 2,
        main = "Top10 by Station (하차)")


#승차와 하차승객수 사이에 역별로 순위 차이가 있는지 확인
st_on_t10$승차총승객수 <- as.numeric(st_on_t10$승차총승객수)
st_off_t10$하차총승객수 <- as.numeric(st_off_t10$하차총승객수)

par(mfrow = c(1, 2))

# Plot for 승차총승객수 하위10개 
barplot(st_on_t10$승차총승객수, names.arg = st_on_t10$역명, las = 2,
        main = "Bottom10 by Station (승차)")

# Plot for 하차총승객수 하위10개 
barplot(st_off_t10$하차총승객수, names.arg = st_off_t10$역명, las = 2,
        main = "Bottom10 by Station (하차)")


sb_h10$역별총승객수 <- as.numeric(sb_h10$역별총승객수)
sb_t10$역별총승객수 <- as.numeric(sb_t10$역별총승객수)

par(mfrow = c(1, 2))

# Plot for 역별총승객수 상위10개 
barplot(sb_h10$역별총승객수, names.arg = sb_h10$역명, las = 2,
        main = "Top10 by Station")

# Plot for 역별총승객수 하위10개 
barplot(sb_t10$역별총승객수, names.arg = sb_t10$역명, las = 2,
        main = "Bottom10 by Station")



#서울 지역으로 필터링하고 상하위 10개 지하철역 & 스타벅스 매장수 데이터 프레임 생성

# 상위
# 스타벅스 같이 있는 파일 불러오기
top_df <- read.csv("top_station_starbucks.csv", header = TRUE)
head(top_df)

# top_df$역.명 열의 값 중에서 ~해당하는 행의 빈도를 계산
top_counts <- table(top_df$역.명)
top_counts[c("서울역", "잠실(송파구청)", "홍대입구", "고속터미널", "강남", "사당",
             "선릉", "구로디지털단지", "신림", "가산디지털단지")]

# 데이터프레임으로 바꾸기
seoul_tp10_sb <- as.data.frame(top_counts)
seoul_tp10_sb
# 열 이름 변경
colnames(seoul_tp10_sb) <- c("station", "스타벅스매장수")

# dplyr 패키지 설치 및 불러오기
install.packages("dplyr")
library(dplyr)

# 역명을 기준으로 데이터프레임 병합
sb_h10 <- left_join(sb_h10, seoul_tp10_sb, by = c("역명" = "station"))

# 병합된 최종 데이터프레임
print(sb_h10)



# 수치형으로 변환
sb_h10$스타벅스매장수 <- as.numeric(sb_h10$스타벅스매장수)

# 바 그래프 그리기
barplot(sb_h10$스타벅스매장수, 
        names.arg = sb_h10$역명,
        col = "skyblue",
        main = "Top 10 Station & Starbucks in Seoul",
        ylim = c(0, max(sb_h10$스타벅스매장수) + 4),
        ylab = "Number of Starbucks",
        las = 2)

#하위(스타벅스 매장수 열 추가)
sb_t10$스타벅스매장수 <- c(0, 1, 0, 0, 0, 0, 1, 0, 0, 0)
sb_t10

# 수치형 변환
sb_t10$스타벅스매장수 <- as.numeric(sb_t10$스타벅스매장수)

# bar plot for bottom 10station
barplot(sb_t10$스타벅스매장수, 
        names.arg = sb_t10$역명,
        col = 'skyblue',
        main = "Bottom 10 Station & Starbucks in Seoul",
        ylab = 'Number of Starbucks',
        ylim = c(0, max(sb_h10$스타벅스매장수) + 4),
        las = 2)


# 상위 산점도 그리기
plot(sb_h10$역별총승객수, sb_h10$스타벅스매장수, 
     main = "역별총승객수와 스타벅스 매장수 산점도",
     xlab = "역별총승객수", ylab = "스타벅스 매장수",
     pch = 19, col = "green")

# 각 점에 대한 라벨(역명) 추가
text(sb_h10$역별총승객수, sb_h10$스타벅스매장수, labels = sb_h10$역명, pos = 4, cex = 0.5)


# 하위 산점도 그리기
plot(sb_t10$역별총승객수, sb_t10$스타벅스매장수, 
     main = "역별총승객수와 스타벅스 매장수 산점도",
     xlab = "역별총승객수", ylab = "스타벅스 매장수", 
     pch = 19, col = "yellow")

# 각 점에 대한 라벨(역명) 추가
text(sb_t10$역별총승객수, sb_t10$스타벅스매장수, labels = sb_t10$역명, pos = 4, cex = 0.5)



#1129
df_40 <- read.csv("상위 40개 역 스벅 교차수정 (1).csv", header = TRUE, row.names = NULL)
head(df_40)

# unique한 역명 추출
unique_stations <- unique(df_40$역.명)
print(unique_stations)


# top_df$역.명 열의 값 중에서 ~해당하는 행의 빈도를 계산
top40_st <- table(df_40$역.명)
top40_st[unique_stations]

# 데이터프레임으로 바꾸기
top40_st <- as.data.frame(top40_st)
top40_st
# 열 이름 변경
colnames(top40_st) <- c("역명", "스타벅스매장수")

# 역명이 일치하는 행 필터링
top40 <- get_station[get_station$역명 %in% unique_stations, ]
top40

library(dplyr)

# 역명을 기준으로 데이터프레임 병합
top40_final <- left_join(top40_st, top40, by = c("역명" = "역명"))
top40_final
# 임의로 넣었던 대림역은 역주변에 스타벅스가 존재하지 않기 때문에 0값으로 바꿔주기
top40_final$스타벅스매장수[top40_final$역명 == '대림(구로구청)'] <- 0

# 병합된 최종 데이터프레임
print(top40_final)


# 상위 40개 시각화
# 산점도
par(mfrow = c(1, 1))

plot(top40_final$역별총승객수, top40_final$스타벅스매장수, 
     main = "역별총승객수와 스타벅스 매장수 산점도",
     xlab = "역별총승객수", ylab = "스타벅스 매장수", 
     pch = 19, col = "Blue")



subway <- read.csv('서울시역별승하차수.csv', header = TRUE, fileEncoding = "CP949")
subway_40 <- head(subway[order(subway$승하차수, decreasing = TRUE), ], 40)
head(subway_40)

starbucks_40 <- read.csv("상위 40개 역 스벅 교차수정 (1).csv", header = TRUE, row.names = NULL)
head(starbucks_40)

setdiff(sort(subway_40$역명), starbucks_40$역.명) # subway 기준
setdiff(starbucks_40$역.명, sort(subway_40$역명)) # starbucks 기준

# starbucks_40에서 해당 역들의 이름 수정
new_starbucks_names <- c("건대입구", "교대(법원.검찰청)", "동대문역사문화공원(DDP)", "서울대입구(관악구청)", "서울역", "역삼", "왕십리(성동구청)", "청량리(서울시립대입구)")
starbucks_40$역.명[starbucks_40$역.명 %in% setdiff(starbucks_40$역.명, sort(subway_40$역명))] <- new_starbucks_names

setdiff(sort(subway_40$역명), starbucks_40$역.명) # subway 기준
setdiff(starbucks_40$역.명, sort(subway_40$역명)) # starbucks 기준

# top_df$역.명 열의 값 중에서 ~해당하는 행의 빈도를 계산
top40_st <- as.data.frame(table(starbucks_40$역.명))
top40_st

# 열 이름 변경
colnames(top40_st) <- c("역명", "스타벅스매장수")
top40_st

library(dplyr)

# subway_40와 top40_st를 '역명'을 기준으로 left join하기
top40 <- left_join(subway_40, top40_st, by = "역명")

# 임의로 넣었던 대림역은 역주변에 스타벅스가 존재하지 않기 때문에 0값으로 바꿔주기
top40$스타벅스매장수[top40$역명 == '대림(구로구청)'] <- 0

# 병합된 최종 데이터프레임
print(top40)

library(ggplot2)

# 회귀분석 수행
model <- lm(승하차수 ~ 스타벅스매장수, data = top40)

# 회귀분석 결과 출력
summary(model)

# 회귀분석 결과 시각화
ggplot(top40, aes(x = 스타벅스매장수, y = 승하차수)) +
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "스타벅스매장수", y = "승하차수") +
  ggtitle("승하차수와 스타벅스매장수 회귀분석")

# 40개의 행을 10개씩 4개의 그룹으로 나누기
top40$group <- rep(1:4, each = 10)
top40$group <- factor(top40$group)

# Boxplot 그리기
ggplot(top40, aes(x = group, y = 승하차수, fill = group)) +
  geom_boxplot() +
  labs(x = "그룹", y = "승하차수", title = "그룹별 승하차수 Boxplot") +
  theme_minimal()

# ANOVA 분석 수행
anova_result <- aov(승하차수 ~ group, data = top40)

# ANOVA 분석 결과 출력
summary(anova_result)

