# Data-Analysis-bus (2022)
## 버스 승객 예측
인하공업전문대학 컴퓨터정보공학과 김수림
> Library : RandomForestm, AdaBoost, LightGBM, scikit-learn  
> Environment :
## Overview
* 주제 선정  
매년 제주도에 거주하는 인구수는 점점 증가하고, 관광을 목적으로 제주도를 방문하는 인구도 증가하는 추세다. 또한 제주도 일부 지역은 교통 체증이 서울보다 심각하다고 한다. 이런 경우에는 이동 인구를 사전에 예측하여 대중교통을 배치하면 시민들이 느끼는 불편함을 줄이고 교통체증도 완화 된다.
따라서, 2019년 9월 제주도 버스 정류장 별 승차 인원 데이터를 이용해 10월 퇴근 시간 버스 승차 인원 예측하고자 한다.  

* 사용한 데이터  
국내 데이터 경쟁 플랫폼인 DACON에서 진행된 ‘퇴근 시간 버스 승차 인원 예측 경진대회’의 데이터들은 이용하였다.  
> train.csv : 2019년 9월 각 날짜, 출근 시간(6시-12시)의 버스 정류장별 승하차 인원과 퇴근 시간(18시-20시)의 버스 정류장별 승차 인원 데이터  
> test.csv : 2019년 10월 각 날짜, 출근 시간(6시-12시)의 버스 정류장별 승하차 인원  
> weather.csv : 2019년 9월 제주도 전체 오전 10시 기상정보 데이터   
> rain.csv : 2019년 9월 고산, 서귀포, 제주, 성산 지점별 오전 6시~11시 평균 기상 정보 데이터  
> df_location.csv, life_location.csv :위도, 경도 정보로 나타나는 제주도 주소를 Geocoder-Xr 프로그램을 이용해 지번 주소, 도로명 주소로 변환하여 나타낸 데이터  
> bus_btn.csv : 승객 버스 카드 ID별 승객이 탑승한 버버스 ID, 날짜  
> jeju_financial_life_data.csv : 제주도 우편번호를 단위로 구분한 금융 생활 통계 자료 (경도, 위도, 직업군별 비율, 평균 연소득, 평균 소비액)

* 분석 목표  
![image](https://user-images.githubusercontent.com/71176581/191956529-54f431b4-a272-4ccd-a1a4-fbad63f3f084.png)  
본 연구는 2019년 9, 10월 제주도 출근 시간, 9월 퇴근 시간 버스 정류장별 승차 인원 데이터를 이용해 2019년 10월 제주도 퇴근 시간 버스 정류장별 승차 인원을 예측하는 것을 목표로 한다.  

머신러닝 모델에 입력으로 들어갈 변수는 숫자 형태여야 하므로 숫자 형태가 아닌 변수를 제거하고 변수명 df 에 저장한다. 그런 다음, df 데이터를 모델 학습하기 위한 학습데이터와 학습데이터를 테스트하기 위한 테스트 데이터로 나눈다.  
학습데이터, 테스트 데이터를 구분하는 cue 변수가 0인 경우 X_train으로, 1인 경우 X_test로 정의하고 훈련에 사용할 변수를 cue 값이 0인 경우의 18-20시에 버스 승차 인원수(18~20_ride)를 y_train으로 정의한다.

머신러닝 모델을 훈련하기 위해 여러 개의 모델을 적절하게 결합해 최종값을 도출하는 앙상블 모델을 이용하였다. 이 연구에는 Random Forest, XGboost, LightGBM, Adaboost, Gradient Tree Boosting 라이브러리를 사용하여 머신러닝 모델을 훈련한다.  

* 추가한 데이터  
앞에서 사용된 데이터 이외에도 퇴근 시간 버스 정류장 별 승차 인원을 예측할 때, 예측 인원에 영향을 미칠 수 있는 유가, 날씨 데이터를 추가한다. 
날짜별 유가 정보를 제공해주는 사이트인 Opinet을 이용하여 2019년 9월 1일부터 10월 16일까지 제주도 주유소 평균 휘발유, 경유 판매가격(￦)을 수집한다. 날짜별 날씨 정보는 기상청 날씨누리 사이트를 이용하여 2019년 9월 한달 간 오전 6-11시 기온정보(°C), 오전 7-11시 강수량(mm) 정보를 수집한다.  
휘발유가격 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 경유가격  
<img src = "https://user-images.githubusercontent.com/71176581/191965090-8aa2b9f7-823b-4ddf-82e9-534df890c2cb.png" width="150" height="200"> <img src = "https://user-images.githubusercontent.com/71176581/191965232-1f503541-3ab8-4863-8a77-5d32ee390829.png" width="150" height="200">  
날씨 정보  
![image](https://user-images.githubusercontent.com/71176581/191976155-47397365-bcea-4ea0-9643-645cee8602c1.png)

![image](https://user-images.githubusercontent.com/71176581/191989735-22b0ef33-8ce1-4da5-ab6e-bad906ae0a34.png)  
![image](https://user-images.githubusercontent.com/71176581/191989958-c1f962d8-7a06-4fa3-af3d-37b44a618100.png)  
![image](https://user-images.githubusercontent.com/71176581/191990363-3c4c92cc-9d8a-4436-9e82-9a47006204ba.png)  
  
![image](https://user-images.githubusercontent.com/71176581/191992468-8e24d9c4-1883-41ea-b202-b34c748d8c3d.png) 
![image](https://user-images.githubusercontent.com/71176581/191992686-82eaa342-d835-41c8-8715-e4787c95ef20.png)

![image](https://user-images.githubusercontent.com/71176581/191994371-9cf403e7-360e-4a95-a333-2a525758f704.png)
![image](https://user-images.githubusercontent.com/71176581/191993886-4d5da3fe-3e43-439a-80de-7f04fc5ecc3b.png)

![image](https://user-images.githubusercontent.com/71176581/192001627-fab4cb22-72e0-4829-8fe6-9a439a8859e7.png)  
