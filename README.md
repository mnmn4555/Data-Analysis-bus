# Data-Analysis-bus (2022)
## 제주도 버스 퇴근시간 승객 예측
인하공업전문대학 컴퓨터정보공학과 김수림
> Library : RandomForestm, AdaBoost, LightGBM, Gradient Tree Boosting, scikit-learn  
> Environment : Google Colab, 국가슈퍼컴퓨팅센터-neuron
## Overview
* 주제 선정  
매년 제주도에 거주하는 인구수는 점점 증가하고, 관광을 목적으로 제주도를 방문하는 인구도 증가하는 추세다. 또한 제주도 일부 지역은 교통 체증이 서울보다 심각하다고 한다. 이런 경우에는 이동 인구를 사전에 예측하여 대중교통을 배치하면 시민들이 느끼는 불편함을 줄이고 교통체증도 완화 된다.
따라서, 2019년 9월 제주도 버스 정류장 별 승차 인원 데이터를 이용해 10월 퇴근 시간 버스 승차 인원 예측하고자 한다.  

* 사용한 데이터  
국내 데이터 경쟁 플랫폼인 DACON에서 진행된 ‘퇴근 시간 버스 승차 인원 예측 경진대회’의 데이터들은 이용한다.  
> train.csv : 2019년 9월 각 날짜, 출근 시간(6시-12시)의 버스 정류장별 승하차 인원과 퇴근 시간(18시-20시)의 버스 정류장별 승차 인원 데이터  
> test.csv : 2019년 10월 각 날짜, 출근 시간(6시-12시)의 버스 정류장별 승하차 인원  
> weather.csv : 2019년 9월 제주도 전체 오전 10시 기상정보 데이터   
> rain.csv : 2019년 9월 고산, 서귀포, 제주, 성산 지점별 오전 6시~11시 평균 기상 정보 데이터  
> df_location.csv, life_location.csv :위도, 경도 정보로 나타나는 제주도 주소를 Geocoder-Xr 프로그램을 이용해 지번 주소, 도로명 주소로 변환하여 나타낸 데이터  
> bus_btn.csv : 승객 버스 카드 ID별 승객이 탑승한 버버스 ID, 날짜  
> jeju_financial_life_data.csv : 제주도 우편번호를 단위로 구분한 금융 생활 통계 자료 (경도, 위도, 직업군별 비율, 평균 연소득, 평균 소비액)  

* RMSE (Root Mean Square Deviation: 평균 제곱근 오차)  
![image](https://user-images.githubusercontent.com/71176581/192127456-f3ca3b88-0d91-4b87-87d6-0b418ee445ae.png)  
추정값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때 흔히 사용하는 측도로 정밀도를 표현할 때 적합하다. 실제 값(x)에서 예측값(y) 차이의 제곱 합을 데이터 전체(m) 데이터로 나눈 뒤, 제곱근을 구하면 RMSE가 구해진다. 예측한 퇴근시간 버스 승차인원이 실제 퇴근시간 버스 승차인원과 유사할수록 RMSE가 낮다. 이를 통해. RMSE가 낮을수록 학습이 잘된 모델로 평가할 수 있다.  

* 분석 목표  
본 연구는 2019년 9, 10월 제주도 출근 시간, 9월 퇴근 시간 버스 정류장별 승차 인원 데이터를 이용해 2019년 10월 제주도 퇴근 시간 버스 정류장별 승차 인원을 예측하는 것을 목표로 한다.  
![image](https://user-images.githubusercontent.com/71176581/191956529-54f431b4-a272-4ccd-a1a4-fbad63f3f084.png)     
머신러닝 모델에 입력으로 들어갈 변수는 숫자 형태여야 하므로 숫자 형태가 아닌 변수를 제거하고 변수명 df 에 저장한다. 그런 다음, df 데이터를 모델 학습하기 위한 학습데이터와 학습데이터를 테스트하기 위한 테스트 데이터로 나눈다.  
학습데이터, 테스트 데이터를 구분하는 cue 변수가 0인 경우 X_train으로, 1인 경우 X_test로 정의하고 훈련에 사용할 변수를 cue 값이 0인 경우의 18-20시에 버스 승차 인원수(18~20_ride)를 y_train으로 정의한다.  

머신러닝 모델을 훈련하기 위해 여러 개의 모델을 적절하게 결합해 최종값을 도출하는 앙상블 모델을 이용한다. 모델 구축에 사용할 데이터는 다음과 같다.  
![image](https://user-images.githubusercontent.com/71176581/192127642-fb255611-bff7-41dc-a591-e02bc46b7211.png)  
이 연구에는 Random Forest, LightGBM, Adaboost, Gradient Tree Boosting 라이브러리를 사용하여 머신러닝 모델을 훈련한다.  

* 알고리즘 별 모델 상관계수, 임시 스코어  
여기서 말하는 임시 스코어는 Dacon 사이트에 학습한 모델 파일을 제출했을 때 보이는 public score RMSE로 과적합을 방지하여 실제 값과 차이를 구한 스코어를 기준으로 연구를 진행한다.  
> 1. RandomForest, LightGBM 상관계수·임시계수와 10월 퇴근시간 버스 승차인원 예측  
> 
> <img src ="https://user-images.githubusercontent.com/71176581/192128088-770db020-0c36-474b-9e2c-003f33897783.png" weight = "100" height = "100">
> <img src ="https://user-images.githubusercontent.com/71176581/192410519-4eb5f12e-dd36-4a16-9838-d660d3aedc3e.png" weight = "150" height = "150">
> <img src ="https://user-images.githubusercontent.com/71176581/192128300-4248b41c-1b6c-4a42-b779-10209f20a42a.png" weight = "300" height = "200">
>
> 2. AdaBoost 상관계수·임시계수와 10월 퇴근시간 버스 승차인원 예측  
> <img src ="https://user-images.githubusercontent.com/71176581/192128096-0b7d3d2e-2e59-4ee5-8b3f-c7935038aadb.png" weight = "80" height = "80">
> <img src ="https://user-images.githubusercontent.com/71176581/192410590-be527a97-3f1f-423a-91ac-bb1016d168cc.png" weight = "150" height = "150">
> <img src ="https://user-images.githubusercontent.com/71176581/192128206-5c0ab19a-41b8-4aea-80b5-64e4ab993141.png" weight = "300" height = "200">  
> 
> 3. Gradient Tree Boosting 상관계수·임시계수와 10월 퇴근시간 버스 승차인원 예측   
> <img src = "https://user-images.githubusercontent.com/71176581/192128109-17f2546c-b6d4-43e6-a54b-4c47c3decbc3.png" weight = "100" height = "100">
> <img src = "https://user-images.githubusercontent.com/71176581/192410672-f0981586-b124-4610-af36-f7bc4a404a9d.png" weight = "250" height = "200">
> <img src ="https://user-images.githubusercontent.com/71176581/192128213-a6d23316-372b-4407-a613-60da874f2e8b.png" weight = "300" height = "200">  
> AdaBoost를 이용해 구한 10월 퇴근시간 버스 승차인원은 다른 알고리즘을 이용해 나온 퇴근시간 버스 승차인원에 비해 현저히 높은 수치의 값이 보여진다. 따라서 AdaBoost는 이 연구에는 적합하지 않은 알고리즘이다. 한가지 아쉬운점이 있다면 실제 2019년 10월 제주도 버스 퇴근시간 승차인원 정보를 구할수 없어 알고리즘들을 이용해 예측한 퇴근시간 승차인원과 비교할 수가 없었다. 

 * 추가 데이터  
많은 데이터 중 유가 정보가 퇴근시간 버스 승차인원 예측에 영향을 줄 수 있다고 생각한다. 날짜별 유가 정보를 제공해주는 사이트인 Opinet을 이용하여 2019년 9월 1일 ~ 2019년 10월 16일 제주도 주유소 평균 휘발유, 경유 판매가격(￦)을 이용한다.  
> * RandomForest, LightGBM, Gradient Tree Boosting(1,3번 이용) 상관계수·임시계수  
>
> 유가 정보 추가 전   
> <img src = "https://user-images.githubusercontent.com/71176581/192409665-0bc39b4f-0d32-43be-b32b-396a6ecfde5a.png" weight = "450" height = "300">  
> 
> 유가 정보 추가 후  
> <img src = "https://user-images.githubusercontent.com/71176581/192409679-2f61c235-52c0-480a-b4af-056bcd75f929.png" weight = "450" height = "300">  
> 유가 정보를 추가하고나서 전반적으로 임시 스코어와 상관계수 값이 높아져 퇴근시간 버스 승차인원을 더 정확하게 예측할 수 있다.  * 결론  
>

* 추가 연구  
앞서 진행했던 연구는 2019년 10월 퇴근시간 버스 승차 인원을 예측하기 위해 RandomForest, AdaBoost, LightGBM 알고리즘을 사용했다. 이를 통해 버스 승차 인원 예측에 영향을 줄 수있는 데이터를 추가하여 알고리즘 간 RMSE, 상관관계를 구해 알고리즘 별 성능을 비교하고자 한다.

앞서 진행했던 연구는 2019년 10월 퇴근시간 버스 승차 인원의 실제 정보가 없어서 실제 예측했던 모델의 정확도를 판단하기에는 어려웠다. 따라서 2019년 9월 한달간 제주도 버스 정류장별 날짜별 실제 퇴근시간 버스 승차 인원 정보와 예측했던 퇴근시간 버스 승차 인원정보가 얼마나 정확하게 예측했는지 알아보고자 추가 연구를 진행하게 되었다.  
입력 데이터는 train.csv에서 2019년 9월 날짜별 정류장, 버스 노선, 시간대별 승차인원과 2019년 9월 한달 간 제주도 유가, 날씨정보를 추가로 이용한다. 출력 데이터는 날짜별 저녁 승차 인원이다.  

* train.csv 파일 
<img src ="https://user-images.githubusercontent.com/71176581/192088832-e0661f9c-89bd-4675-be8a-336464218334.png" weight = "1674" height = "200">

* 추가한 데이터  
퇴근 시간 버스 정류장 별 승차 인원을 예측할 때, 예측 인원에 영향을 미칠 수 있는 유가, 날씨 데이터를 train.csv에 추가한다.
날짜별 유가 정보를 제공해주는 사이트인 Opinet을 이용하여 2019년 9월 한달간 제주도 주유소 평균 휘발유, 경유 판매가격(￦)을 이용한다. 날짜별 날씨 정보는 기상청 날씨누리 사이트를 이용하여 2019년 9월 한달 간 오전 6-11시 기온정보(°C), 오전 7-11시 강수량(mm) 정보를 수집한다.  

> 유가 정보 - Data Scaling
> 유가 정보는 다른 요소들과 비교하여 데이터 스케일이 현저히 커 데이터를 정제하지 않고 학습하면 제대로 동작하지 않는다. 따라서 유가 정보의 Data Scaling이 필요하다.
> 데이터 스케일링 중 최소값을 0, 최대값을 1, 그 외의 모든 값을 0과1 사이의 값으로 변환하는 MinMaxScaler()을 이용하여 유가 정보를 정제한다.  
> 유가 정보 Data Scaling 전
> ![image](https://user-images.githubusercontent.com/71176581/192081726-3d0023a0-32a5-4c8d-b10e-8ec8f373a095.png)  
> 유가 정보 Data Scaling 후
> ![image](https://user-images.githubusercontent.com/71176581/192081739-05124347-4c82-4439-a902-0653b7f687f3.png)  
>
> 날씨 정보  
> ![image](https://user-images.githubusercontent.com/71176581/191976155-47397365-bcea-4ea0-9643-645cee8602c1.png)  

scikit-learn에서 제공하는 train_test_split를 이용하여 학습 데이터는 전체 입력데이터 중에 80%, 테스트 데이터는 20%로 무작위하게 나눴다.  
![image](https://user-images.githubusercontent.com/71176581/192127130-269a80a2-f621-4c25-989e-2a2df77400ee.png)  

* GridSearchCV  
GridSearchCV는 scikit-learn에서 분류, 회귀 알고리즘에 사용되는 하이퍼파라미터를 순차적으로 입력해 학습,측정 과정을 통해 가장 적합한 파라미터를 알려주는 것이다. 위 연구에서 퇴근시간 버스 승차인원을 예측하는데 사용했던 Randomforest, AdaBoost, LightGBM 알고리즘을 사용해 GridSearchCV를 진행했다. 그 결과는 다음과 같다.
> RandomForest  
> ![image](https://user-images.githubusercontent.com/71176581/191989735-22b0ef33-8ce1-4da5-ab6e-bad906ae0a34.png)  
> AdaBoost  
> ![image](https://user-images.githubusercontent.com/71176581/191989958-c1f962d8-7a06-4fa3-af3d-37b44a618100.png)  
> LightGBM  
![image](https://user-images.githubusercontent.com/71176581/191990363-3c4c92cc-9d8a-4436-9e82-9a47006204ba.png)  

* 성능 비교 
train.csv에 유가, 날씨 요소를 추가한 데이터를 이용해 RandomForest, AdaBoost, LightGBM 별 5번의 교차검증을 통해 생성된 RMSE값과 평균값을 구한다.  (train.csv에서의 요소들을 통틀어 dacon이라는 명칭을 사용한다.)  
> 유가 Data Scaling 전  
> ![image](https://user-images.githubusercontent.com/71176581/192081751-7ce780aa-eaea-4138-a1fd-ac6be69d2c30.png)  
> 유가 Data Scaling 후  
> ![image](https://user-images.githubusercontent.com/71176581/192081758-4d2133d9-9739-42e5-af46-96ea04d1c43c.png)  
> 위 2개 그래프를 보면 유가 정보를 스케일링 전, 후 데이터를 이용해 생성된 RMSE가 크게 차이없는 것을 볼 수 있다. 따라서 유가 정보 스케일링 없어도 정확한 머신러닝 학습이 된다고 본다.  

* 모델 앙상블  
여러 모델들을 사용해 성능을 올려 더욱 일반화된 모델을 완성하는 기법으로, 대표적으로 Voting, Bagging, Stacking이 있다.  
> Voting  
> 다른 알고리즘 model을 조합하여 투표를 통해 결과를 도출하는 방식으로 AdaBoost, RandomForest, LightGBM 모델을 조합하여 5번의 교차검증을 통해 RMSE 평균값을 구한다.  
> ![image](https://user-images.githubusercontent.com/71176581/192081780-e5d40d7f-7dc4-40fb-a9bd-5db0d9d17e63.png)  
> Bagging (Bootstrap Aggregating)  
> 여러 개의 데이터 중첩을 허용하여 샘플 중복 생성을 통해 결과를 도출하는 방식으로 AdaBoost, RandomForest, LightGBM 각각 알고리즘을 5번 교차검증하여 RMSE 평균값을 구한다.
> ![image](https://user-images.githubusercontent.com/71176581/192081792-07f4c33b-06ab-4ab5-ae54-918b577cb2f2.png)  
> Stacking  
> 여러 모델을 기반으로 예측된 결과를 통해 meta 모델이 다시 한번 예측하는 방식으로 AdaBoost, RandomForest, LightGBM 중 1개 모델을 meta 모델로 지정하여 5번의 교차검증을 통해 RMSE 평균값을 구한다.  
> ![image](https://user-images.githubusercontent.com/71176581/192081872-3af0b689-e9a1-4672-bd11-79865c2eb1ee.png)  

* 알고리즘 간 상관관계  
교차검증에 사용한 csv파일을 이용해 AdaBoost, RandomForest, LightGBM 알고리즘 간 상관관계는 다음과 같다. 그 결과 RandomForest와 LightGBM 알고리즘은 높은 상관도를 보이고 있고 AdaBoost는 다른 알고리즘과 낮은 상관도를 보이고 있다.
> ![image](https://user-images.githubusercontent.com/71176581/192126559-ae19b6bd-7f97-44b7-994f-b5e22fbd2fe3.png)  

* 결론  
선행연구에 사용된 알고리즘들을 GridSearchCV를 통해 최적 하이퍼파라미터를 구했다. 기존 데이터에 버스 승차 인원 예측에 영향을 미칠 수 있는 요인 중 날짜별 유가와 날씨 정보를 추가하였다. 추가 데이터 중 유가가 다른 요인에 비해 수치가 커 정확한 예측 할수 없다고 생각해서 데이터 스케일링을 통해서 범위를 축소시켰다. 그러나 예상과 달리 데이터 스케일링 전후 RMSE값의 차이가 미미했고 유가 정보 스케일링 하지 않아도 정확한 학습이 된다고 본다. 동일한 입력 데이터에서 RandomForest, AdaBoost, LightGBM 마다 성능을 비교하기 위해서 RMSE, 알고리즘 간의 상관계수를 비교했다. 각각 알고리즘에서 가장 RMSE 성능이 좋았던 입력 데이터는 RandomForest - dacon, AdaBoost - dacon+oil, LightGBM - dacon+weather 이였다. 또한 여러 모델들을 사용해 성능을 올리기 위해 모델 앙상블을 이용한다. 대표적으로 Voting, Bagging, Stacking이 있다. Voting을 사용했을 때, RandomForest, LightGBM을 이용해 교차검증한 값이 가장 좋았다. Baggigng을 사용했을 때, LightGBM을 이용해 교차검증한 값이 가장 좋았고 심지어 이전 교차검증과 비교해서도 가장 좋은 성능을 보이고 있다. Stacking은 성능이 좋지 않다. RandomForest, AdaBoost, LightGBM간 
