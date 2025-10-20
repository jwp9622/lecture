<<<<<<< HEAD
## - https://shorturl.at/SduTO
## 데이터
## - https://shorturl.at/toAqE
## - https://shorturl.at/4LiWF
---
# 📘 **3부. 스마트 헬스 개요 및 기반 기술**

# 📗 **1장. 스마트 헬스란 무엇인가**


## 🔹 1. 스마트 헬스의 정의 및 등장 배경

스마트 헬스(Smart Health)란 정보통신기술(ICT), 인공지능(AI), IoT 센서, 빅데이터 등을 활용하여 개인 맞춤형 건강 관리와 예방 중심의 헬스케어 서비스를 제공하는 패러다임입니다. 기존의 병원 중심 치료 위주에서 벗어나, **언제 어디서나 건강 정보를 수집·분석하고, 실시간으로 피드백을 제공**함으로써 예방과 자기 주도 건강관리를 가능하게 합니다.

### ▪ 주요 배경

* **고령화 사회의 도래**: 만성질환 증가와 의료비 부담 증가
* **ICT 기술 발전**: IoT 센서, 웨어러블 디바이스 보급
* **데이터 기반 의료**: EHR, PHR, 웨어러블 로그 등 다양한 의료 데이터
* **의료 자원의 불균형**: 지역 간 의료 접근성 격차 해소 필요성

---

## 🔹 2. 기존 헬스케어와 스마트 헬스의 비교

| 구분     | 기존 헬스케어      | 스마트 헬스                  |
| ------ | ------------ | ----------------------- |
| 중심 방식  | 병원 방문, 진료 중심 | 비대면, 실시간 데이터 기반         |
| 사용자 역할 | 수동적(의사 주도)   | 능동적(사용자 스스로 건강 관리)      |
| 데이터 수집 | 병원 내 검사      | 센서, 웨어러블, 스마트폰 등 실시간 수집 |
| 기술 활용  | 제한적 (EMR 등)  | IoT, AI, 빅데이터, 클라우드 활용  |
| 목적     | 질병 진단 및 치료   | 예방, 조기 발견, 건강 증진        |

---

## 🔹 3. 스마트 헬스 구성 요소

스마트 헬스는 여러 기술과 시스템이 통합적으로 작동하여 개인의 건강을 관리합니다. 구성 요소는 다음과 같습니다.

| 구성 요소            | 설명                                  |
| ---------------- | ----------------------------------- |
| **IoT 센서/디바이스**  | 심박수, 운동량, 수면 등을 실시간 측정하는 웨어러블 기기    |
| **데이터 수집 및 통신**  | BLE, WiFi, NB-IoT 등 다양한 방식으로 데이터 전송 |
| **헬스 데이터 플랫폼**   | 수집된 데이터를 통합 저장, 분석 가능한 환경           |
| **AI/ML 분석 기술**  | 건강 상태 예측, 이상 징후 감지 등 지능형 분석 수행      |
| **피드백 및 알림 시스템** | 사용자 맞춤형 건강 리포트, 경고 메시지 제공           |

---

## 🔹 4. 스마트 헬스 기술 발전 연표

| 연도        | 주요 발전 내용                               |
| --------- | -------------------------------------- |
| 2010년대 초반 | 웨어러블 기기(예: Fitbit, Jawbone) 상용화        |
| 2014년     | 애플 헬스킷(HealthKit), 구글 핏(Google Fit) 출시 |
| 2016년     | 딥러닝 기반 피부병, 안저 분석 기술 상용화               |
| 2020년 이후  | 코로나19 대응 비대면 진료, 스마트 병원 확산             |
| 2023년 이후  | 생체신호 기반 스트레스, 수면, 심혈관 리스크 예측 솔루션 등장    |

---

# 📗 **2장. 스마트 헬스를 위한 데이터 이해**



## 🔹 1. 헬스케어 데이터의 유형

스마트 헬스에서 활용되는 데이터는 매우 다양하며, 각기 다른 형식과 특성을 가집니다. 대표적인 헬스케어 데이터는 다음과 같습니다.

| 데이터 유형                              | 설명                      | 예시                           |
| ----------------------------------- | ----------------------- | ---------------------------- |
| **EMR (Electronic Medical Record)** | 병원 내에서 수집되는 환자 진료 기록    | 진단코드, 투약 내역, 검사 결과 등         |
| **EHR (Electronic Health Record)**  | 여러 기관 간 공유 가능한 의료기록     | EMR + 생활습관, 백신, 영상 등         |
| **PHR (Personal Health Record)**    | 개인이 수집/관리하는 건강 데이터      | 스마트워치, 앱 기반 자가 기록 등          |
| **생체신호 (Biosignals)**               | 신체 기능에서 측정되는 전기적/물리적 신호 | ECG, PPG, EMG, EEG, 체온, 호흡 등 |
| **행동 및 환경 데이터**                     | 사용자의 운동, 수면, 위치, 날씨 등   | 걸음 수, 수면 시간, GPS, 온습도 등      |

---

## 🔹 2. 대표적인 생체신호와 특징

스마트 헬스에서는 다양한 \*\*생체신호(Biosignal)\*\*를 분석하여 건강 상태를 추정합니다. 대표적인 신호들은 다음과 같습니다.

| 신호              | 측정 대상      | 주요 활용            | 특징                 |
| --------------- | ---------- | ---------------- | ------------------ |
| **ECG (심전도)**   | 심장 전기 신호   | 심박수, HRV, 부정맥 진단 | 고해상도, R-peak 검출    |
| **PPG (광용적맥파)** | 혈류 변화      | 맥박수, 혈중 산소포화도    | 착용 간편, 운동 시 노이즈 민감 |
| **EEG (뇌파)**    | 뇌의 전기활동    | 수면 분석, 발작 감지     | 채널 수 많고 처리 복잡      |
| **EMG (근전도)**   | 근육 수축      | 근피로도 분석, 재활치료    | 짧은 시간 신호, 잡음 영향 큼  |
| **호흡/체온**       | 호흡률, 체온 변화 | 호흡기질환, 발열 감지     | 환경 온도에 영향 받을 수 있음  |

> ECG: [위키백과 심전도](https://ko.wikipedia.org/wiki/%EC%8B%AC%EC%A0%84%EB%8F%84)
> PPG: [LED로 심박수를 측정한다고? '광혈류측정 센서(PPG)'](https://news.samsungdisplay.com/30140)
> EEG: [위키백과 뇌파](https://ko.wikipedia.org/wiki/%EB%87%8C%ED%8C%8C)
> EMG: [위키백과 근전도 검사](https://ko.wikipedia.org/wiki/%EA%B7%BC%EC%A0%84%EB%8F%84_%EA%B2%80%EC%82%AC)
> 호흡 센서: [호흡분석기 'PACER'](https://blog.naver.com/geekstarter/223752501610)
---


## 🔹 3. 웨어러블 헬스 센서의 개요

웨어러블 헬스 센서는 사용자의 생체신호 또는 행동 정보를 실시간으로 측정하고 기록하는 IoT 기반 장치입니다. 손목, 가슴, 귀, 발목, 피부 등에 부착되어 동작하며, 스마트 헬스의 핵심 데이터 수집 도구로 활용됩니다.

| 센서 형태 | 예시 기기                       | 측정 정보            |
| ----- | --------------------------- | ---------------- |
| 손목형   | Apple Watch, Galaxy Watch   | 심박수, 운동량, 수면, 체온 |
| 패치형   | Zephyr BioPatch, VitalPatch | ECG, PPG, 호흡, 체온 |
| 귀걸이형  | Earin, Cosinuss One         | 심박수, 체온          |
| 반지형   | Oura Ring                   | HRV, 수면 단계       |
| 의류형   | Hexoskin, Athos             | 호흡, EMG, 심전도     |

---

## 🔹 4. 센서의 측정 원리

| 센서 종류            | 측정 원리                    | 측정 항목               |
| ---------------- | ------------------------ | ------------------- |
| **ECG 센서**       | 피부 표면 전극을 통해 심장 전기신호 측정  | 심박수, R-R 간격, 부정맥 탐지 |
| **PPG 센서**       | 적외선/녹색광을 혈관에 조사하여 반사광 측정 | 맥박수, 혈중 산소포화도       |
| **IMU (관성측정센서)** | 가속도계와 자이로스코프 기반          | 걸음 수, 자세, 활동 인식     |
| **체온 센서**        | 서미스터, 적외선 측정             | 피부 온도, 중심 체온 추정     |
| **호흡 센서**        | 압력 변화, 스트레인 게이지 활용       | 호흡률, 폐활량 추정         |

---

## 🔹 5. 데이터 전송 및 통신 기술

웨어러블 센서는 수집된 데이터를 스마트폰 또는 클라우드로 전송하기 위해 다양한 통신 기술을 사용합니다.

| 통신 기술                          | 특징               | 적용 사례               |
| ------------------------------ | ---------------- | ------------------- |
| **BLE (Bluetooth Low Energy)** | 짧은 거리, 저전력       | 스마트워치 ↔ 스마트폰        |
| **Wi-Fi**                      | 빠른 속도, 전력 소모 큼   | 스마트 체중계 ↔ 가정용 Wi-Fi |
| **NB-IoT / LTE-M**             | 저전력, 장거리, 셀룰러 기반 | 병원 서버로 데이터 전송       |
| **ZigBee**                     | 저전력, 다수 센서 연결    | 실내용 건강 모니터링 시스템     |
| **UWB (초광대역)**                 | 위치 정확도 높음        | 실내 환자 추적, 낙상 감지     |

---

## 🔹 4. 웨어러블 센서의 데이터 특성

* **연속성**: 실시간 연속 측정으로 시계열 데이터 생성
* **노이즈 포함**: 움직임, 피부 접촉 불량 등으로 인한 잡음 존재
* **사용자 간 다양성**: 생리적 차이로 인해 개인별 기준 상이
* **전력 소모 고려 필요**: 센서 설계 및 수집 주기 최적화 필요

---

## 🔹 6. 웨어러블 센서 선택 시 고려 요소

| 고려 항목       | 설명                         |
| ----------- | -------------------------- |
| **정확도**     | 의료 기준 충족 여부 (예: FDA 인증 여부) |
| **배터리 수명**  | 지속적인 측정 가능 시간              |
| **편의성**     | 사용자의 착용감, 위치 제한            |
| **통신 방식**   | 사용 환경에 맞는 연결성              |
| **데이터 접근성** | API 제공 여부, 데이터 내보내기 가능성    |

---

# 📗 **4장. 스마트 헬스 서비스 사례**



## 🔹 1. 스마트 헬스 서비스의 분류

스마트 헬스 서비스는 제공 주체와 기술 방식에 따라 다음과 같이 구분할 수 있습니다.

| 유형            | 설명                 | 예시                           |
| ------------- | ------------------ | ---------------------------- |
| **개인 건강 관리형** | 웨어러블 기반 실시간 건강 관리  | Apple Health, Samsung Health |
| **질병 예측/진단형** | AI 기반 조기 진단, 위험 예측 | SkinVision, Lunit INSIGHT    |
| **원격 모니터링형**  | 병원과 환자 간 연결, 지속 추적 | Livongo, Dexcom              |
| **스마트 병원형**   | 병원 내 디지털 시스템 통합    | 세브란스 스마트 병원, Mayo Clinic     |

---

## 🔹 2. 스마트 헬스 서비스 사례

### (1) **Apple Health [(애플 헬스)](https://www.apple.com/health/)**

* **기능**: 심박수, 운동량, 수면 기록, 심방세동 감지
* **센서**: Apple Watch (ECG, PPG, IMU 등 내장)
* **분석**: iOS 기반의 건강 앱에서 시각화
* **특징**: EHR 연동, 미국 내 일부 병원과 직접 연결 가능

### (2) **Fitbit [(by Google)](https://store.google.com/gb/category/watches_trackers?hl=en-GB)**

* **기능**: 운동, 수면, 스트레스 추적
* **AI 기술**: 수면 점수 계산, HRV 기반 스트레스 지수
* **데이터 통합**: Fitbit 앱 + Google Health 통합 플랫폼
* **특징**: FDA 승인 ECG 기능 제공

### (3) **SkinVision [(네덜란드)](https://www.skinvision.com/)**

* **기능**: 피부암 위험도 자가 진단
* **기술**: 스마트폰 카메라 기반 CNN 피부 분석
* **성과**: 흑색종 조기 발견 정확도 95% 이상
* **활용**: 사용자가 주기적 사진 촬영 → AI 분석 결과 확인


### (4) **삼성 헬스 [(Samsung Health)](https://www.samsung.com/sec/apps/samsung-health/)**

* **기능**: 걸음 수, 수면, 스트레스, 혈중 산소포화도 측정
* **센서 연동**: Galaxy Watch 시리즈
* **분석**: HRV 기반 스트레스 추정, 수면 단계 자동 분석
* **특징**: 삼성 스마트폰과 자동 연동, 글로벌 1억 이상 사용자

---

## 🔹 3. 스마트 헬스 서비스 설계 시 고려사항

| 항목             | 고려 요소                     |
| -------------- | ------------------------- |
| **데이터 정확성**    | 의료기기 수준 인증 필요 (FDA, CE 등) |
| **개인화 수준**     | 연령, 성별, 상태별 맞춤형 알고리즘      |
| **연동성**        | EHR, 병원 시스템, 모바일 앱 연계 가능성 |
| **설명 가능성**     | AI 결과의 근거 제시 여부           |
| **보안 및 프라이버시** | 생체정보 암호화, GDPR/개인정보보호법 준수 |

---

# 4부: 머신러닝의 이해와 활용

---

# 📖 **1장. 머신러닝 개요**



## ✨ 1. 머신러닝이란 무엇인가

머신러닝(Machine Learning)이란,  
명시적으로 프로그램을 작성하지 않고도 데이터를 이용하여 컴퓨터가 스스로 학습하고 성능을 개선하는 기술입니다.

Arthur Samuel은 머신러닝을 "**명시적으로 프로그래밍하지 않고 컴퓨터가 학습할 수 있게 하는 연구 분야**"라고 정의하였습니다.

머신러닝은 입력 데이터로부터 패턴을 학습하여,  
**새로운 데이터에 대해 예측하거나 분류하는 모델**을 구축하는 것을 목표로 합니다.

### ➡️ 머신러닝 기본 구성 요소
| 구성 요소 | 설명 |
|:--|:--|
| 데이터 | 학습 및 예측을 위한 입력 자료 |
| 모델 | 데이터의 패턴을 학습하는 구조 |
| 학습 알고리즘 | 모델을 최적화하는 방법 |
| 예측 | 학습한 모델로 새로운 데이터에 대한 결과 생성 |

## ✨ 2. 머신러닝의 주요 분류

머신러닝은 학습 방식에 따라 크게 세 가지로 분류됩니다.

| 분류 | 설명 | 예시 |
|:--|:--|:--|
| 지도학습 (Supervised Learning) | 입력과 정답(label)을 이용하여 학습 | 분류(Classification), 회귀(Regression) |
| 비지도학습 (Unsupervised Learning) | 정답 없이 데이터 구조를 학습 | 클러스터링(Clustering), 차원 축소(Dimensionality Reduction) |
| 강화학습 (Reinforcement Learning) | 보상을 통해 최적 행동을 학습 | 게임 플레이, 로봇 제어 |


---

## ✨ 3. 머신러닝 워크플로우

머신러닝 프로젝트는 다음과 같은 단계를 거칩니다.

### ➡️ 전형적인 머신러닝 워크플로우

1. **문제 정의**
2. **데이터 수집**
3. **데이터 전처리 및 탐색**
4. **특성 선택 및 생성**
5. **모델 선택 및 학습**
6. **모델 평가**
7. **모델 개선 및 최적화**
8. **최종 모델 배포**

```
데이터 수집 → 데이터 전처리 → 모델 학습 → 모델 평가 → 하이퍼파라미터 튜닝 → 최종 예측
```

- 이 과정에서 **데이터 전처리와 모델 선택**이 성능에 가장 큰 영향을 미칩니다.
---

# 📖 **2장. 데이터 전처리와 특성 공학**



## ✨ 1. 결측치 처리와 이상값 탐지

### ➡️ 결측치(Missing Value) 처리

머신러닝 모델은 결측값을 포함하는 데이터를 직접 다루지 못하는 경우가 많습니다.  
따라서 적절한 방법으로 결측값을 처리해야 합니다.

| 방법 | 설명 | 예시 |
|:--|:--|:--|
| 삭제 (Drop) | 결측치가 있는 행 또는 열을 제거 | `dropna()` |
| 대체 (Imputation) | 평균, 중앙값, 최빈값 등으로 대체 | `fillna(value)` |

> ✅ 데이터의 분포를 고려하여 대체하는 것이 중요합니다.

### ➡️ 이상값(Outlier) 탐지

이상값은 데이터 분포에서 벗어난 값입니다.  
주로 IQR, Z-Score 등을 활용하여 탐지합니다.

| 방법 | 설명 |
|:--|:--|
| IQR 방법 | Q1, Q3 기준으로 이상 범위 외 값 탐지 |
| Z-Score 방법 | 평균 대비 표준편차 범위를 벗어난 값 탐지 |

---

## ✨ 2. 범주형 변수 인코딩

머신러닝 모델은 수치형 데이터만 입력으로 받기 때문에,  
범주형(categorical) 데이터는 수치로 변환해야 합니다.

| 인코딩 방법 | 설명 | 예시 |
|:--|:--|:--|
| 레이블 인코딩 (Label Encoding) | 각 카테고리를 정수로 매핑 | Male → 0, Female → 1 |
| 원-핫 인코딩 (One-Hot Encoding) | 각 카테고리를 0/1로 변환하는 벡터 생성 | `get_dummies(), to_categorical()` 사용 |

> ✅ 베이지안 룰, 트리 기반 모델은 레이블 인코딩을 사용해도 괜찮지만,  
> ✅ 선형 모델(SVM, 로지스틱 회귀 등)에서는 원-핫 인코딩이 더 적합합니다.

---

## ✨ 3. 정규화와 표준화

특성(feature)들의 스케일이 다를 경우,  
머신러닝 모델은 특정 특성에 지나치게 민감하게 반응할 수 있습니다.  
이를 방지하기 위해 스케일 조정이 필요합니다.

| 방법 | 설명 | 수식 |
|:--|:--|:--|
| 정규화 (Normalization) | 모든 값을 0~1 범위로 변환 | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ |
| 표준화 (Standardization) | 평균 0, 표준편차 1로 변환 | $x' = \frac{x - \mu}{\sigma}$ |

> ✅ KNN, SVM 같은 거리 기반 알고리즘은 정규화/표준화가 매우 중요합니다.

---

## ✨ 4. 특성 선택과 차원 축소

모든 특성이 예측에 유용한 것은 아닙니다.  
**특성 선택(Feature Selection)** 은 중요한 특성만 골라 모델 성능을 향상시키는 과정입니다.

| 방법 | 설명 |
|:--|:--|
| 필터 방식 (Filter) | 통계적 기준(상관계수 등)으로 선택 |
| 래퍼 방식 (Wrapper) | 모델을 통해 최적 특성 조합 탐색 |
| 임베디드 방식 (Embedded) | 모델 학습 중 특성 선택 (ex. Lasso) |

또한 고차원 문제를 해결하기 위해  
**차원 축소(Dimensionality Reduction)** 기법(PCA 등)을 사용할 수 있습니다.

---

## ✨ 5. 데이터 분할: 학습/검증/테스트셋

모델의 성능을 정확히 평가하기 위해, 데이터를 다음과 같이 분할합니다.

| 구분 | 설명 | 일반적인 비율 |
|:--|:--|:--|
| 학습셋 (Training Set) | 모델을 학습시키는 데이터 | 60~80% |
| 검증셋 (Validation Set) | 하이퍼파라미터 튜닝용 데이터 | 10~20% |
| 테스트셋 (Test Set) | 최종 성능 평가용 데이터 | 10~20% |

> ✅ 교차 검증(Cross Validation) 기법을 사용하면 보다 안정적인 평가가 가능합니다.

---

## 🛠️ 실습: 타이타닉 데이터 전처리 

```python
# 타이타닉 데이터 전처리

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print(titanic.info())
print(titanic.head(10))

# 1. 결측치 처리
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
print(titanic.info())
print(titanic.head(10))

# 2. 범주형 변수 인코딩
print(titanic['Sex'].value_counts())
print(titanic['Embarked'].value_counts())
titanic['Sex'] = titanic['Sex'].map({'female': 0, 'male': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C':0, 'S':1, 'Q':2})
print(titanic.head(10))

# 3. 불필요한 열 제거
titanic.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
print(titanic.head(10))

# 4. 특성과 타깃 분리
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 정규화 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("타이타닉 데이터 전처리 완료!")
print("훈련 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
```
---


# 📘 3장. 회귀 분석(Regression)

회귀 분석은 **연속적인 값을 예측**할 때 사용하는 대표적인 기계학습 알고리즘입니다.

| 항목      | 설명                                                |
| ------- | ------------------------------------------------- |
| 목적      | 입력 변수로부터 수치형 출력값 예측                               |
| 예시      | 기온, 습도 → 작물 생장량, 수확량 예측                           |
| 대표 알고리즘 | 선형 회귀(Linear Regression), 릿지 회귀(Ridge), 라쏘(Lasso) |

### 적용 예

* 기상 데이터 기반 **수확량 예측**
* 이산화탄소 농도에 따른 **생육 속도 예측**
---

# 🔹 선형 회귀 (Linear Regression)



## 1. 역사적 배경


#### 1.. 기원

* **1805년**: 프랑스 수학자 **Adrien-Marie Legendre**가 ‘최소제곱법(Least Squares Method)’을 도입하여 천문학적 관측 데이터의 오차를 최소화하려 했습니다.
* **1809년**: 독일 수학자 **Carl Friedrich Gauss**도 동일한 방법을 독자적으로 개발하여 **정규분포와의 연결**을 이론화하였습니다.

#### 2.. 통계학으로의 확장

* **19세기 후반**: 통계학자 **Francis Galton**이 인간 키의 상관관계 분석을 통해 ‘회귀(regression)’라는 용어를 도입하였습니다.

  * 아버지 키와 아들 키 간의 관계에서 평균으로 되돌아가는 성질 → “회귀(regression) toward the mean”

#### 3.. 컴퓨터 시대 이후

* 20세기 중반 이후 회귀 분석은 컴퓨터를 통해 자동화되며 통계학, 경제학, 공학, 생물학 등 다양한 분야에서 **기초 예측 도구**로 자리잡게 됩니다.

---

## 2. 이론적 알고리즘 

#### 1.. 목표
입력 변수 $X$와 출력 변수 $y$ 사이의 **선형 관계**를 모델링하여, 새로운 입력 값에 대한 **연속적 출력 값**을 예측하는 것.

#### 2.. 수식 표현

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{Xw} + b
$$

* $\hat{y}$: 예측값
* $\mathbf{X}$: 입력 벡터
* $\mathbf{w}$: 가중치 벡터
* $b$: 절편(intercept)

#### 3.. 목적 함수 (비용 함수, Loss Function)

\*\*MSE (Mean Squared Error)\*\*를 사용하여 오차를 최소화합니다:

$$
J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 3. 가정 조건

선형 회귀는 다음의 통계적 가정을 전제로 합니다:

| 가정                              | 설명                    |
| ------------------------------- | --------------------- |
| 선형성 (Linearity)                 | 입력 변수와 출력 변수 간의 선형 관계 |
| 독립성 (Independence)              | 관측값 간의 독립성            |
| 등분산성 (Homoscedasticity)         | 오차의 분산이 일정함           |
| 정규성 (Normality)                 | 잔차(오차)가 정규 분포를 따름     |
| 다중공선성 없음 (No multicollinearity) | 입력 변수 간 과도한 상관관계 없음   |

---

## 4. 회귀분석 프로젝트 사례

### 🔹 1. 작물 생장 예측 

회귀 분석은 작물의 생장량(예: 잎 면적, 키, 생중량 등)을 예측하기 위해 주로 사용됩니다. 입력 변수로는 온도, 습도, CO₂, 일조량 등의 **환경 정보**가 사용되며, 출력은 **생육 지표의 연속적 수치**입니다.

#### 회귀 분석 흐름

1. 데이터 수집: 시계열 센서 + 생육 측정 데이터
2. 특성 선택: 평균 온도, 누적 일조량 등
3. 모델 학습: 선형 회귀 또는 다항 회귀
4. 성능 평가: MAE, RMSE, R²

---

### 🔹 2. 날씨 데이터 기반 수확 시기 예측

수확 시기 예측은 생육률과 외부 환경 요인을 기반으로 작물의 **최적 수확 시점**을 분류하거나 회귀 문제로 모델링합니다.

| 입력 변수 | 설명            |
| ----- | ------------- |
| 생육일수  | 파종 후 경과 일수    |
| 평균 기온 | 생육기 평균 기온     |
| 총 일조량 | 누적 일조 시간      |
| 강수량   | 수분 스트레스 영향 고려 |

####  예측 접근 방식

* **회귀 모델**: 수확까지 남은 일수 예측
* **이진 분류 모델**: ‘수확 적기 여부’ (예/아니오)


---

## 실습 1: 다중 선형 회귀를 활용한 보스턴 주택 가격 예측



**Boston Housing 데이터셋**은 미국 보스턴 지역의 주택 가격에 영향을 주는 다양한 변수들(방 수, 지역 범죄율, 교육 수준 등)을 포함한 데이터입니다.
다중 선형 회귀 모델을 학습시켜, **주어진 특성들로부터 집값을 예측**하는 모델을 구현합니다.
> OpenML에서 보스턴 주택 데이터셋을 가져옴.


### 데이터 정보

Boston Housing 데이터셋에는 다음과 같은 특성(Feature)이 포함되어 있습니다:

| 변수명      | 설명                        |
| -------- | ------------------------- |
| CRIM     | 지역 범죄율                    |
| ZN       | 25,000 평방피트 이상 거주지역 비율    |
| INDUS    | 비소매상업지역 면적 비율             |
| CHAS     | 찰스강 인접 여부 (1: 인접, 0: 그 외) |
| NOX      | 일산화질소 농도                  |
| RM       | 주택 1가구당 평균 방 개수           |
| AGE      | 1940년 이전에 지어진 주택 비율       |
| DIS      | 5개 보스턴 고용센터까지 거리          |
| RAD      | 방사형 고속도로 접근성 지수           |
| TAX      | 재산세율                      |
| PTRATIO  | 학생-교사 비율                  |
| B        | 흑인 인구 비율 지표               |
| LSTAT    | 저소득층 비율 (%)               |
| **MEDV** | 주택 가격 (목표값, 단위: \$1000s)  |



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml

# 1. 데이터 로딩
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data
y = boston.target

print(X.info())

# categorical 변수 --> 수치형으로 변경
X['CHAS'] = X['CHAS'].astype(int)
X['RAD'] = X['RAD'].astype(int)

# 2. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ 모델 평가")
print(f"▶ MSE: {mse:.2f}")
print(f"▶ RMSE: {rmse:.2f}")
print(f"▶ R² Score: {r2:.2f}")

# 4. 회귀 계수 출력 및 해석
coefficients = pd.Series(model.coef_, index=X.columns)

print("\n✅ 회귀 계수:")
print(coefficients.sort_values(ascending=False))

print("\n✅ 주요 변수 해석:")
print(f"NOX (일산화질소 농도): {coefficients['NOX']:.3f} → 일산화질소 농도 높을수록 집값 하락")
print(f"RM (방 개수): {coefficients['RM']:.3f} → 방이 많을수록 집값 상승")
print(f"CHAS (찰스강 인접 여부): {coefficients['CHAS']:.3f} → 찰스강 인접할수록 집값 상승")
```
---
## 실습 2: GreenHouse 수확량 예측

### 데이터 셋
- Autonomous Greenhouse Challenge (AGC) 데이터셋
- 링크: https://www.kaggle.com/datasets/piantic/autonomous-greenhouse-challengeagc-2nd-2019/data
- **Autonomous Greenhouse Challenge (AGC)** 데이터셋에는 여러 폴더가 존재하는데, 각각이 의미하는 바는 다음과 같습니다:


### 폴더 구조 및 설명

| 폴더명            | 설명                                                                                                             |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| **AICU**       | 2019–2020년 챌린지에 참가한 **AiCU 팀**이 사용한 온실 데이터를 포함합니다. 센서 입력 값(온도, 습도, CO₂, PAR, 토양 수분 등)과 AI 기반 제어 전략이 기록되어 있습니다. |
| **Automatoes** | **Automatoes 팀** (2등 또는 우승 팀)의 데이터를 나타냅니다. 이번 대회에서 토마토 재배에 사용된 환경 조작 전략과 해당 센서 데이터를 포함합니다.                     |
| **Dialog**     | 데이터셋 제공 구조상 **Dialog**는 챌린지의 “참가자–주최 측 간 대화 데이터”일 가능성이 높습니다. 실제 환경 센서 값이나 실험 데이터를 담고 있지는 않을 확률이 큽니다.           |

### 사용 폴더:  **Automatoes** 

### 사용 파일 목록

| 파일명                     | 설명                          | 사용 목적         |
| ----------------------- | --------------------------- | ------------- |
| `GreenhouseClimate.csv` | 온실 내부 온도, 습도, CO₂, 광량, 급수 등 | **입력 변수 (X)** |
| `Production.csv`        | 품질 등급별 수확량 (A, B), 수확 일자 포함 | **목표 변수 (y)** |


### 선택된 주요 변수

| 변수명       | 설명                          | 단위        | 파일                |
| --------- | --------------------------- | --------- | ----------------- |
| `Tair`    | 온실 내 공기 온도                  | °C        | GreenhouseClimate |
| `Rhair`   | 온실 내 상대 습도                  | %         | GreenhouseClimate |
| `CO2air`  | 온실 내 CO₂ 농도                 | ppm       | GreenhouseClimate |
| `Tot_PAR` | 총 광합성 유효 복사량 (태양 + LED/HPS) | μmol/m²/s | GreenhouseClimate |
| `Cum_irr` | 하루 누적 관수량                   | L/m²      | GreenhouseClimate |
| `ProdA`   | 상급 품질 토마토 수확량               | kg/m²     | Production        |

※ 시간 단위는 5분 간격이며, 일 단위로 리샘플링 후 사용합니다.


* 먼저, AICU 폴더의 GreenhouseClimate.csv와 Production.csv 를 업로드합니다. 

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 파일 경로 설정
climate_path = "/content/GreenhouseClimate.csv"
prod_path = "/content/Production.csv"

# 2. 데이터 불러오기
climate = pd.read_csv(climate_path)
climate['Time'] = pd.to_datetime(climate['Time'], unit='D', origin='1900-01-01')

production = pd.read_csv(prod_path)
production['Time'] = pd.to_datetime(production['Time'], unit='D', origin='1900-01-01')

# 3. 필요한 변수만 추출
climate = climate[['Time', 'Tair', 'Rhair', 'CO2air', 'Tot_PAR', 'Cum_irr']]
production = production[['Time', 'ProdA']]  # 목표: Class A 수확량

# 4. 시간 단위 평균 (하루 단위로)
climate_indexed = climate.set_index('Time') 
production_indexed = production.set_index('Time')

numerical_cols = ['Tair', 'Rhair', 'CO2air', 'Tot_PAR', 'Cum_irr']
for col in numerical_cols:
    climate_indexed[col] = pd.to_numeric(climate_indexed[col], errors='coerce')

climate_daily = climate_indexed[numerical_cols].resample('D').mean().reset_index()
production_daily = production_indexed.resample('D').sum().reset_index()

# 5. 병합
df = pd.merge(climate_daily, production_daily, on='Time')

# 6. 결측치 제거
df.dropna(inplace=True)

# 7. X, y 분리
X = df[numerical_cols]
y = df['ProdA']

# 8. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. 모델 훈련 (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 11. 예측 및 평가
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 12. 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Ground Truth')
plt.plot(y_pred, label='Predicted (LinearReg)', linestyle='--')
plt.title("Tomato Production Prediction (Linear Regression)")
plt.xlabel("Sample Index")
plt.ylabel("Production (kg/m²)")
plt.legend()
plt.grid()
plt.show()
```


---
# 🔹 로지스틱 회귀(Logistic Regression)


## 1. 로지스틱 회귀란?


로지스틱 회귀는 **분류(classification)** 문제를 해결하는 지도 학습 알고리즘입니다. 이름은 "회귀"이지만, 실제로는 **출력값을 확률로 예측하고, 이 확률을 바탕으로 클래스(0 또는 1)를 분류**하는 데 사용됩니다.


* 출력값은 **0과 1 사이의 확률**로 표현됩니다.
* **시그모이드 함수**를 사용해 선형 조합을 확률로 변환합니다.
* **이진 분류**에 주로 사용되며, **다중 클래스 확장**도 가능합니다.



| 항목    | 설명                                      |
| ----- | --------------------------------------- |
| 목적    | 확률 기반 이진 분류                             |
| 예측값   | 시그모이드 함수 출력 $\hat{y} \in (0,1)$         |
| 결정 기준 | $\hat{y} \ge 0.5 \rightarrow 1$, 그 외는 0 |
| 손실 함수 | 로지스틱 손실 (Binary Cross-Entropy)          |
| 최적화   | 경사하강법 등 사용                              |
| 확장    | Scikit-learn은 다중 클래스 분류 가능                |

---

### 문제 1. 로지스틱 회귀에서 예측값 $\hat{y}$는 어떤 범위를 가지는가?

① 0 또는 1
② 음수에서 양수
③ 0 이상 1 이하의 실수
④ 정수

**정답**: ③ 0 이상 1 이하의 실수
**해설**: 시그모이드 함수의 출력값은 항상 **(0, 1)** 사이의 **확률값**입니다.


### 문제 2. 로지스틱 회귀에서 시그모이드 함수의 역할은?

① 특성 선택
② 정규화
③ 선형 조합을 확률값으로 변환
④ 가중치 감소

**정답**: ③ 선형 조합을 확률값으로 변환
**해설**: $\sigma(z) = \frac{1}{1 + e^{-z}}$은 선형 조합 $z$를 0\~1 사이의 확률로 변환합니다.

---
## 2. 로지스틱 회귀 실습


### 📦 데이터셋: Ai4I 2020 Predictive Maintenance Dataset

이 데이터셋은 **스마트 제조 환경에서의 예측 유지보수**를 위한 시뮬레이션 데이터로, 센서 측정값과 기계 고장 여부를 포함합니다. 공개적으로 접근 가능한 이 데이터셋은 다양한 연구에 활용되고 있습니다.


* **출처**: UCI Machine Learning Repository
* **데이터 형태**: CSV (약 10,000개 샘플, 14개 열)
* **목적**: 공장 설비에서 발생할 수 있는 \*\*기계 고장(Machine Failure)\*\*을 예측하기 위한 **이진 분류(binary classification)** 문제입니다.



#### 전체 컬럼 설명표

| 변수명 (컬럼명)                 | 데이터 유형     | 설명                                    |
| ------------------------- | ---------- | ------------------------------------- |
| `UDI`                     | 정수 (int)   | 고유 식별자 (Unique ID)                    |
| `Product ID`              | 문자열 (str)  | 제품 고유 식별자                             |
| `Type`                    | 문자열 (str)  | 제품 유형 (L, M, H 세 가지 타입)               |
| `Air temperature [K]`     | 실수 (float) | 공기 온도 (켈빈 단위)                         |
| `Process temperature [K]` | 실수 (float) | 공정 온도 (켈빈 단위)                         |
| `Rotational speed [rpm]`  | 실수 (float) | 회전 속도 (분당 회전수, rpm)                   |
| `Torque [Nm]`             | 실수 (float) | 토크 (Nm)                               |
| `Tool wear [min]`         | 실수 (float) | 공구 마모 시간 (분)                          |
| `TWF`                     | 0/1 (int)  | Tool Wear Failure (공구 마모 고장 여부)       |
| `HDF`                     | 0/1 (int)  | Heat Dissipation Failure (열 방출 고장 여부) |
| `PWF`                     | 0/1 (int)  | Power Failure (전력 고장 여부)              |
| `OSF`                     | 0/1 (int)  | Overstrain Failure (과부하 고장 여부)        |
| `RNF`                     | 0/1 (int)  | Random Failures (임의 고장 여부)            |
| `Machine failure`         | 0/1 (int)  | **최종 고장 발생 여부 (Target 변수)**           |

---

#### 타겟 변수 (Machine failure)

* `Machine failure`는 위의 TWF, HDF, PWF, OSF, RNF 다섯 개의 고장 중 **하나라도 발생하면 1**, 그렇지 않으면 0입니다.

$$
\text{Machine failure} = \begin{cases}
1 & \text{(TWF or HDF or PWF or OSF or RNF = 1)} \\
0 & \text{(모두 0)}
\end{cases}
$$


#### 예시 행 데이터

| UDI | Product ID | Type | Air Temp \[K] | Proc Temp \[K] | Rot Speed | Torque | Tool Wear | TWF | HDF | PWF | OSF | RNF | Machine failure |
| --- | ---------- | ---- | ------------- | -------------- | --------- | ------ | --------- | --- | --- | --- | --- | --- | --------------- |
| 1   | M14860     | M    | 298.1         | 308.6          | 1551      | 42.8   | 0         | 0   | 0   | 0   | 0   | 0   | 0               |


#### 주요 분석 포인트

* **제품 유형(Type)**: L, M, H 간 성능 차이 존재 가능
* **온도/회전수/토크 등 물리 센서 변수**와 **고장 발생 여부** 간 관계 분석 가능
* **고장 비율이 낮아 클래스 불균형 문제가 존재할 수 있음**


#### 활용 가능 문제 유형

| 문제 유형 | 가능 여부 | 설명                            |
| ----- | ----- | ----------------------------- |
| 이진 분류 | ✅     | 머신 고장 여부 예측                   |
| 다중 분류 | ✅     | TWF, HDF, PWF 등 개별 고장 유형 분류   |
| 이상 탐지 | ✅     | 고장 발생이 매우 드물다면 이상탐지 문제로 접근 가능 |
| 회귀 문제 | ❌     | 현재는 회귀형 목표변수 없음               |

---

### 로지스틱 회귀 실습 코드

```python
# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 데이터 전처리
# 불필요한 열 제거
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)


## 범주형 변수 처리
print(df.info())
print(df['Type'].value_counts())

df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

# 타겟 변수와 특성 분리
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]' ] ]
y = df['Machine failure']

print(X.info())

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. 예측 및 평가
y_pred = model.predict(X_test_scaled)

# 분류 성능 평가
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

---
# 📘 4장. 분류 모델

---
# 📘 5장. k-최근접 이웃 알고리즘 (k-Nearest Neighbors, kNN)


## 1. 개념

\*\*k-최근접 이웃 알고리즘(kNN)\*\*은 가장 단순하면서도 강력한 **비모수(non-parametric)** 지도 학습 알고리즘 중 하나입니다.

* 주어진 입력 데이터에 대해 **가장 가까운 k개의 이웃**을 찾고,
* 그 이웃들의 레이블을 바탕으로 예측을 수행합니다.


## 2. 원리

1. **훈련 단계 (Training)**

   * 특별한 학습 과정이 없습니다. 모든 학습 데이터를 **기억**해둡니다.
   * 즉, **기억 기반 모델(Memory-based model)** 또는 **Lazy Learning**이라고 합니다.

2. **예측 단계 (Prediction)**
   새로운 데이터 포인트 $x$가 들어오면:

   * 학습 데이터 전체와의 거리를 계산합니다.
   * 거리 기준으로 가장 가까운 **k개의 데이터**를 선택합니다.
   * **분류 문제**:

     * 다수결 투표(Majority Voting)를 통해 가장 많이 등장한 클래스 선택
   * **회귀 문제**:

     * k개의 평균을 사용해 예측


### 거리 계산

가장 널리 쓰이는 거리 함수는 \*\*유클리드 거리(Euclidean distance)\*\*입니다:

$$
d(x, x_i) = \sqrt{ \sum_{j=1}^{n} (x_j - x_{ij})^2 }
$$

그 외에도:

* 맨해튼 거리: $\sum |x_j - x_{ij}|$
* 코사인 거리: $1 - \cos(\theta)$

---

## 3. 하이퍼파라미터

| 하이퍼파라미터      | 설명                                             |
| ------------ | ---------------------------------------------- |
| **k** (이웃 수) | 너무 작으면 과적합, 너무 크면 과소적합                         |
| **거리 척도**    | 유클리드, 맨해튼, 코사인 등                               |
| **가중치 방식**   | 이웃 거리에 따라 가중치를 둘 수도 있음 (`uniform`, `distance`) |

---

## 4. 장점과 단점

| 장점                    | 단점                            |
| --------------------- | ----------------------------- |
| 구현이 간단함               | 계산 비용이 큼 (예측 시 전체 데이터와 거리 계산) |
| 비모수 모델 (데이터 분포 가정 없음) | 고차원에서는 성능 저하 (차원의 저주)         |
| 다중 클래스/회귀 모두 가능       | 훈련 데이터가 많을수록 속도 느림            |


---

# 📘 6장. 서포트 벡터 머신(Support Vector Machine, SVM)

## 1. 역사적 배경

* **1992년**: 러시아 출신의 **Vladimir Vapnik**과 **Alexey Chervonenkis**가 이론적 기초인 **VC 차원**(Vapnik–Chervonenkis dimension)과 **구간 리스크 최소화** 개념을 발표했습니다.
* **1995년**: Vapnik과 Cortes가 현대적 의미의 SVM 알고리즘을 공식화함. 이 논문은 "Support-Vector Networks"라는 제목으로 발표되며, **마진 최대화**와 \*\*커널 트릭(Kernel Trick)\*\*을 통해 비선형 분류까지 확장됨.
* **2000년대**: Bioinformatics, 문서 분류, 이미지 인식 등 다양한 분야에서 **고차원 소규모 데이터**에 강한 분류기로 널리 활용됨.

## 2. SVM의 핵심 아이디어

* **마진 최대화**: 클래스를 구분하는 **결정 경계**(Decision Boundary) 중, 가장 \*\*넓은 여백(Margin)\*\*을 가지는 선형 결정 경계를 찾음.
* **서포트 벡터**: 결정 경계에 가장 가까운 데이터 포인트.
* **커널 기법**: 비선형 데이터를 고차원으로 매핑하여 선형 분리 가능하게 만드는 방법.


## 3. 수학적 수식

### 목적: 최대 마진 분리 초평면 찾기

선형 SVM의 목적은 다음과 같은 초평면을 찾는 것입니다.

$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

클래스는 다음 조건을 만족해야 합니다:

$$
y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

### 최적화 문제

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1
$$

이 문제는 **Convex Quadratic Programming** 형태이며, Lagrange multiplier 기법으로 최적화됩니다.

### 🔄 비선형 SVM (Kernel Trick)

$$
\phi(\mathbf{x})^\top \phi(\mathbf{x'}) = K(\mathbf{x}, \mathbf{x'})
$$

* 대표 커널: 선형(linear), 다항식(polynomial), RBF(Gaussian), 시그모이드(sigmoid)

---

# 📘 7장. 의사결정나무 (Decision Tree) 알고리즘

## 1. 역사적 배경

* **1960년대**: 통계학 분야에서 분류 및 회귀를 위한 트리 기반 모델이 연구되기 시작함.
* **1986년**: **J. Ross Quinlan**이 발표한 **ID3(Iterative Dichotomiser 3)** 알고리즘이 널리 주목받으며 의사결정나무 알고리즘의 표준이 됨.
* 이후 **C4.5**(1993), **CART(Classification and Regression Tree)**(Breiman et al., 1984) 등 다양한 버전이 개발됨.
* 현재는 **Random Forest**, **XGBoost** 등의 앙상블 기반 트리 모델의 기본 단위로도 사용됨.


## 2. 이론적 개념

### 기본 아이디어

의사결정나무는 데이터를 \*\*특성(feature)\*\*에 따라 분할(split)하여 **트리 형태로 분류** 또는 **예측**을 수행합니다.

* 루트 노드: 전체 데이터
* 내부 노드: 특성 기준으로 데이터를 분할
* 리프 노드: 최종 예측 결과(클래스 or 수치 값)


### 분할 기준 (지니, 엔트로피, 분산)

#### (1) **분류(Classification)** 문제:

* **지니 불순도(Gini Impurity)**

  $$
  Gini = 1 - \sum_{i=1}^n p_i^2
  $$
* **엔트로피(Entropy)**

  $$
  Entropy = - \sum_{i=1}^n p_i \log_2 p_i
  $$

→ 두 기준 모두 불순도(혼합도)를 최소화하도록 분할합니다.

#### (2) **회귀(Regression)** 문제:

* **MSE (Mean Squared Error)** 또는 **MAE**를 사용하여 분할

---

###  가지치기 (Pruning)

* **사전 가지치기 (Pre-pruning)**: 깊이 제한, 최소 노드 수 설정 등
* **사후 가지치기 (Post-pruning)**: 트리를 완성한 후 불필요한 노드를 제거하여 과적합 방지

---

## 3. 알고리즘 요약

1. 루트 노드에서 시작
2. 각 특성에 대해 최적 분할 기준을 평가 (지니, 엔트로피 등)
3. 가장 불순도를 많이 줄이는 특성으로 분할
4. 리프 노드가 될 조건(순도가 높거나 최대 깊이 도달 등)까지 반복
5. 예측은 리프 노드의 클래스(또는 평균값)로 결정

---

# 📘 8장. Random Forest 알고리즘


## 1. 역사적 배경

| 항목        | 내용                                                        |
| --------- | --------------------------------------------------------- |
| **고안자**   | 레오 브레이먼 (Leo Breiman), 2001년 발표                           |
| **논문**    | “Random Forests” (2001)                                   |
| **기반 개념** | 배깅(Bagging: Bootstrap Aggregating) + 결정 트리(Decision Tree) |
| **개발 목적** | 개별 결정트리의 과적합(overfitting)을 줄이고 예측 성능을 향상시키기 위해            |

**이전 배경**

* 1996년: Breiman이 Bagging(배깅) 기법 제안
* 이후, 여러 개의 배깅된 결정 트리에 무작위성을 더해 **Random Forest**로 확장

---

## 2. 이론적 배경

### 핵심 개념

**Random Forest는 여러 개의 Decision Tree를 훈련시킨 뒤, 각 트리의 결과를 종합하여 최종 예측을 수행하는 앙상블 학습 기법입니다.**

---

### 🔷 Bagging: Bootstrap Aggregating

* **Bootstrap**: 데이터셋에서 **중복 허용 무작위 샘플링**으로 N개의 학습용 샘플 생성
* **Aggregating**: 각각의 모델 결과를 평균(회귀) 또는 투표(분류)를 통해 종합

---

### 🔷 Randomness (무작위성) 도입

1. **데이터 샘플 무작위 선택** (Bagging)
2. **각 노드에서 특성(feature) 무작위 선택** (예: √p 개 중 최적 분할 특성 선택)

이 두 가지 무작위성으로 인해 **트리 간 상관성 감소**, 결과적으로 **앙상블 효과 향상** 및 **과적합 감소** 효과를 얻음.

---

### 🔷 알고리즘 동작 과정

1. **T개의 결정 트리 생성**

   * 각 트리는 Bootstrapping된 데이터로 학습
2. **각 트리에서 무작위 피처 서브셋 선택**
3. **각 트리의 결과 예측**

   * 분류: **다수결 투표(Majority Voting)**
   * 회귀: **평균값 산출(Averaging)**


### 요약 다이어그램

```
                [ Training Set ]
                    ↓ Bootstrapping
     ┌──────────────┬──────────────┬──────────────┐
     │              │              │              │
[ Tree 1 ]     [ Tree 2 ]     [ Tree 3 ]   ...  [ Tree N ]
     ↓              ↓              ↓              ↓
[ 예측값 1 ]   [ 예측값 2 ]   [ 예측값 3 ]   ...  [ 예측값 N ]
     ↓────────────── Aggregation ──────────────↓
                [ 최종 예측값 ]
```

---

### 🔷 주요 하이퍼파라미터

| 하이퍼파라미터             | 설명                    |
| ------------------- | --------------------- |
| `n_estimators`      | 생성할 트리 개수             |
| `max_features`      | 각 노드 분할 시 고려할 최대 특성 수 |
| `max_depth`         | 각 트리의 최대 깊이           |
| `min_samples_split` | 노드를 분할하기 위한 최소 샘플 수   |
| `bootstrap`         | 부트스트랩 샘플 사용 여부        |

---


## 3. 장점 vs 단점

| 장점                   | 단점                 |
| -------------------- | ------------------ |
| 높은 예측 성능             | 해석 어려움 (블랙박스)      |
| 과적합 방지 효과            | 느린 학습 속도 (트리가 많으면) |
| 변수 중요도 제공 가능         | 큰 메모리 사용량          |
| 범주형, 연속형 변수 모두 처리 가능 |                    |

---

## 4. 활용 예시

* **IoT**: 센서 고장 예측, 장비 이상 탐지
* **헬스케어**: 질병 진단 분류
* **금융**: 부도 위험 평가
* **산업 공정**: 예지 정비(Predictive Maintenance)

---
# 📘 9장. XGBoost (eXtreme Gradient Boosting) 알고리즘

## **1. 역사적 배경**

### (1) 부스팅 알고리즘의 흐름

| 연도     | 알고리즘                         | 설명                                   |
| ------ | ---------------------------- | ------------------------------------ |
| 1990년대 | AdaBoost                     | 가장 처음 널리 사용된 부스팅 모델. 가중치 기반 분류기 조합   |
| 2000년대 | Gradient Boosting (GBM)      | 잔차(residual)에 대한 기울기를 기반으로 약한 학습기 추가 |
| 2014년  | **XGBoost** (by Tianqi Chen) | GBM의 단점 보완 → 속도, 성능, 병렬성에서 혁신적 개선    |

> 📝 참고: XGBoost는 **Kaggle** 등의 머신러닝 경진대회에서 압도적 성능을 보여주며 대중적으로 널리 퍼졌습니다.

---

## **2. 이론적 배경**

### (1) Gradient Boosting의 기본 개념

* 목적: 여러 개의 **약한 학습기(weak learner, 보통은 결정 트리)** 를 순차적으로 학습하여 점점 예측력을 높임.
* 방식: 이전 모델이 만든 **오차(잔차)** 를 예측하도록 다음 모델을 훈련

#### 기본 수식

모델의 예측값을 $F(x)$라 할 때, 이를 반복적으로 갱신:

$$
F_{m}(x) = F_{m-1}(x) + \gamma h_m(x)
$$

* $h_m(x)$: 현재 단계에서 학습하는 새로운 트리
* $\gamma$: 학습률 (learning rate)

---

### (2) XGBoost의 핵심 아이디어

XGBoost는 기존 Gradient Boosting의 약점을 보완하면서 **속도와 정확도를 극대화**하는 데 초점을 맞춘 알고리즘입니다.

#### 🔧 개선 요소

| 요소              | 설명                          |
| --------------- | --------------------------- |
| **정규화 추가**      | 과적합 방지를 위해 **L1/L2** 정규화 포함 |
| **병렬 트리 생성**    | 트리의 노드를 병렬로 분할하여 연산 속도 향상   |
| **히스토그램 기반 학습** | 연속값을 이산화하여 학습 효율 향상         |
| **스파스 데이터 처리**  | 결측값을 자동 감지하여 최적 분기 생성       |
| **Cache 최적화**   | 내부 메모리 접근 최적화로 빠른 연산        |

---

### (3) XGBoost의 목적 함수 (Objective Function)

XGBoost는 목적 함수 $Obj$를 다음과 같이 정의합니다:

$$
Obj = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

* $l$: 손실 함수 (예: MSE, Log loss 등)
* $\Omega(f_k)$: 복잡도 패널티 (트리의 깊이, 노드 수 등)

#### 복잡도 항

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

* $T$: 리프 노드의 수
* $w_j$: 리프 노드의 출력 값
* $\gamma$, $\lambda$: 정규화 계수

---

### (4) XGBoost 주요 하이퍼파라미터

| 파라미터                      | 설명                |
| ------------------------- | ----------------- |
| `n_estimators`            | 생성할 트리 수          |
| `max_depth`               | 트리 최대 깊이          |
| `learning_rate`           | 각 단계의 기여율         |
| `subsample`               | 훈련 샘플 비율 (과적합 방지) |
| `colsample_bytree`        | 트리마다 사용할 특성 비율    |
| `reg_alpha`, `reg_lambda` | L1, L2 정규화 계수     |

---
# 10장. 분류 모델에 대한 실습

## 1. 실습 코드

```python
# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 전처리
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
X.columns = ['Type', 'AirTemp', 'ProcTemp', 'RotSpeed', 'Torque', 'ToolWear']
y = df['Machine failure']

print(X.head())

# 4. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# 5. 정규화 (kNN, SVM만 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 정의
models = {
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
}

# 7. 결과 저장
results = {}

# 8. 학습 및 예측
for name, model in models.items():
    if name in ['kNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "confusion_matrix": cm,
        "classification_report": cr
    }

# 9. 혼동 행렬 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, result) in zip(axes, results.items()):
    sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 10. 성능 출력
for name, result in results.items():
    print(f"\n===== {name} =====")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("\nClassification Report:")
    print(pd.DataFrame(result["classification_report"]).transpose())
```

## 2. 특징 중요도

```python
# 11. 중요도 추출 및 비교
features = X.columns
dt_importance = models['Decision Tree'].feature_importances_
rf_importance = models['XGBoost'].feature_importances_

# 12. 시각화
import numpy as np
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(features))
width = 0.35

bars1 = ax.bar(x - width/2, dt_importance, width, label='Decision Tree')
bars2 = ax.bar(x + width/2, rf_importance, width, label='Random Forest')

# 상단에 값 표시
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# 그래프 설정
ax.set_ylabel("Feature Importance")
ax.set_title("Feature Importance: Decision Tree vs Random Forest")
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

=======
## - https://shorturl.at/SduTO
## 데이터
## - https://shorturl.at/toAqE
## - https://shorturl.at/4LiWF
---
# 📘 **3부. 스마트 헬스 개요 및 기반 기술**

# 📗 **1장. 스마트 헬스란 무엇인가**


## 🔹 1. 스마트 헬스의 정의 및 등장 배경

스마트 헬스(Smart Health)란 정보통신기술(ICT), 인공지능(AI), IoT 센서, 빅데이터 등을 활용하여 개인 맞춤형 건강 관리와 예방 중심의 헬스케어 서비스를 제공하는 패러다임입니다. 기존의 병원 중심 치료 위주에서 벗어나, **언제 어디서나 건강 정보를 수집·분석하고, 실시간으로 피드백을 제공**함으로써 예방과 자기 주도 건강관리를 가능하게 합니다.

### ▪ 주요 배경

* **고령화 사회의 도래**: 만성질환 증가와 의료비 부담 증가
* **ICT 기술 발전**: IoT 센서, 웨어러블 디바이스 보급
* **데이터 기반 의료**: EHR, PHR, 웨어러블 로그 등 다양한 의료 데이터
* **의료 자원의 불균형**: 지역 간 의료 접근성 격차 해소 필요성

---

## 🔹 2. 기존 헬스케어와 스마트 헬스의 비교

| 구분     | 기존 헬스케어      | 스마트 헬스                  |
| ------ | ------------ | ----------------------- |
| 중심 방식  | 병원 방문, 진료 중심 | 비대면, 실시간 데이터 기반         |
| 사용자 역할 | 수동적(의사 주도)   | 능동적(사용자 스스로 건강 관리)      |
| 데이터 수집 | 병원 내 검사      | 센서, 웨어러블, 스마트폰 등 실시간 수집 |
| 기술 활용  | 제한적 (EMR 등)  | IoT, AI, 빅데이터, 클라우드 활용  |
| 목적     | 질병 진단 및 치료   | 예방, 조기 발견, 건강 증진        |

---

## 🔹 3. 스마트 헬스 구성 요소

스마트 헬스는 여러 기술과 시스템이 통합적으로 작동하여 개인의 건강을 관리합니다. 구성 요소는 다음과 같습니다.

| 구성 요소            | 설명                                  |
| ---------------- | ----------------------------------- |
| **IoT 센서/디바이스**  | 심박수, 운동량, 수면 등을 실시간 측정하는 웨어러블 기기    |
| **데이터 수집 및 통신**  | BLE, WiFi, NB-IoT 등 다양한 방식으로 데이터 전송 |
| **헬스 데이터 플랫폼**   | 수집된 데이터를 통합 저장, 분석 가능한 환경           |
| **AI/ML 분석 기술**  | 건강 상태 예측, 이상 징후 감지 등 지능형 분석 수행      |
| **피드백 및 알림 시스템** | 사용자 맞춤형 건강 리포트, 경고 메시지 제공           |

---

## 🔹 4. 스마트 헬스 기술 발전 연표

| 연도        | 주요 발전 내용                               |
| --------- | -------------------------------------- |
| 2010년대 초반 | 웨어러블 기기(예: Fitbit, Jawbone) 상용화        |
| 2014년     | 애플 헬스킷(HealthKit), 구글 핏(Google Fit) 출시 |
| 2016년     | 딥러닝 기반 피부병, 안저 분석 기술 상용화               |
| 2020년 이후  | 코로나19 대응 비대면 진료, 스마트 병원 확산             |
| 2023년 이후  | 생체신호 기반 스트레스, 수면, 심혈관 리스크 예측 솔루션 등장    |

---

# 📗 **2장. 스마트 헬스를 위한 데이터 이해**



## 🔹 1. 헬스케어 데이터의 유형

스마트 헬스에서 활용되는 데이터는 매우 다양하며, 각기 다른 형식과 특성을 가집니다. 대표적인 헬스케어 데이터는 다음과 같습니다.

| 데이터 유형                              | 설명                      | 예시                           |
| ----------------------------------- | ----------------------- | ---------------------------- |
| **EMR (Electronic Medical Record)** | 병원 내에서 수집되는 환자 진료 기록    | 진단코드, 투약 내역, 검사 결과 등         |
| **EHR (Electronic Health Record)**  | 여러 기관 간 공유 가능한 의료기록     | EMR + 생활습관, 백신, 영상 등         |
| **PHR (Personal Health Record)**    | 개인이 수집/관리하는 건강 데이터      | 스마트워치, 앱 기반 자가 기록 등          |
| **생체신호 (Biosignals)**               | 신체 기능에서 측정되는 전기적/물리적 신호 | ECG, PPG, EMG, EEG, 체온, 호흡 등 |
| **행동 및 환경 데이터**                     | 사용자의 운동, 수면, 위치, 날씨 등   | 걸음 수, 수면 시간, GPS, 온습도 등      |

---

## 🔹 2. 대표적인 생체신호와 특징

스마트 헬스에서는 다양한 \*\*생체신호(Biosignal)\*\*를 분석하여 건강 상태를 추정합니다. 대표적인 신호들은 다음과 같습니다.

| 신호              | 측정 대상      | 주요 활용            | 특징                 |
| --------------- | ---------- | ---------------- | ------------------ |
| **ECG (심전도)**   | 심장 전기 신호   | 심박수, HRV, 부정맥 진단 | 고해상도, R-peak 검출    |
| **PPG (광용적맥파)** | 혈류 변화      | 맥박수, 혈중 산소포화도    | 착용 간편, 운동 시 노이즈 민감 |
| **EEG (뇌파)**    | 뇌의 전기활동    | 수면 분석, 발작 감지     | 채널 수 많고 처리 복잡      |
| **EMG (근전도)**   | 근육 수축      | 근피로도 분석, 재활치료    | 짧은 시간 신호, 잡음 영향 큼  |
| **호흡/체온**       | 호흡률, 체온 변화 | 호흡기질환, 발열 감지     | 환경 온도에 영향 받을 수 있음  |

> ECG: [위키백과 심전도](https://ko.wikipedia.org/wiki/%EC%8B%AC%EC%A0%84%EB%8F%84)
> PPG: [LED로 심박수를 측정한다고? '광혈류측정 센서(PPG)'](https://news.samsungdisplay.com/30140)
> EEG: [위키백과 뇌파](https://ko.wikipedia.org/wiki/%EB%87%8C%ED%8C%8C)
> EMG: [위키백과 근전도 검사](https://ko.wikipedia.org/wiki/%EA%B7%BC%EC%A0%84%EB%8F%84_%EA%B2%80%EC%82%AC)
> 호흡 센서: [호흡분석기 'PACER'](https://blog.naver.com/geekstarter/223752501610)
---


## 🔹 3. 웨어러블 헬스 센서의 개요

웨어러블 헬스 센서는 사용자의 생체신호 또는 행동 정보를 실시간으로 측정하고 기록하는 IoT 기반 장치입니다. 손목, 가슴, 귀, 발목, 피부 등에 부착되어 동작하며, 스마트 헬스의 핵심 데이터 수집 도구로 활용됩니다.

| 센서 형태 | 예시 기기                       | 측정 정보            |
| ----- | --------------------------- | ---------------- |
| 손목형   | Apple Watch, Galaxy Watch   | 심박수, 운동량, 수면, 체온 |
| 패치형   | Zephyr BioPatch, VitalPatch | ECG, PPG, 호흡, 체온 |
| 귀걸이형  | Earin, Cosinuss One         | 심박수, 체온          |
| 반지형   | Oura Ring                   | HRV, 수면 단계       |
| 의류형   | Hexoskin, Athos             | 호흡, EMG, 심전도     |

---

## 🔹 4. 센서의 측정 원리

| 센서 종류            | 측정 원리                    | 측정 항목               |
| ---------------- | ------------------------ | ------------------- |
| **ECG 센서**       | 피부 표면 전극을 통해 심장 전기신호 측정  | 심박수, R-R 간격, 부정맥 탐지 |
| **PPG 센서**       | 적외선/녹색광을 혈관에 조사하여 반사광 측정 | 맥박수, 혈중 산소포화도       |
| **IMU (관성측정센서)** | 가속도계와 자이로스코프 기반          | 걸음 수, 자세, 활동 인식     |
| **체온 센서**        | 서미스터, 적외선 측정             | 피부 온도, 중심 체온 추정     |
| **호흡 센서**        | 압력 변화, 스트레인 게이지 활용       | 호흡률, 폐활량 추정         |

---

## 🔹 5. 데이터 전송 및 통신 기술

웨어러블 센서는 수집된 데이터를 스마트폰 또는 클라우드로 전송하기 위해 다양한 통신 기술을 사용합니다.

| 통신 기술                          | 특징               | 적용 사례               |
| ------------------------------ | ---------------- | ------------------- |
| **BLE (Bluetooth Low Energy)** | 짧은 거리, 저전력       | 스마트워치 ↔ 스마트폰        |
| **Wi-Fi**                      | 빠른 속도, 전력 소모 큼   | 스마트 체중계 ↔ 가정용 Wi-Fi |
| **NB-IoT / LTE-M**             | 저전력, 장거리, 셀룰러 기반 | 병원 서버로 데이터 전송       |
| **ZigBee**                     | 저전력, 다수 센서 연결    | 실내용 건강 모니터링 시스템     |
| **UWB (초광대역)**                 | 위치 정확도 높음        | 실내 환자 추적, 낙상 감지     |

---

## 🔹 4. 웨어러블 센서의 데이터 특성

* **연속성**: 실시간 연속 측정으로 시계열 데이터 생성
* **노이즈 포함**: 움직임, 피부 접촉 불량 등으로 인한 잡음 존재
* **사용자 간 다양성**: 생리적 차이로 인해 개인별 기준 상이
* **전력 소모 고려 필요**: 센서 설계 및 수집 주기 최적화 필요

---

## 🔹 6. 웨어러블 센서 선택 시 고려 요소

| 고려 항목       | 설명                         |
| ----------- | -------------------------- |
| **정확도**     | 의료 기준 충족 여부 (예: FDA 인증 여부) |
| **배터리 수명**  | 지속적인 측정 가능 시간              |
| **편의성**     | 사용자의 착용감, 위치 제한            |
| **통신 방식**   | 사용 환경에 맞는 연결성              |
| **데이터 접근성** | API 제공 여부, 데이터 내보내기 가능성    |

---

# 📗 **4장. 스마트 헬스 서비스 사례**



## 🔹 1. 스마트 헬스 서비스의 분류

스마트 헬스 서비스는 제공 주체와 기술 방식에 따라 다음과 같이 구분할 수 있습니다.

| 유형            | 설명                 | 예시                           |
| ------------- | ------------------ | ---------------------------- |
| **개인 건강 관리형** | 웨어러블 기반 실시간 건강 관리  | Apple Health, Samsung Health |
| **질병 예측/진단형** | AI 기반 조기 진단, 위험 예측 | SkinVision, Lunit INSIGHT    |
| **원격 모니터링형**  | 병원과 환자 간 연결, 지속 추적 | Livongo, Dexcom              |
| **스마트 병원형**   | 병원 내 디지털 시스템 통합    | 세브란스 스마트 병원, Mayo Clinic     |

---

## 🔹 2. 스마트 헬스 서비스 사례

### (1) **Apple Health [(애플 헬스)](https://www.apple.com/health/)**

* **기능**: 심박수, 운동량, 수면 기록, 심방세동 감지
* **센서**: Apple Watch (ECG, PPG, IMU 등 내장)
* **분석**: iOS 기반의 건강 앱에서 시각화
* **특징**: EHR 연동, 미국 내 일부 병원과 직접 연결 가능

### (2) **Fitbit [(by Google)](https://store.google.com/gb/category/watches_trackers?hl=en-GB)**

* **기능**: 운동, 수면, 스트레스 추적
* **AI 기술**: 수면 점수 계산, HRV 기반 스트레스 지수
* **데이터 통합**: Fitbit 앱 + Google Health 통합 플랫폼
* **특징**: FDA 승인 ECG 기능 제공

### (3) **SkinVision [(네덜란드)](https://www.skinvision.com/)**

* **기능**: 피부암 위험도 자가 진단
* **기술**: 스마트폰 카메라 기반 CNN 피부 분석
* **성과**: 흑색종 조기 발견 정확도 95% 이상
* **활용**: 사용자가 주기적 사진 촬영 → AI 분석 결과 확인


### (4) **삼성 헬스 [(Samsung Health)](https://www.samsung.com/sec/apps/samsung-health/)**

* **기능**: 걸음 수, 수면, 스트레스, 혈중 산소포화도 측정
* **센서 연동**: Galaxy Watch 시리즈
* **분석**: HRV 기반 스트레스 추정, 수면 단계 자동 분석
* **특징**: 삼성 스마트폰과 자동 연동, 글로벌 1억 이상 사용자

---

## 🔹 3. 스마트 헬스 서비스 설계 시 고려사항

| 항목             | 고려 요소                     |
| -------------- | ------------------------- |
| **데이터 정확성**    | 의료기기 수준 인증 필요 (FDA, CE 등) |
| **개인화 수준**     | 연령, 성별, 상태별 맞춤형 알고리즘      |
| **연동성**        | EHR, 병원 시스템, 모바일 앱 연계 가능성 |
| **설명 가능성**     | AI 결과의 근거 제시 여부           |
| **보안 및 프라이버시** | 생체정보 암호화, GDPR/개인정보보호법 준수 |

---

# 4부: 머신러닝의 이해와 활용

---

# 📖 **1장. 머신러닝 개요**



## ✨ 1. 머신러닝이란 무엇인가

머신러닝(Machine Learning)이란,  
명시적으로 프로그램을 작성하지 않고도 데이터를 이용하여 컴퓨터가 스스로 학습하고 성능을 개선하는 기술입니다.

Arthur Samuel은 머신러닝을 "**명시적으로 프로그래밍하지 않고 컴퓨터가 학습할 수 있게 하는 연구 분야**"라고 정의하였습니다.

머신러닝은 입력 데이터로부터 패턴을 학습하여,  
**새로운 데이터에 대해 예측하거나 분류하는 모델**을 구축하는 것을 목표로 합니다.

### ➡️ 머신러닝 기본 구성 요소
| 구성 요소 | 설명 |
|:--|:--|
| 데이터 | 학습 및 예측을 위한 입력 자료 |
| 모델 | 데이터의 패턴을 학습하는 구조 |
| 학습 알고리즘 | 모델을 최적화하는 방법 |
| 예측 | 학습한 모델로 새로운 데이터에 대한 결과 생성 |

## ✨ 2. 머신러닝의 주요 분류

머신러닝은 학습 방식에 따라 크게 세 가지로 분류됩니다.

| 분류 | 설명 | 예시 |
|:--|:--|:--|
| 지도학습 (Supervised Learning) | 입력과 정답(label)을 이용하여 학습 | 분류(Classification), 회귀(Regression) |
| 비지도학습 (Unsupervised Learning) | 정답 없이 데이터 구조를 학습 | 클러스터링(Clustering), 차원 축소(Dimensionality Reduction) |
| 강화학습 (Reinforcement Learning) | 보상을 통해 최적 행동을 학습 | 게임 플레이, 로봇 제어 |


---

## ✨ 3. 머신러닝 워크플로우

머신러닝 프로젝트는 다음과 같은 단계를 거칩니다.

### ➡️ 전형적인 머신러닝 워크플로우

1. **문제 정의**
2. **데이터 수집**
3. **데이터 전처리 및 탐색**
4. **특성 선택 및 생성**
5. **모델 선택 및 학습**
6. **모델 평가**
7. **모델 개선 및 최적화**
8. **최종 모델 배포**

```
데이터 수집 → 데이터 전처리 → 모델 학습 → 모델 평가 → 하이퍼파라미터 튜닝 → 최종 예측
```

- 이 과정에서 **데이터 전처리와 모델 선택**이 성능에 가장 큰 영향을 미칩니다.
---

# 📖 **2장. 데이터 전처리와 특성 공학**



## ✨ 1. 결측치 처리와 이상값 탐지

### ➡️ 결측치(Missing Value) 처리

머신러닝 모델은 결측값을 포함하는 데이터를 직접 다루지 못하는 경우가 많습니다.  
따라서 적절한 방법으로 결측값을 처리해야 합니다.

| 방법 | 설명 | 예시 |
|:--|:--|:--|
| 삭제 (Drop) | 결측치가 있는 행 또는 열을 제거 | `dropna()` |
| 대체 (Imputation) | 평균, 중앙값, 최빈값 등으로 대체 | `fillna(value)` |

> ✅ 데이터의 분포를 고려하여 대체하는 것이 중요합니다.

### ➡️ 이상값(Outlier) 탐지

이상값은 데이터 분포에서 벗어난 값입니다.  
주로 IQR, Z-Score 등을 활용하여 탐지합니다.

| 방법 | 설명 |
|:--|:--|
| IQR 방법 | Q1, Q3 기준으로 이상 범위 외 값 탐지 |
| Z-Score 방법 | 평균 대비 표준편차 범위를 벗어난 값 탐지 |

---

## ✨ 2. 범주형 변수 인코딩

머신러닝 모델은 수치형 데이터만 입력으로 받기 때문에,  
범주형(categorical) 데이터는 수치로 변환해야 합니다.

| 인코딩 방법 | 설명 | 예시 |
|:--|:--|:--|
| 레이블 인코딩 (Label Encoding) | 각 카테고리를 정수로 매핑 | Male → 0, Female → 1 |
| 원-핫 인코딩 (One-Hot Encoding) | 각 카테고리를 0/1로 변환하는 벡터 생성 | `get_dummies(), to_categorical()` 사용 |

> ✅ 베이지안 룰, 트리 기반 모델은 레이블 인코딩을 사용해도 괜찮지만,  
> ✅ 선형 모델(SVM, 로지스틱 회귀 등)에서는 원-핫 인코딩이 더 적합합니다.

---

## ✨ 3. 정규화와 표준화

특성(feature)들의 스케일이 다를 경우,  
머신러닝 모델은 특정 특성에 지나치게 민감하게 반응할 수 있습니다.  
이를 방지하기 위해 스케일 조정이 필요합니다.

| 방법 | 설명 | 수식 |
|:--|:--|:--|
| 정규화 (Normalization) | 모든 값을 0~1 범위로 변환 | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ |
| 표준화 (Standardization) | 평균 0, 표준편차 1로 변환 | $x' = \frac{x - \mu}{\sigma}$ |

> ✅ KNN, SVM 같은 거리 기반 알고리즘은 정규화/표준화가 매우 중요합니다.

---

## ✨ 4. 특성 선택과 차원 축소

모든 특성이 예측에 유용한 것은 아닙니다.  
**특성 선택(Feature Selection)** 은 중요한 특성만 골라 모델 성능을 향상시키는 과정입니다.

| 방법 | 설명 |
|:--|:--|
| 필터 방식 (Filter) | 통계적 기준(상관계수 등)으로 선택 |
| 래퍼 방식 (Wrapper) | 모델을 통해 최적 특성 조합 탐색 |
| 임베디드 방식 (Embedded) | 모델 학습 중 특성 선택 (ex. Lasso) |

또한 고차원 문제를 해결하기 위해  
**차원 축소(Dimensionality Reduction)** 기법(PCA 등)을 사용할 수 있습니다.

---

## ✨ 5. 데이터 분할: 학습/검증/테스트셋

모델의 성능을 정확히 평가하기 위해, 데이터를 다음과 같이 분할합니다.

| 구분 | 설명 | 일반적인 비율 |
|:--|:--|:--|
| 학습셋 (Training Set) | 모델을 학습시키는 데이터 | 60~80% |
| 검증셋 (Validation Set) | 하이퍼파라미터 튜닝용 데이터 | 10~20% |
| 테스트셋 (Test Set) | 최종 성능 평가용 데이터 | 10~20% |

> ✅ 교차 검증(Cross Validation) 기법을 사용하면 보다 안정적인 평가가 가능합니다.

---

## 🛠️ 실습: 타이타닉 데이터 전처리 

```python
# 타이타닉 데이터 전처리

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print(titanic.info())
print(titanic.head(10))

# 1. 결측치 처리
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
print(titanic.info())
print(titanic.head(10))

# 2. 범주형 변수 인코딩
print(titanic['Sex'].value_counts())
print(titanic['Embarked'].value_counts())
titanic['Sex'] = titanic['Sex'].map({'female': 0, 'male': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C':0, 'S':1, 'Q':2})
print(titanic.head(10))

# 3. 불필요한 열 제거
titanic.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
print(titanic.head(10))

# 4. 특성과 타깃 분리
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 정규화 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("타이타닉 데이터 전처리 완료!")
print("훈련 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
```
---


# 📘 3장. 회귀 분석(Regression)

회귀 분석은 **연속적인 값을 예측**할 때 사용하는 대표적인 기계학습 알고리즘입니다.

| 항목      | 설명                                                |
| ------- | ------------------------------------------------- |
| 목적      | 입력 변수로부터 수치형 출력값 예측                               |
| 예시      | 기온, 습도 → 작물 생장량, 수확량 예측                           |
| 대표 알고리즘 | 선형 회귀(Linear Regression), 릿지 회귀(Ridge), 라쏘(Lasso) |

### 적용 예

* 기상 데이터 기반 **수확량 예측**
* 이산화탄소 농도에 따른 **생육 속도 예측**
---

# 🔹 선형 회귀 (Linear Regression)



## 1. 역사적 배경


#### 1.. 기원

* **1805년**: 프랑스 수학자 **Adrien-Marie Legendre**가 ‘최소제곱법(Least Squares Method)’을 도입하여 천문학적 관측 데이터의 오차를 최소화하려 했습니다.
* **1809년**: 독일 수학자 **Carl Friedrich Gauss**도 동일한 방법을 독자적으로 개발하여 **정규분포와의 연결**을 이론화하였습니다.

#### 2.. 통계학으로의 확장

* **19세기 후반**: 통계학자 **Francis Galton**이 인간 키의 상관관계 분석을 통해 ‘회귀(regression)’라는 용어를 도입하였습니다.

  * 아버지 키와 아들 키 간의 관계에서 평균으로 되돌아가는 성질 → “회귀(regression) toward the mean”

#### 3.. 컴퓨터 시대 이후

* 20세기 중반 이후 회귀 분석은 컴퓨터를 통해 자동화되며 통계학, 경제학, 공학, 생물학 등 다양한 분야에서 **기초 예측 도구**로 자리잡게 됩니다.

---

## 2. 이론적 알고리즘 

#### 1.. 목표
입력 변수 $X$와 출력 변수 $y$ 사이의 **선형 관계**를 모델링하여, 새로운 입력 값에 대한 **연속적 출력 값**을 예측하는 것.

#### 2.. 수식 표현

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{Xw} + b
$$

* $\hat{y}$: 예측값
* $\mathbf{X}$: 입력 벡터
* $\mathbf{w}$: 가중치 벡터
* $b$: 절편(intercept)

#### 3.. 목적 함수 (비용 함수, Loss Function)

\*\*MSE (Mean Squared Error)\*\*를 사용하여 오차를 최소화합니다:

$$
J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 3. 가정 조건

선형 회귀는 다음의 통계적 가정을 전제로 합니다:

| 가정                              | 설명                    |
| ------------------------------- | --------------------- |
| 선형성 (Linearity)                 | 입력 변수와 출력 변수 간의 선형 관계 |
| 독립성 (Independence)              | 관측값 간의 독립성            |
| 등분산성 (Homoscedasticity)         | 오차의 분산이 일정함           |
| 정규성 (Normality)                 | 잔차(오차)가 정규 분포를 따름     |
| 다중공선성 없음 (No multicollinearity) | 입력 변수 간 과도한 상관관계 없음   |

---

## 4. 회귀분석 프로젝트 사례

### 🔹 1. 작물 생장 예측 

회귀 분석은 작물의 생장량(예: 잎 면적, 키, 생중량 등)을 예측하기 위해 주로 사용됩니다. 입력 변수로는 온도, 습도, CO₂, 일조량 등의 **환경 정보**가 사용되며, 출력은 **생육 지표의 연속적 수치**입니다.

#### 회귀 분석 흐름

1. 데이터 수집: 시계열 센서 + 생육 측정 데이터
2. 특성 선택: 평균 온도, 누적 일조량 등
3. 모델 학습: 선형 회귀 또는 다항 회귀
4. 성능 평가: MAE, RMSE, R²

---

### 🔹 2. 날씨 데이터 기반 수확 시기 예측

수확 시기 예측은 생육률과 외부 환경 요인을 기반으로 작물의 **최적 수확 시점**을 분류하거나 회귀 문제로 모델링합니다.

| 입력 변수 | 설명            |
| ----- | ------------- |
| 생육일수  | 파종 후 경과 일수    |
| 평균 기온 | 생육기 평균 기온     |
| 총 일조량 | 누적 일조 시간      |
| 강수량   | 수분 스트레스 영향 고려 |

####  예측 접근 방식

* **회귀 모델**: 수확까지 남은 일수 예측
* **이진 분류 모델**: ‘수확 적기 여부’ (예/아니오)


---

## 실습 1: 다중 선형 회귀를 활용한 보스턴 주택 가격 예측



**Boston Housing 데이터셋**은 미국 보스턴 지역의 주택 가격에 영향을 주는 다양한 변수들(방 수, 지역 범죄율, 교육 수준 등)을 포함한 데이터입니다.
다중 선형 회귀 모델을 학습시켜, **주어진 특성들로부터 집값을 예측**하는 모델을 구현합니다.
> OpenML에서 보스턴 주택 데이터셋을 가져옴.


### 데이터 정보

Boston Housing 데이터셋에는 다음과 같은 특성(Feature)이 포함되어 있습니다:

| 변수명      | 설명                        |
| -------- | ------------------------- |
| CRIM     | 지역 범죄율                    |
| ZN       | 25,000 평방피트 이상 거주지역 비율    |
| INDUS    | 비소매상업지역 면적 비율             |
| CHAS     | 찰스강 인접 여부 (1: 인접, 0: 그 외) |
| NOX      | 일산화질소 농도                  |
| RM       | 주택 1가구당 평균 방 개수           |
| AGE      | 1940년 이전에 지어진 주택 비율       |
| DIS      | 5개 보스턴 고용센터까지 거리          |
| RAD      | 방사형 고속도로 접근성 지수           |
| TAX      | 재산세율                      |
| PTRATIO  | 학생-교사 비율                  |
| B        | 흑인 인구 비율 지표               |
| LSTAT    | 저소득층 비율 (%)               |
| **MEDV** | 주택 가격 (목표값, 단위: \$1000s)  |



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml

# 1. 데이터 로딩
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data
y = boston.target

print(X.info())

# categorical 변수 --> 수치형으로 변경
X['CHAS'] = X['CHAS'].astype(int)
X['RAD'] = X['RAD'].astype(int)

# 2. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ 모델 평가")
print(f"▶ MSE: {mse:.2f}")
print(f"▶ RMSE: {rmse:.2f}")
print(f"▶ R² Score: {r2:.2f}")

# 4. 회귀 계수 출력 및 해석
coefficients = pd.Series(model.coef_, index=X.columns)

print("\n✅ 회귀 계수:")
print(coefficients.sort_values(ascending=False))

print("\n✅ 주요 변수 해석:")
print(f"NOX (일산화질소 농도): {coefficients['NOX']:.3f} → 일산화질소 농도 높을수록 집값 하락")
print(f"RM (방 개수): {coefficients['RM']:.3f} → 방이 많을수록 집값 상승")
print(f"CHAS (찰스강 인접 여부): {coefficients['CHAS']:.3f} → 찰스강 인접할수록 집값 상승")
```
---
## 실습 2: GreenHouse 수확량 예측

### 데이터 셋
- Autonomous Greenhouse Challenge (AGC) 데이터셋
- 링크: https://www.kaggle.com/datasets/piantic/autonomous-greenhouse-challengeagc-2nd-2019/data
- **Autonomous Greenhouse Challenge (AGC)** 데이터셋에는 여러 폴더가 존재하는데, 각각이 의미하는 바는 다음과 같습니다:


### 폴더 구조 및 설명

| 폴더명            | 설명                                                                                                             |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| **AICU**       | 2019–2020년 챌린지에 참가한 **AiCU 팀**이 사용한 온실 데이터를 포함합니다. 센서 입력 값(온도, 습도, CO₂, PAR, 토양 수분 등)과 AI 기반 제어 전략이 기록되어 있습니다. |
| **Automatoes** | **Automatoes 팀** (2등 또는 우승 팀)의 데이터를 나타냅니다. 이번 대회에서 토마토 재배에 사용된 환경 조작 전략과 해당 센서 데이터를 포함합니다.                     |
| **Dialog**     | 데이터셋 제공 구조상 **Dialog**는 챌린지의 “참가자–주최 측 간 대화 데이터”일 가능성이 높습니다. 실제 환경 센서 값이나 실험 데이터를 담고 있지는 않을 확률이 큽니다.           |

### 사용 폴더:  **Automatoes** 

### 사용 파일 목록

| 파일명                     | 설명                          | 사용 목적         |
| ----------------------- | --------------------------- | ------------- |
| `GreenhouseClimate.csv` | 온실 내부 온도, 습도, CO₂, 광량, 급수 등 | **입력 변수 (X)** |
| `Production.csv`        | 품질 등급별 수확량 (A, B), 수확 일자 포함 | **목표 변수 (y)** |


### 선택된 주요 변수

| 변수명       | 설명                          | 단위        | 파일                |
| --------- | --------------------------- | --------- | ----------------- |
| `Tair`    | 온실 내 공기 온도                  | °C        | GreenhouseClimate |
| `Rhair`   | 온실 내 상대 습도                  | %         | GreenhouseClimate |
| `CO2air`  | 온실 내 CO₂ 농도                 | ppm       | GreenhouseClimate |
| `Tot_PAR` | 총 광합성 유효 복사량 (태양 + LED/HPS) | μmol/m²/s | GreenhouseClimate |
| `Cum_irr` | 하루 누적 관수량                   | L/m²      | GreenhouseClimate |
| `ProdA`   | 상급 품질 토마토 수확량               | kg/m²     | Production        |

※ 시간 단위는 5분 간격이며, 일 단위로 리샘플링 후 사용합니다.


* 먼저, AICU 폴더의 GreenhouseClimate.csv와 Production.csv 를 업로드합니다. 

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 파일 경로 설정
climate_path = "/content/GreenhouseClimate.csv"
prod_path = "/content/Production.csv"

# 2. 데이터 불러오기
climate = pd.read_csv(climate_path)
climate['Time'] = pd.to_datetime(climate['Time'], unit='D', origin='1900-01-01')

production = pd.read_csv(prod_path)
production['Time'] = pd.to_datetime(production['Time'], unit='D', origin='1900-01-01')

# 3. 필요한 변수만 추출
climate = climate[['Time', 'Tair', 'Rhair', 'CO2air', 'Tot_PAR', 'Cum_irr']]
production = production[['Time', 'ProdA']]  # 목표: Class A 수확량

# 4. 시간 단위 평균 (하루 단위로)
climate_indexed = climate.set_index('Time') 
production_indexed = production.set_index('Time')

numerical_cols = ['Tair', 'Rhair', 'CO2air', 'Tot_PAR', 'Cum_irr']
for col in numerical_cols:
    climate_indexed[col] = pd.to_numeric(climate_indexed[col], errors='coerce')

climate_daily = climate_indexed[numerical_cols].resample('D').mean().reset_index()
production_daily = production_indexed.resample('D').sum().reset_index()

# 5. 병합
df = pd.merge(climate_daily, production_daily, on='Time')

# 6. 결측치 제거
df.dropna(inplace=True)

# 7. X, y 분리
X = df[numerical_cols]
y = df['ProdA']

# 8. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. 모델 훈련 (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 11. 예측 및 평가
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 12. 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Ground Truth')
plt.plot(y_pred, label='Predicted (LinearReg)', linestyle='--')
plt.title("Tomato Production Prediction (Linear Regression)")
plt.xlabel("Sample Index")
plt.ylabel("Production (kg/m²)")
plt.legend()
plt.grid()
plt.show()
```


---
# 🔹 로지스틱 회귀(Logistic Regression)


## 1. 로지스틱 회귀란?


로지스틱 회귀는 **분류(classification)** 문제를 해결하는 지도 학습 알고리즘입니다. 이름은 "회귀"이지만, 실제로는 **출력값을 확률로 예측하고, 이 확률을 바탕으로 클래스(0 또는 1)를 분류**하는 데 사용됩니다.


* 출력값은 **0과 1 사이의 확률**로 표현됩니다.
* **시그모이드 함수**를 사용해 선형 조합을 확률로 변환합니다.
* **이진 분류**에 주로 사용되며, **다중 클래스 확장**도 가능합니다.



| 항목    | 설명                                      |
| ----- | --------------------------------------- |
| 목적    | 확률 기반 이진 분류                             |
| 예측값   | 시그모이드 함수 출력 $\hat{y} \in (0,1)$         |
| 결정 기준 | $\hat{y} \ge 0.5 \rightarrow 1$, 그 외는 0 |
| 손실 함수 | 로지스틱 손실 (Binary Cross-Entropy)          |
| 최적화   | 경사하강법 등 사용                              |
| 확장    | Scikit-learn은 다중 클래스 분류 가능                |

---

### 문제 1. 로지스틱 회귀에서 예측값 $\hat{y}$는 어떤 범위를 가지는가?

① 0 또는 1
② 음수에서 양수
③ 0 이상 1 이하의 실수
④ 정수

**정답**: ③ 0 이상 1 이하의 실수
**해설**: 시그모이드 함수의 출력값은 항상 **(0, 1)** 사이의 **확률값**입니다.


### 문제 2. 로지스틱 회귀에서 시그모이드 함수의 역할은?

① 특성 선택
② 정규화
③ 선형 조합을 확률값으로 변환
④ 가중치 감소

**정답**: ③ 선형 조합을 확률값으로 변환
**해설**: $\sigma(z) = \frac{1}{1 + e^{-z}}$은 선형 조합 $z$를 0\~1 사이의 확률로 변환합니다.

---
## 2. 로지스틱 회귀 실습


### 📦 데이터셋: Ai4I 2020 Predictive Maintenance Dataset

이 데이터셋은 **스마트 제조 환경에서의 예측 유지보수**를 위한 시뮬레이션 데이터로, 센서 측정값과 기계 고장 여부를 포함합니다. 공개적으로 접근 가능한 이 데이터셋은 다양한 연구에 활용되고 있습니다.


* **출처**: UCI Machine Learning Repository
* **데이터 형태**: CSV (약 10,000개 샘플, 14개 열)
* **목적**: 공장 설비에서 발생할 수 있는 \*\*기계 고장(Machine Failure)\*\*을 예측하기 위한 **이진 분류(binary classification)** 문제입니다.



#### 전체 컬럼 설명표

| 변수명 (컬럼명)                 | 데이터 유형     | 설명                                    |
| ------------------------- | ---------- | ------------------------------------- |
| `UDI`                     | 정수 (int)   | 고유 식별자 (Unique ID)                    |
| `Product ID`              | 문자열 (str)  | 제품 고유 식별자                             |
| `Type`                    | 문자열 (str)  | 제품 유형 (L, M, H 세 가지 타입)               |
| `Air temperature [K]`     | 실수 (float) | 공기 온도 (켈빈 단위)                         |
| `Process temperature [K]` | 실수 (float) | 공정 온도 (켈빈 단위)                         |
| `Rotational speed [rpm]`  | 실수 (float) | 회전 속도 (분당 회전수, rpm)                   |
| `Torque [Nm]`             | 실수 (float) | 토크 (Nm)                               |
| `Tool wear [min]`         | 실수 (float) | 공구 마모 시간 (분)                          |
| `TWF`                     | 0/1 (int)  | Tool Wear Failure (공구 마모 고장 여부)       |
| `HDF`                     | 0/1 (int)  | Heat Dissipation Failure (열 방출 고장 여부) |
| `PWF`                     | 0/1 (int)  | Power Failure (전력 고장 여부)              |
| `OSF`                     | 0/1 (int)  | Overstrain Failure (과부하 고장 여부)        |
| `RNF`                     | 0/1 (int)  | Random Failures (임의 고장 여부)            |
| `Machine failure`         | 0/1 (int)  | **최종 고장 발생 여부 (Target 변수)**           |

---

#### 타겟 변수 (Machine failure)

* `Machine failure`는 위의 TWF, HDF, PWF, OSF, RNF 다섯 개의 고장 중 **하나라도 발생하면 1**, 그렇지 않으면 0입니다.

$$
\text{Machine failure} = \begin{cases}
1 & \text{(TWF or HDF or PWF or OSF or RNF = 1)} \\
0 & \text{(모두 0)}
\end{cases}
$$


#### 예시 행 데이터

| UDI | Product ID | Type | Air Temp \[K] | Proc Temp \[K] | Rot Speed | Torque | Tool Wear | TWF | HDF | PWF | OSF | RNF | Machine failure |
| --- | ---------- | ---- | ------------- | -------------- | --------- | ------ | --------- | --- | --- | --- | --- | --- | --------------- |
| 1   | M14860     | M    | 298.1         | 308.6          | 1551      | 42.8   | 0         | 0   | 0   | 0   | 0   | 0   | 0               |


#### 주요 분석 포인트

* **제품 유형(Type)**: L, M, H 간 성능 차이 존재 가능
* **온도/회전수/토크 등 물리 센서 변수**와 **고장 발생 여부** 간 관계 분석 가능
* **고장 비율이 낮아 클래스 불균형 문제가 존재할 수 있음**


#### 활용 가능 문제 유형

| 문제 유형 | 가능 여부 | 설명                            |
| ----- | ----- | ----------------------------- |
| 이진 분류 | ✅     | 머신 고장 여부 예측                   |
| 다중 분류 | ✅     | TWF, HDF, PWF 등 개별 고장 유형 분류   |
| 이상 탐지 | ✅     | 고장 발생이 매우 드물다면 이상탐지 문제로 접근 가능 |
| 회귀 문제 | ❌     | 현재는 회귀형 목표변수 없음               |

---

### 로지스틱 회귀 실습 코드

```python
# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 데이터 전처리
# 불필요한 열 제거
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)


## 범주형 변수 처리
print(df.info())
print(df['Type'].value_counts())

df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

# 타겟 변수와 특성 분리
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]' ] ]
y = df['Machine failure']

print(X.info())

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. 예측 및 평가
y_pred = model.predict(X_test_scaled)

# 분류 성능 평가
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

---
# 📘 4장. 분류 모델

---
# 📘 5장. k-최근접 이웃 알고리즘 (k-Nearest Neighbors, kNN)


## 1. 개념

\*\*k-최근접 이웃 알고리즘(kNN)\*\*은 가장 단순하면서도 강력한 **비모수(non-parametric)** 지도 학습 알고리즘 중 하나입니다.

* 주어진 입력 데이터에 대해 **가장 가까운 k개의 이웃**을 찾고,
* 그 이웃들의 레이블을 바탕으로 예측을 수행합니다.


## 2. 원리

1. **훈련 단계 (Training)**

   * 특별한 학습 과정이 없습니다. 모든 학습 데이터를 **기억**해둡니다.
   * 즉, **기억 기반 모델(Memory-based model)** 또는 **Lazy Learning**이라고 합니다.

2. **예측 단계 (Prediction)**
   새로운 데이터 포인트 $x$가 들어오면:

   * 학습 데이터 전체와의 거리를 계산합니다.
   * 거리 기준으로 가장 가까운 **k개의 데이터**를 선택합니다.
   * **분류 문제**:

     * 다수결 투표(Majority Voting)를 통해 가장 많이 등장한 클래스 선택
   * **회귀 문제**:

     * k개의 평균을 사용해 예측


### 거리 계산

가장 널리 쓰이는 거리 함수는 \*\*유클리드 거리(Euclidean distance)\*\*입니다:

$$
d(x, x_i) = \sqrt{ \sum_{j=1}^{n} (x_j - x_{ij})^2 }
$$

그 외에도:

* 맨해튼 거리: $\sum |x_j - x_{ij}|$
* 코사인 거리: $1 - \cos(\theta)$

---

## 3. 하이퍼파라미터

| 하이퍼파라미터      | 설명                                             |
| ------------ | ---------------------------------------------- |
| **k** (이웃 수) | 너무 작으면 과적합, 너무 크면 과소적합                         |
| **거리 척도**    | 유클리드, 맨해튼, 코사인 등                               |
| **가중치 방식**   | 이웃 거리에 따라 가중치를 둘 수도 있음 (`uniform`, `distance`) |

---

## 4. 장점과 단점

| 장점                    | 단점                            |
| --------------------- | ----------------------------- |
| 구현이 간단함               | 계산 비용이 큼 (예측 시 전체 데이터와 거리 계산) |
| 비모수 모델 (데이터 분포 가정 없음) | 고차원에서는 성능 저하 (차원의 저주)         |
| 다중 클래스/회귀 모두 가능       | 훈련 데이터가 많을수록 속도 느림            |


---

# 📘 6장. 서포트 벡터 머신(Support Vector Machine, SVM)

## 1. 역사적 배경

* **1992년**: 러시아 출신의 **Vladimir Vapnik**과 **Alexey Chervonenkis**가 이론적 기초인 **VC 차원**(Vapnik–Chervonenkis dimension)과 **구간 리스크 최소화** 개념을 발표했습니다.
* **1995년**: Vapnik과 Cortes가 현대적 의미의 SVM 알고리즘을 공식화함. 이 논문은 "Support-Vector Networks"라는 제목으로 발표되며, **마진 최대화**와 \*\*커널 트릭(Kernel Trick)\*\*을 통해 비선형 분류까지 확장됨.
* **2000년대**: Bioinformatics, 문서 분류, 이미지 인식 등 다양한 분야에서 **고차원 소규모 데이터**에 강한 분류기로 널리 활용됨.

## 2. SVM의 핵심 아이디어

* **마진 최대화**: 클래스를 구분하는 **결정 경계**(Decision Boundary) 중, 가장 \*\*넓은 여백(Margin)\*\*을 가지는 선형 결정 경계를 찾음.
* **서포트 벡터**: 결정 경계에 가장 가까운 데이터 포인트.
* **커널 기법**: 비선형 데이터를 고차원으로 매핑하여 선형 분리 가능하게 만드는 방법.


## 3. 수학적 수식

### 목적: 최대 마진 분리 초평면 찾기

선형 SVM의 목적은 다음과 같은 초평면을 찾는 것입니다.

$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

클래스는 다음 조건을 만족해야 합니다:

$$
y_i (\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

### 최적화 문제

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1
$$

이 문제는 **Convex Quadratic Programming** 형태이며, Lagrange multiplier 기법으로 최적화됩니다.

### 🔄 비선형 SVM (Kernel Trick)

$$
\phi(\mathbf{x})^\top \phi(\mathbf{x'}) = K(\mathbf{x}, \mathbf{x'})
$$

* 대표 커널: 선형(linear), 다항식(polynomial), RBF(Gaussian), 시그모이드(sigmoid)

---

# 📘 7장. 의사결정나무 (Decision Tree) 알고리즘

## 1. 역사적 배경

* **1960년대**: 통계학 분야에서 분류 및 회귀를 위한 트리 기반 모델이 연구되기 시작함.
* **1986년**: **J. Ross Quinlan**이 발표한 **ID3(Iterative Dichotomiser 3)** 알고리즘이 널리 주목받으며 의사결정나무 알고리즘의 표준이 됨.
* 이후 **C4.5**(1993), **CART(Classification and Regression Tree)**(Breiman et al., 1984) 등 다양한 버전이 개발됨.
* 현재는 **Random Forest**, **XGBoost** 등의 앙상블 기반 트리 모델의 기본 단위로도 사용됨.


## 2. 이론적 개념

### 기본 아이디어

의사결정나무는 데이터를 \*\*특성(feature)\*\*에 따라 분할(split)하여 **트리 형태로 분류** 또는 **예측**을 수행합니다.

* 루트 노드: 전체 데이터
* 내부 노드: 특성 기준으로 데이터를 분할
* 리프 노드: 최종 예측 결과(클래스 or 수치 값)


### 분할 기준 (지니, 엔트로피, 분산)

#### (1) **분류(Classification)** 문제:

* **지니 불순도(Gini Impurity)**

  $$
  Gini = 1 - \sum_{i=1}^n p_i^2
  $$
* **엔트로피(Entropy)**

  $$
  Entropy = - \sum_{i=1}^n p_i \log_2 p_i
  $$

→ 두 기준 모두 불순도(혼합도)를 최소화하도록 분할합니다.

#### (2) **회귀(Regression)** 문제:

* **MSE (Mean Squared Error)** 또는 **MAE**를 사용하여 분할

---

###  가지치기 (Pruning)

* **사전 가지치기 (Pre-pruning)**: 깊이 제한, 최소 노드 수 설정 등
* **사후 가지치기 (Post-pruning)**: 트리를 완성한 후 불필요한 노드를 제거하여 과적합 방지

---

## 3. 알고리즘 요약

1. 루트 노드에서 시작
2. 각 특성에 대해 최적 분할 기준을 평가 (지니, 엔트로피 등)
3. 가장 불순도를 많이 줄이는 특성으로 분할
4. 리프 노드가 될 조건(순도가 높거나 최대 깊이 도달 등)까지 반복
5. 예측은 리프 노드의 클래스(또는 평균값)로 결정

---

# 📘 8장. Random Forest 알고리즘


## 1. 역사적 배경

| 항목        | 내용                                                        |
| --------- | --------------------------------------------------------- |
| **고안자**   | 레오 브레이먼 (Leo Breiman), 2001년 발표                           |
| **논문**    | “Random Forests” (2001)                                   |
| **기반 개념** | 배깅(Bagging: Bootstrap Aggregating) + 결정 트리(Decision Tree) |
| **개발 목적** | 개별 결정트리의 과적합(overfitting)을 줄이고 예측 성능을 향상시키기 위해            |

**이전 배경**

* 1996년: Breiman이 Bagging(배깅) 기법 제안
* 이후, 여러 개의 배깅된 결정 트리에 무작위성을 더해 **Random Forest**로 확장

---

## 2. 이론적 배경

### 핵심 개념

**Random Forest는 여러 개의 Decision Tree를 훈련시킨 뒤, 각 트리의 결과를 종합하여 최종 예측을 수행하는 앙상블 학습 기법입니다.**

---

### 🔷 Bagging: Bootstrap Aggregating

* **Bootstrap**: 데이터셋에서 **중복 허용 무작위 샘플링**으로 N개의 학습용 샘플 생성
* **Aggregating**: 각각의 모델 결과를 평균(회귀) 또는 투표(분류)를 통해 종합

---

### 🔷 Randomness (무작위성) 도입

1. **데이터 샘플 무작위 선택** (Bagging)
2. **각 노드에서 특성(feature) 무작위 선택** (예: √p 개 중 최적 분할 특성 선택)

이 두 가지 무작위성으로 인해 **트리 간 상관성 감소**, 결과적으로 **앙상블 효과 향상** 및 **과적합 감소** 효과를 얻음.

---

### 🔷 알고리즘 동작 과정

1. **T개의 결정 트리 생성**

   * 각 트리는 Bootstrapping된 데이터로 학습
2. **각 트리에서 무작위 피처 서브셋 선택**
3. **각 트리의 결과 예측**

   * 분류: **다수결 투표(Majority Voting)**
   * 회귀: **평균값 산출(Averaging)**


### 요약 다이어그램

```
                [ Training Set ]
                    ↓ Bootstrapping
     ┌──────────────┬──────────────┬──────────────┐
     │              │              │              │
[ Tree 1 ]     [ Tree 2 ]     [ Tree 3 ]   ...  [ Tree N ]
     ↓              ↓              ↓              ↓
[ 예측값 1 ]   [ 예측값 2 ]   [ 예측값 3 ]   ...  [ 예측값 N ]
     ↓────────────── Aggregation ──────────────↓
                [ 최종 예측값 ]
```

---

### 🔷 주요 하이퍼파라미터

| 하이퍼파라미터             | 설명                    |
| ------------------- | --------------------- |
| `n_estimators`      | 생성할 트리 개수             |
| `max_features`      | 각 노드 분할 시 고려할 최대 특성 수 |
| `max_depth`         | 각 트리의 최대 깊이           |
| `min_samples_split` | 노드를 분할하기 위한 최소 샘플 수   |
| `bootstrap`         | 부트스트랩 샘플 사용 여부        |

---


## 3. 장점 vs 단점

| 장점                   | 단점                 |
| -------------------- | ------------------ |
| 높은 예측 성능             | 해석 어려움 (블랙박스)      |
| 과적합 방지 효과            | 느린 학습 속도 (트리가 많으면) |
| 변수 중요도 제공 가능         | 큰 메모리 사용량          |
| 범주형, 연속형 변수 모두 처리 가능 |                    |

---

## 4. 활용 예시

* **IoT**: 센서 고장 예측, 장비 이상 탐지
* **헬스케어**: 질병 진단 분류
* **금융**: 부도 위험 평가
* **산업 공정**: 예지 정비(Predictive Maintenance)

---
# 📘 9장. XGBoost (eXtreme Gradient Boosting) 알고리즘

## **1. 역사적 배경**

### (1) 부스팅 알고리즘의 흐름

| 연도     | 알고리즘                         | 설명                                   |
| ------ | ---------------------------- | ------------------------------------ |
| 1990년대 | AdaBoost                     | 가장 처음 널리 사용된 부스팅 모델. 가중치 기반 분류기 조합   |
| 2000년대 | Gradient Boosting (GBM)      | 잔차(residual)에 대한 기울기를 기반으로 약한 학습기 추가 |
| 2014년  | **XGBoost** (by Tianqi Chen) | GBM의 단점 보완 → 속도, 성능, 병렬성에서 혁신적 개선    |

> 📝 참고: XGBoost는 **Kaggle** 등의 머신러닝 경진대회에서 압도적 성능을 보여주며 대중적으로 널리 퍼졌습니다.

---

## **2. 이론적 배경**

### (1) Gradient Boosting의 기본 개념

* 목적: 여러 개의 **약한 학습기(weak learner, 보통은 결정 트리)** 를 순차적으로 학습하여 점점 예측력을 높임.
* 방식: 이전 모델이 만든 **오차(잔차)** 를 예측하도록 다음 모델을 훈련

#### 기본 수식

모델의 예측값을 $F(x)$라 할 때, 이를 반복적으로 갱신:

$$
F_{m}(x) = F_{m-1}(x) + \gamma h_m(x)
$$

* $h_m(x)$: 현재 단계에서 학습하는 새로운 트리
* $\gamma$: 학습률 (learning rate)

---

### (2) XGBoost의 핵심 아이디어

XGBoost는 기존 Gradient Boosting의 약점을 보완하면서 **속도와 정확도를 극대화**하는 데 초점을 맞춘 알고리즘입니다.

#### 🔧 개선 요소

| 요소              | 설명                          |
| --------------- | --------------------------- |
| **정규화 추가**      | 과적합 방지를 위해 **L1/L2** 정규화 포함 |
| **병렬 트리 생성**    | 트리의 노드를 병렬로 분할하여 연산 속도 향상   |
| **히스토그램 기반 학습** | 연속값을 이산화하여 학습 효율 향상         |
| **스파스 데이터 처리**  | 결측값을 자동 감지하여 최적 분기 생성       |
| **Cache 최적화**   | 내부 메모리 접근 최적화로 빠른 연산        |

---

### (3) XGBoost의 목적 함수 (Objective Function)

XGBoost는 목적 함수 $Obj$를 다음과 같이 정의합니다:

$$
Obj = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

* $l$: 손실 함수 (예: MSE, Log loss 등)
* $\Omega(f_k)$: 복잡도 패널티 (트리의 깊이, 노드 수 등)

#### 복잡도 항

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

* $T$: 리프 노드의 수
* $w_j$: 리프 노드의 출력 값
* $\gamma$, $\lambda$: 정규화 계수

---

### (4) XGBoost 주요 하이퍼파라미터

| 파라미터                      | 설명                |
| ------------------------- | ----------------- |
| `n_estimators`            | 생성할 트리 수          |
| `max_depth`               | 트리 최대 깊이          |
| `learning_rate`           | 각 단계의 기여율         |
| `subsample`               | 훈련 샘플 비율 (과적합 방지) |
| `colsample_bytree`        | 트리마다 사용할 특성 비율    |
| `reg_alpha`, `reg_lambda` | L1, L2 정규화 계수     |

---
# 10장. 분류 모델에 대한 실습

## 1. 실습 코드

```python
# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 전처리
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
X.columns = ['Type', 'AirTemp', 'ProcTemp', 'RotSpeed', 'Torque', 'ToolWear']
y = df['Machine failure']

print(X.head())

# 4. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# 5. 정규화 (kNN, SVM만 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 정의
models = {
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
}

# 7. 결과 저장
results = {}

# 8. 학습 및 예측
for name, model in models.items():
    if name in ['kNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "confusion_matrix": cm,
        "classification_report": cr
    }

# 9. 혼동 행렬 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, result) in zip(axes, results.items()):
    sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 10. 성능 출력
for name, result in results.items():
    print(f"\n===== {name} =====")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("\nClassification Report:")
    print(pd.DataFrame(result["classification_report"]).transpose())
```

## 2. 특징 중요도

```python
# 11. 중요도 추출 및 비교
features = X.columns
dt_importance = models['Decision Tree'].feature_importances_
rf_importance = models['XGBoost'].feature_importances_

# 12. 시각화
import numpy as np
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(features))
width = 0.35

bars1 = ax.bar(x - width/2, dt_importance, width, label='Decision Tree')
bars2 = ax.bar(x + width/2, rf_importance, width, label='Random Forest')

# 상단에 값 표시
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# 그래프 설정
ax.set_ylabel("Feature Importance")
ax.set_title("Feature Importance: Decision Tree vs Random Forest")
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

>>>>>>> 6759c28b4288f0c35ae1f2157cef39a269f336a9
```