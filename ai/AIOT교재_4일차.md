<<<<<<< HEAD
# - https://shorturl.at/aLUfI (대문자 i)

# - https://shorturl.at/3nQFr
--- 
# 📘 **5부. 딥러닝 기술과 활용**
---

# 📘 **1장. 심층 신경망의 개요와 구조**



심층 신경망(Deep Neural Network, DNN)은 기존의 인공신경망(ANN, Artificial Neural Network)에서 발전한 구조로, 2006년 Geoffrey Hinton과 Yoshua Bengio 등이 제안한 **딥러닝(Deep Learning)** 개념이 널리 퍼지면서 주목받기 시작하였습니다. 초기에는 XOR 문제 해결이나 음성 인식 등에 활용되었지만, **2012년 ILSVRC에서 AlexNet이 대회 우승**을 하면서 이미지, 음성, 시계열 데이터 등 다양한 도메인에서 DNN이 표준으로 자리 잡게 되었습니다.
- [위키피디아:심층신경망](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- [위키백과: 딥러닝](https://ko.wikipedia.org/wiki/%EB%94%A5_%EB%9F%AC%EB%8B%9D)
- [위키독스: deep neural network](https://wikidocs.net/120152)
---


## 1. 인공신경망(ANN)의 구조 

| 계층 유형             | 설명         |
| ----------------- | ---------- |
| 입력층(Input Layer)  | 데이터 입력을 담당 |
| 은닉층(Hidden Layer) | 내부 특징 추출   |
| 출력층(Output Layer) | 최종 예측값 생성  |

## 2. 심층 신경망(DNN)이란?

* 은닉층이 **2개 이상**인 ANN 구조를 DNN이라 합니다.
* 각 계층은 **가중치(weight)**, **편향(bias)**, \*\*활성화 함수(activation)\*\*로 구성됩니다.

## 3. 주요 활성화 함수 비교

| 함수      | 수식                                  | 특징                    |
| ------- | ----------------------------------- | --------------------- |
| ReLU    | $f(x) = \max(0, x)$                 | 빠른 학습, 음수에서 죽는 뉴런 가능성 |
| Sigmoid | $\frac{1}{1 + e^{-x}}$              | 출력 0\~1, 경사 소실 문제 존재  |
| Tanh    | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | -1\~1 출력, 중심화된 출력     |

---

## 4. 복습 문제

### 문제 1. 다층 퍼셉트론(Multilayer Perceptron, MLP)의 기본 구성 요소 3가지를 기술하시오.

🟩 **정답:** 입력층, 은닉층, 출력층

📝 **설명:** MLP는 최소 하나 이상의 은닉층과 비선형 활성화 함수를 포함해야 학습할 수 있습니다.

---

### 문제 2. ReLU 함수의 주요 단점은 무엇인가요?

🟩 **정답:** 음수 입력에 대해 출력이 0이 되어 학습이 멈출 수 있음 (죽은 뉴런 문제)

📝 **설명:** 이를 해결하기 위해 Leaky ReLU, ELU 등이 제안되었습니다.

---

## 5. 실습: Ai4I 2020 Dataset을 활용한 MLP 분류


| 단계         | 내용                                    |
| ---------- | ------------------------------------- |
| **입력 데이터** | Type, 온도, 회전 속도, 토크, 마모 시간            |
| **출력 (y)** | 고장(1), 정상 (0)|
| **모델 구조**  | 64 → 32 → 1 (sigmoid)                |
| **손실 함수**  | `binary_crossentropy` (다중 이진 타겟용)     |



```python
# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 2. 데이터 로드 및 전처리
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

# 3. 특성과 타겟 추출
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

# 4. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. MLP 모델 구성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 학습 수행
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 7. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 8. 학습 곡선 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
```

---

# 📘 **2장. 학습과 최적화 알고리즘**



초기 신경망 학습 방식은 단순한 \*\*경사하강법(Gradient Descent)\*\*이었습니다. 하지만 층이 깊어지면서 학습이 어려워지는 **기울기 소실(vanishing gradient)** 문제가 발생했고, 1986년 Rumelhart, Hinton, Williams가 제안한 **역전파(Backpropagation)** 알고리즘이 이를 해결하는 핵심으로 자리잡았습니다.
이후 학습 안정성과 속도 향상을 위해 다양한 \*\*최적화 알고리즘(SGD, Momentum, RMSProp, Adam 등)\*\*이 제안되며, 딥러닝의 폭발적 발전이 가능해졌습니다.

---


## 1. 순전파(Forward Propagation)

* 입력 데이터를 계층을 따라 \*\*선형 변환(Wx + b)\*\*과 **비선형 활성화 함수**를 통해 출력까지 전달합니다.
* 예시:

  $$
  z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)} \quad,\quad a^{(l)} = \text{ReLU}(z^{(l)})
  $$

---

## 2. 손실 함수(Loss Function)

* **분류:** Binary Cross Entropy

  $$
  \mathcal{L} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  $$
* **회귀:** MSE, MAE 사용

---

## 3. 역전파(Backpropagation)

* 손실 함수의 오차를 바탕으로 각 층의 가중치에 대해 **체인룰로 미분**하여 기울기를 계산합니다.
* 각 계층의 가중치와 편향을 다음 식으로 업데이트합니다:

  $$
  W := W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
  $$

  $$
  b := b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
  $$

---

## 4. 최적화 알고리즘 비교

| 알고리즘     | 설명                 | 특징        |
| -------- | ------------------ | --------- |
| SGD      | 확률적 경사하강법          | 불안정하지만 빠름 |
| Momentum | 속도 개념 도입           | 진동 억제     |
| RMSProp  | 최근 기울기 제곱 평균       | 학습률 자동 조절 |
| Adam     | Momentum + RMSProp | 가장 널리 사용됨 |

---

## 5. 학습 관련 용어

| 용어            | 정의                     |
| ------------- | ---------------------- |
| Batch         | 한 번 학습에 사용하는 샘플 수      |
| Epoch         | 전체 데이터를 한 번 모두 학습      |
| Learning Rate | 가중치 업데이트 폭             |
| 정규화           | 과적합 방지 (L2, Dropout 등) |

---

## 6. 실습: Optimizer에 따른 학습 성능 비교 (Ai4I 2020 Dataset)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
# 동일하므로 아래서부터 수행

# 모델 생성 함수
def create_model(optimizer):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 최적화기 별 학습 결과 저장
optimizers = {'SGD': SGD(), 'RMSprop': RMSprop(), 'Adam': Adam()}
histories = {}

for name, opt in optimizers.items():
    print(f"\n[ {name} Optimizer 학습 시작 ]")
    model = create_model(opt)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    histories[name] = history
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"{name} Optimizer 성능:\n", classification_report(y_test, y_pred))

# 정확도 시각화
plt.figure(figsize=(10, 5))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc')
plt.title("Optimizer별 Validation Accuracy 비교")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 📘 **3장. 모델 성능 향상 전략**


초기 딥러닝 모델은 **학습이 느리고 불안정**하며, **과적합(overfitting)** 문제가 자주 발생했습니다. 이를 해결하기 위해 2010년대 초반 다양한 기술이 등장했습니다.

* **Dropout**(2014, Hinton)은 은닉 뉴런을 무작위로 제거해 일반화 성능을 향상시켰고,
* **Batch Normalization**(2015, Ioffe & Szegedy)은 학습 안정성과 속도를 크게 개선했습니다.
* 또한, **Early Stopping**, **Regularization**, **Hyperparameter Tuning** 등도 실무에서 중요한 전략으로 자리잡았습니다.

---


## 1. 초기화 전략 (Weight Initialization)

| 방법        | 설명          | 특징             |
| --------- | ----------- | -------------- |
| Zero      | 0으로 초기화     | 학습되지 않음 (비권장)  |
| Random    | 난수 초기화      | 계층마다 불균형 가능    |
| He/Xavier | 활성화 함수에 최적화 | ReLU/탄젠트용으로 추천 |

---

## 2. 활성화 함수 선택 가이드

* **ReLU**: 빠른 수렴, 음수 입력에 민감
* **Leaky ReLU**: 음수 입력 보완
* **ELU/SELU**: 고급 네트워크에서 활용

---

## 3. 정규화 기법

| 기법        | 설명           | 효과         |
| --------- | ------------ | ---------- |
| L1        | 가중치의 절댓값 합   | 희소한 모델 생성  |
| L2        | 가중치 제곱합      | 일반적 과적합 방지 |
| Dropout   | 일부 뉴런 무작위 제거 | 앙상블 효과 유사  |
| BatchNorm | 배치 단위 정규화    | 학습 안정성 증가  |

---

## 4. Early Stopping

* 검증 정확도가 **향상되지 않으면 조기 종료**
* `patience` 설정을 통해 n epoch 동안 개선 없을 경우 중단

---

## 5. 하이퍼파라미터 튜닝

| 요소            | 설명            |
| ------------- | ------------- |
| Learning Rate | 가장 민감한 파라미터   |
| Batch Size    | 메모리-속도 균형     |
| Layer 수       | 복잡도와 학습 시간 조정 |
| Hidden Unit 수 | 과소/과적합 조절 도구  |
| Activation    | 비선형성 선택       |

---

## 6. 복습 문제

### 문제 1. Dropout이 적용된 모델의 일반화 성능이 향상되는 이유는 무엇인가요?

🟩 **정답:** 학습 시 일부 뉴런을 무작위로 제거하여 과적합을 방지하고, 다양한 경로를 통해 학습되기 때문에 앙상블 효과가 발생합니다.

---

### 문제 2. 하이퍼파라미터 중 모델의 학습률이 너무 작을 경우 어떤 문제가 발생하나요?

🟩 **정답:** 수렴 속도가 매우 느려져 학습 시간이 비효율적으로 길어짐

---

## 7. 실습: Early Stopping과 Dropout을 이용한 모델 성능 향상

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 콜백
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop], verbose=1)

# 성능 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss with EarlyStopping")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 📘 **4장. DNN을 활용한 IoT 데이터 분류**


## 1. 분류(Classification) 문제 정의

* 입력: 다양한 센서 특성 (온도, 회전 속도, 토크, 해충 사진 등)
* 출력: 이벤트 발생 여부 (예: 기계 고장 여부, 해충 종류)

---

## 2. 분류 모델에서의 DNN 구성


| 항목         | 이진 분류               | 다중 분류 (one-hot encoding)               |
| ---------- | ------------------- | -------------------------- |
| 출력층 노드 수   | 1                   | N개 (해충 종류 수)               |
| 출력층 활성화 함수 | Sigmoid             | Softmax         |
| 손실 함수      | Binary Crossentropy | Categorical Crossentropy (one-hot encoding) |
| 예측 방식      | 0 or 1    (0~1 사이 확률)  | 각 라벨별로 0 또는 1              |
| 예시         | 기계 고장 여부            | 해충 종류    |

---

## 3. 성능 평가 지표

| 지표               | 정의                 | 특징                   |
| ---------------- | ------------------ | -------------------- |
| Accuracy         | 정확도 = (TP+TN)/(전체) | 클래스 불균형 시 왜곡 가능      |
| Precision        | 정밀도 = TP / (TP+FP) | 예측이 맞은 비율            |
| Recall           | 재현율 = TP / (TP+FN) | 실제 양성 중 맞춘 비율        |
| F1-score         | 정밀도와 재현율 조화 평균     | 불균형 클래스에 적합          |
| Confusion Matrix | 예측 vs 실제 분포표       | TP, TN, FP, FN 구조 확인 |

- 혼동 행렬은 다음과 같이 해석합니다:

| 항목 | 의미 |
|------|------|
| **True 1, Pred 1** | 올바르게 고장 예측 (True Positive) |
| **True 0, Pred 0** | 정상 장비 정확 예측 (True Negative) |
| **True 0, Pred 1** | 정상인데 고장으로 예측 (False Positive) |
| **True 1, Pred 0** | 고장인데 정상으로 예측 (False Negative) |
---

## 4. 실전 적용을 위한 고려 사항

* 입력 변수의 **스케일링 필수**
* 클래스 불균형이 심한 경우 **가중치 조정 또는 샘플링 기법** 고려
* 실시간 IoT 적용을 위한 **모델 경량화** 및 **배포 전략** 필요

---

## 5. 복습 문제

### 문제 1. 이진 분류 문제에서 출력층의 활성화 함수로 적절한 것은 무엇인가요?

🟩 **정답:** Sigmoid
📝 **설명:** Sigmoid는 0\~1 사이의 값을 출력하며, 확률로 해석할 수 있어 이진 분류에 적합합니다.

---
## 6. 실습: 다중 분류 문제

### (1) 문제
* **스마트 시티의 네트워크 트래픽 특성 기반 침입 탐지 및 공격 유형 분류**하는 문제.
* 데이터 셋: 공개 데이터인 [UNSW-NB15 네트워크 침입 탐지 데이터셋](https://figshare.com/articles/dataset/UNSW_NB15_training-set_csv/29149946?file=54850502)
- 사이트에서 **UNSW_NB15_training-set.csv** 파일을 다운로드하고,  Colab 환경에 업로드합니다.


#### 📘 `UNSW_NB15_training-set.csv` 파일의 주요 컬럼들


| 컬럼 이름     | 설명                                 |
| --------- | ---------------------------------- |
| `id`      | 각 행의 고유 식별자                        |
| `dur`     | 세션 지속 시간 (초 단위)                    |
| `proto`   | 사용된 전송 프로토콜 (예: TCP, UDP, ICMP 등)  |
| `service` | 목적지 서비스 유형 (예: HTTP, FTP, SMTP 등)  |
| `state`   | 트래픽 세션의 상태 코드 (예: FIN, REJ, INT 등) |
| `spkts`   | 소스에서 보낸 패킷 수                       |
| `dpkts`   | 목적지에서 보낸 패킷 수                      |
| `sbytes`  | 소스가 전송한 바이트 수                      |
| `dbytes`  | 목적지가 전송한 바이트 수                     |
| `rate`    | 평균 패킷 전송률                          |
| `sload`, `dload`   | 소스/목적지의 데이터 전송 속도 (bps)                |
| `stcpb`, `dtcpb`   | TCP 초기 시퀀스 번호                          |
| `tcprtt`           | TCP 왕복 시간 (RTT)                        |
| `synack`, `ackdat` | TCP 핸드셰이크 관련 시간 정보                     |
| `is_sm_ips_ports`  | 동일한 IP/포트를 사용했는지 여부                    |
| `attack_cat`       | 공격 유형 분류 (예: Fuzzers, Exploits, DoS 등) |
| `label`            | 이진 레이블 (0 = 정상, 1 = 공격)                |


* `attack_cat` 컬럼은 다중 클래스 레이블로 활용 가능
* `label` 컬럼은 이진 분류용 레이블
* 대부분의 수치형 컬럼은 **정규화 또는 스케일링 필요**
* `proto`, `service`, `state` 등은 **범주형 인코딩 필요**


#### 📘 공격 유형(`attack_cat` 컬럼): 10종류


| 인덱스 | 공격 유형 (`attack_cat`)        | 설명                                                                  |
| --- | --------------------------- | ------------------------------------------------------------------- |
| 0   | **Analysis**                | 포트 스캔, 패킷 캡처, IDS 회피 등 시스템 정보를 수집하거나 보안 취약점을 분석하기 위한 활동. 종종 정찰이나 침투 준비 단계에서 사용됩니다.         |
| 1   | **Backdoor**                | 시스템에 몰래 접근하는 경로를 만드는 공격 (ex: 원격 셸 생성). 시스템에 비정상적으로 접근하기 위해 설치된 비밀 경로를 통해 인증 없이 접근 가능한 공격입니다.               |
| 2   | **DoS (Denial of Service)** | 서비스 거부 공격. 과도한 트래픽으로 자원을 소모시켜 정상 서비스 방해. 서비스 거부 공격으로, 네트워크 또는 시스템을 마비시켜 정상 사용자가 서비스를 이용하지 못하도록 만듭니다.         |
| 3   | **Exploits**                | 보안 취약점을 이용하여 시스템 제어권을 획득하는 공격. 소프트웨어, 하드웨어, 또는 OS의 보안 취약점을 악용하여 권한 상승 또는 명령 실행 등을 유도하는 공격입니다.      |
| 4   | **Fuzzers**                 | 시스템 또는 네트워크에 비정상적인 입력값을 무작위로 삽입하여 충돌이나 취약점을 유발하는 공격. 다양한 입력 값을 무작위로 전송하여 시스템의 취약점을 발견하려는 기법. 종종 시스템 충돌이나 예외 발생을 유도합니다.   |
| 5   | **Generic**                 | 암호화 알고리즘의 일반적 약점을 활용하여 공격하는 방식. 예를 들어, 동일한 키나 알고리즘을 사용하는 시스템을 노립니다. |
| 6   | **Normal**                  | 정상적인 네트워크 트래픽으로, 공격이 아닌 일반적인 사용자 행위를 나타냅니다.                         |
| 7   | **Reconnaissance**          | 포트 스캐닝, 네트워크 매핑 등과 같은 정보 수집 행위. 공격을 위한 사전 조사에 해당합니다.                |
| 8   | **Shellcode**               | 쉘 명령을 실행하기 위한 바이너리 코드로, 주로 취약점을 악용해 시스템 명령어를 실행하도록 합니다.             |
| 9   | **Worms**                   | 자체 복제 및 확산 기능을 갖춘 악성코드 (네트워크 기반 감염). 네트워크를 통해 스스로 복제 및 확산되며 다른 시스템을 감염시키는 악성 프로그램입니다.                    |



* `attack_cat` 컬럼은 `LabelEncoder`를 사용해 `0~9`의 정수형 클래스로 변환됨.
* 이후 `to_categorical()` 처리해 **다중 클래스 분류의 타겟**으로 활용됩니다.
* `Normal`과 나머지 공격들을 구분하면 **이진 분류**, 전체를 구분하면 **다중 분류 문제**로 접근합니다.

---

### (2) 다중 분류에서의 딥러닝 구조

* 출력층 노드 수 = 클래스 수 (예: 10 가지의 공격 유형 → 10개 노드)
* 출력층 활성화 함수: `softmax`
* 손실 함수: `categorical_crossentropy`

---
### (3) 실습 코드

#### 1. 필요한 라이브러리와 데이터 로드

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 로드
df = pd.read_csv('/content/UNSW_NB15_training-set.csv')
df.head()

df.info()

df.nunique()

df.isnull().sum()

df.isin([np.inf, -np.inf]).sum()
```

---

#### 2. 특성과 타겟 분리

```python

# 특성과 타겟 분리: id와  label 불필요
X = df.drop(columns=['id', 'attack_cat', 'label'])
y = df['attack_cat']

# 결측치 처리
#X.replace([np.inf, -np.inf], np.nan, inplace=True)
#X.fillna(0, inplace=True)

# 범주형 변수 인코딩
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# 레이블 인코딩 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_
print(class_labels)

#  원-핫 인코딩
y_onehot = to_categorical(y_encoded)

```

---

#### 3. 훈련·검증 데이터 분할 및 스케일링

```python

# 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(
		X, y_onehot, test_size=0.2, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


```

---

#### 4. MLP 다중 분류 모델 구축

```python

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', 
		loss='categorical_crossentropy', 
		metrics=['accuracy'])
```

---

#### 5. 모델 학습 및 평가

```python


## EarlyStopping + ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("unsw_nb15_model.h5", save_best_only=True, monitor='val_loss')

# 학습
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
                    validation_split=0.2, 
		    callbacks=[early_stop, checkpoint], verbose=1)
# 예측 및 평가
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=class_labels))

# 혼동 행렬 시각화
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

---

#### 6. 학습 곡선 시각화

```python
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()
```

----
## 7. 다중분류 같지만 다중분류가 아닌 멀티 라벨 분류 문제

Ai4I 2020 Predictive Maintenance Dataset에서 고장의 **원인을 예측하는 다중 분류 문제** 


> 다중분류의 경우 출력층에 있는 뉴런 중 오직 한 개의 뉴런만 출력 '1'이 나옵니다.
> 그러나, 기계 고장 문제는 이 문제는 **Multi-label classification**입니다.
> 즉, 한 장비에 여러 개의 고장 유형이 동시에 발생할 수 있습니다.
> 한 장비에 여러 개의 노드가 1을 출력할 수 있으며 이는 **다중 고장 동시 발생 가능성**을 의미합니다.
> 이런 경우에는 모델의 구성 코드가 달라집니다.
---

### (1) 멀티 라벨 분류에서의 딥러닝 구조

* 출력층 노드 수 = 클래스 수 (예: 5개 고장 원인 → 5개 노드)
* 출력층 활성화 함수: `sigmoid`
* 손실 함수: `binary_crossentropy`


| 고장 유형 | 예측 값 (출력 노드) | 의미              |
| ----- | ------------ | --------------- |
| TWF   | 1            | 공구 마모로 인한 고장 발생 |
| HDF   | 0            | 히터 고장 없음        |
| ...   | ...          | ...             |

* 한 장비에 여러 개의 노드가 1을 출력할 수 있으며 이는 **다중 고장 동시 발생 가능성**을 의미합니다.

---

### (2) 실습 코드: 다중 고장 원인 분류

| 단계         | 내용                                    |
| ---------- | ------------------------------------- |
| **입력 데이터** | Type, 온도, 회전 속도, 토크, 마모 시간            |
| **출력 (y)** | 다중 이진 고장 유형 (TWF, HDF, PWF, OSF, RNF) |
| **모델 구조**  | 128 → 64 → 5 (sigmoid)                |
| **손실 함수**  | `binary_crossentropy` (다중 이진 타겟용)     |
| **활성화 함수** | ReLU + sigmoid 출력                     |
| **예측 해석**  | 각 고장 유형별 0.5 임계값 적용                   |
| **성능 평가**  | 각 고장 유형별 `classification_report` 출력   |



```python
# 1. 라이브러리 로드
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 데이터 전처리
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
 # 고장 정보가 없는 행(모든 열이 0) 을 삭제
df = df[df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0]
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
X.columns = ['Type', 'AirTemp', 'ProcTemp', 'RotSpeed', 'Torque', 'ToolWear']

y = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

# 4. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y

y.sum()

y_test

y_test.sum()


```

```python
# 5. 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
    Dense(64, activation='relu'),
        Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate = 0.002), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 6. 콜백 설정 (EarlyStopping & ModelCheckpoint)
early_stop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# 7. 학습
history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

# 8. 모델 예측
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 9. 각 클래스별 평가 및 혼동 행렬 시각화
for i, col in enumerate(y.columns):
    print(f"\n[고장 유형: {col}]")
    print(classification_report(y_test[col], y_pred[:, i]))

    cm = confusion_matrix(y_test[col], y_pred[:, i])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix: {col}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# 10. 학습 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Multi-label Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

```

---

# 📘 **5장. DNN을 활용한 회귀**

## 1. 회귀 문제의 정의

* 입력: 센서 특성 (온도, 토크, 회전수 등)
* 출력: **연속값** (예: 고장 시점까지 남은 시간, 마모 시간 등)

---

## 2. 분류와 회귀의 주요 차이

| 항목         | 분류                       | 회귀                    |
| ---------- | ------------------------ | --------------------- |
| 출력값        | 범주 (0, 1, 다중 클래스)        | 연속형 수치                |
| 출력층 노드     | 1개 이상 (sigmoid, softmax) | 1개 (선형)               |
| 출력층 활성화 함수 | sigmoid/softmax          | **None (Linear)**     |
| 손실 함수      | crossentropy             | **MSE / MAE / Huber** |

---

## 3. 회귀 모델의 성능 평가 지표

| 지표                        | 수식                                 | 해석                   |   |               |
| ------------------------- | ---------------------------------- | -------------------- | - | ------------- |
| MSE (Mean Squared Error)  | $\frac{1}{n} \sum (y - \hat{y})^2$ | 오차 제곱 평균 (민감)        |   |               |
| MAE (Mean Absolute Error) | ( \frac{1}{n} \sum                 | y - \hat{y}          | ) | 절대 오차 평균 (강건) |
| RMSE                      | $\sqrt{MSE}$                       | 해석 용이                |   |               |
| R² Score                  | $1 - \frac{SS_{res}}{SS_{tot}}$    | 설명력 지표 (1에 가까울수록 좋음) |   |               |

---

## 4. 복습 문제

### 문제 1. 회귀 문제에서 출력층의 활성화 함수로 적절한 것은?

🟩 **정답:** 없음 (선형)
📝 **설명:** 회귀에서는 연속값을 그대로 출력하므로 비선형 함수가 필요 없습니다.

---

### 문제 2. MSE와 MAE의 차이점은 무엇인가요?

🟩 **정답:** MSE는 오차 제곱을 사용하여 큰 오차에 민감하고, MAE는 절대값을 사용하여 이상치에 강건합니다.

---

### 문제 3. 다음 중 R² Score가 1에 가까울수록 의미하는 바는?

🟩 **정답:** 모델이 데이터를 잘 설명하고 예측력이 높다는 뜻입니다.

---

## 5. 실습: **DNN 회귀 기반 수확량 예측**

### (1) 주요 변경

| 항목 | 선형 회귀 | MLP 회귀 |
|------|------------|------------|
| 모델 | `LinearRegression()` | `Keras Sequential` |
| 손실 함수 | MSE | MSE |
| 출력층 | 선형 | 선형 (`Dense(1)`) |
| 평가 | RMSE, R² | 동일 |

### (2) 적용 기능

| 기능 | 적용 내용 |
|------|-----------|
| **정규화** | `kernel_regularizer=l2(0.001)` |
| **Dropout** | 첫 번째 은닉층 30%, 두 번째 은닉층 20% |
| **EarlyStopping** | `patience=10`, `restore_best_weights=True` |

### (3) 실습 코드

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

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

# 5. 병합 및 결측치 제거
df = pd.merge(climate_daily, production_daily, on='Time')
df.dropna(inplace=True)

# 6. X, y 분리
X = df[numerical_cols]
y = df['ProdA']

# 7. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# 9. 모델 구성 (L2 정규화 + Dropout 포함)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),
          kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1)  # 회귀 문제: 선형 출력
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 10. EarlyStopping 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 11. 모델 학습
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0)
```

```python

# 12. 예측 및 평가
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" RMSE (MLP): {rmse:.2f}")
print(f" R² Score (MLP): {r2:.2f}")

# 13. 예측 vs 실제값 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Ground Truth')
plt.plot(y_pred, label='Predicted (MLP)', linestyle='--')
plt.title("Tomato Production Prediction (MLP + Reg + Dropout)")
plt.xlabel("Sample Index")
plt.ylabel("Production (kg/m²)")
plt.legend()
plt.grid()
plt.show()

# 14. 학습 곡선 시각화
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE History with EarlyStopping")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid()
plt.show()

```

---

=======
# - https://shorturl.at/aLUfI (대문자 i)

# - https://shorturl.at/3nQFr
--- 
# 📘 **5부. 딥러닝 기술과 활용**
---

# 📘 **1장. 심층 신경망의 개요와 구조**



심층 신경망(Deep Neural Network, DNN)은 기존의 인공신경망(ANN, Artificial Neural Network)에서 발전한 구조로, 2006년 Geoffrey Hinton과 Yoshua Bengio 등이 제안한 **딥러닝(Deep Learning)** 개념이 널리 퍼지면서 주목받기 시작하였습니다. 초기에는 XOR 문제 해결이나 음성 인식 등에 활용되었지만, **2012년 ILSVRC에서 AlexNet이 대회 우승**을 하면서 이미지, 음성, 시계열 데이터 등 다양한 도메인에서 DNN이 표준으로 자리 잡게 되었습니다.
- [위키피디아:심층신경망](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- [위키백과: 딥러닝](https://ko.wikipedia.org/wiki/%EB%94%A5_%EB%9F%AC%EB%8B%9D)
- [위키독스: deep neural network](https://wikidocs.net/120152)
---


## 1. 인공신경망(ANN)의 구조 

| 계층 유형             | 설명         |
| ----------------- | ---------- |
| 입력층(Input Layer)  | 데이터 입력을 담당 |
| 은닉층(Hidden Layer) | 내부 특징 추출   |
| 출력층(Output Layer) | 최종 예측값 생성  |

## 2. 심층 신경망(DNN)이란?

* 은닉층이 **2개 이상**인 ANN 구조를 DNN이라 합니다.
* 각 계층은 **가중치(weight)**, **편향(bias)**, \*\*활성화 함수(activation)\*\*로 구성됩니다.

## 3. 주요 활성화 함수 비교

| 함수      | 수식                                  | 특징                    |
| ------- | ----------------------------------- | --------------------- |
| ReLU    | $f(x) = \max(0, x)$                 | 빠른 학습, 음수에서 죽는 뉴런 가능성 |
| Sigmoid | $\frac{1}{1 + e^{-x}}$              | 출력 0\~1, 경사 소실 문제 존재  |
| Tanh    | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | -1\~1 출력, 중심화된 출력     |

---

## 4. 복습 문제

### 문제 1. 다층 퍼셉트론(Multilayer Perceptron, MLP)의 기본 구성 요소 3가지를 기술하시오.

🟩 **정답:** 입력층, 은닉층, 출력층

📝 **설명:** MLP는 최소 하나 이상의 은닉층과 비선형 활성화 함수를 포함해야 학습할 수 있습니다.

---

### 문제 2. ReLU 함수의 주요 단점은 무엇인가요?

🟩 **정답:** 음수 입력에 대해 출력이 0이 되어 학습이 멈출 수 있음 (죽은 뉴런 문제)

📝 **설명:** 이를 해결하기 위해 Leaky ReLU, ELU 등이 제안되었습니다.

---

## 5. 실습: Ai4I 2020 Dataset을 활용한 MLP 분류


| 단계         | 내용                                    |
| ---------- | ------------------------------------- |
| **입력 데이터** | Type, 온도, 회전 속도, 토크, 마모 시간            |
| **출력 (y)** | 고장(1), 정상 (0)|
| **모델 구조**  | 64 → 32 → 1 (sigmoid)                |
| **손실 함수**  | `binary_crossentropy` (다중 이진 타겟용)     |



```python
# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 2. 데이터 로드 및 전처리
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

# 3. 특성과 타겟 추출
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

# 4. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. MLP 모델 구성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 학습 수행
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 7. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 8. 학습 곡선 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
```

---

# 📘 **2장. 학습과 최적화 알고리즘**



초기 신경망 학습 방식은 단순한 \*\*경사하강법(Gradient Descent)\*\*이었습니다. 하지만 층이 깊어지면서 학습이 어려워지는 **기울기 소실(vanishing gradient)** 문제가 발생했고, 1986년 Rumelhart, Hinton, Williams가 제안한 **역전파(Backpropagation)** 알고리즘이 이를 해결하는 핵심으로 자리잡았습니다.
이후 학습 안정성과 속도 향상을 위해 다양한 \*\*최적화 알고리즘(SGD, Momentum, RMSProp, Adam 등)\*\*이 제안되며, 딥러닝의 폭발적 발전이 가능해졌습니다.

---


## 1. 순전파(Forward Propagation)

* 입력 데이터를 계층을 따라 \*\*선형 변환(Wx + b)\*\*과 **비선형 활성화 함수**를 통해 출력까지 전달합니다.
* 예시:

  $$
  z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)} \quad,\quad a^{(l)} = \text{ReLU}(z^{(l)})
  $$

---

## 2. 손실 함수(Loss Function)

* **분류:** Binary Cross Entropy

  $$
  \mathcal{L} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  $$
* **회귀:** MSE, MAE 사용

---

## 3. 역전파(Backpropagation)

* 손실 함수의 오차를 바탕으로 각 층의 가중치에 대해 **체인룰로 미분**하여 기울기를 계산합니다.
* 각 계층의 가중치와 편향을 다음 식으로 업데이트합니다:

  $$
  W := W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
  $$

  $$
  b := b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
  $$

---

## 4. 최적화 알고리즘 비교

| 알고리즘     | 설명                 | 특징        |
| -------- | ------------------ | --------- |
| SGD      | 확률적 경사하강법          | 불안정하지만 빠름 |
| Momentum | 속도 개념 도입           | 진동 억제     |
| RMSProp  | 최근 기울기 제곱 평균       | 학습률 자동 조절 |
| Adam     | Momentum + RMSProp | 가장 널리 사용됨 |

---

## 5. 학습 관련 용어

| 용어            | 정의                     |
| ------------- | ---------------------- |
| Batch         | 한 번 학습에 사용하는 샘플 수      |
| Epoch         | 전체 데이터를 한 번 모두 학습      |
| Learning Rate | 가중치 업데이트 폭             |
| 정규화           | 과적합 방지 (L2, Dropout 등) |

---

## 6. 실습: Optimizer에 따른 학습 성능 비교 (Ai4I 2020 Dataset)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
# 동일하므로 아래서부터 수행

# 모델 생성 함수
def create_model(optimizer):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 최적화기 별 학습 결과 저장
optimizers = {'SGD': SGD(), 'RMSprop': RMSprop(), 'Adam': Adam()}
histories = {}

for name, opt in optimizers.items():
    print(f"\n[ {name} Optimizer 학습 시작 ]")
    model = create_model(opt)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    histories[name] = history
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"{name} Optimizer 성능:\n", classification_report(y_test, y_pred))

# 정확도 시각화
plt.figure(figsize=(10, 5))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc')
plt.title("Optimizer별 Validation Accuracy 비교")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 📘 **3장. 모델 성능 향상 전략**


초기 딥러닝 모델은 **학습이 느리고 불안정**하며, **과적합(overfitting)** 문제가 자주 발생했습니다. 이를 해결하기 위해 2010년대 초반 다양한 기술이 등장했습니다.

* **Dropout**(2014, Hinton)은 은닉 뉴런을 무작위로 제거해 일반화 성능을 향상시켰고,
* **Batch Normalization**(2015, Ioffe & Szegedy)은 학습 안정성과 속도를 크게 개선했습니다.
* 또한, **Early Stopping**, **Regularization**, **Hyperparameter Tuning** 등도 실무에서 중요한 전략으로 자리잡았습니다.

---


## 1. 초기화 전략 (Weight Initialization)

| 방법        | 설명          | 특징             |
| --------- | ----------- | -------------- |
| Zero      | 0으로 초기화     | 학습되지 않음 (비권장)  |
| Random    | 난수 초기화      | 계층마다 불균형 가능    |
| He/Xavier | 활성화 함수에 최적화 | ReLU/탄젠트용으로 추천 |

---

## 2. 활성화 함수 선택 가이드

* **ReLU**: 빠른 수렴, 음수 입력에 민감
* **Leaky ReLU**: 음수 입력 보완
* **ELU/SELU**: 고급 네트워크에서 활용

---

## 3. 정규화 기법

| 기법        | 설명           | 효과         |
| --------- | ------------ | ---------- |
| L1        | 가중치의 절댓값 합   | 희소한 모델 생성  |
| L2        | 가중치 제곱합      | 일반적 과적합 방지 |
| Dropout   | 일부 뉴런 무작위 제거 | 앙상블 효과 유사  |
| BatchNorm | 배치 단위 정규화    | 학습 안정성 증가  |

---

## 4. Early Stopping

* 검증 정확도가 **향상되지 않으면 조기 종료**
* `patience` 설정을 통해 n epoch 동안 개선 없을 경우 중단

---

## 5. 하이퍼파라미터 튜닝

| 요소            | 설명            |
| ------------- | ------------- |
| Learning Rate | 가장 민감한 파라미터   |
| Batch Size    | 메모리-속도 균형     |
| Layer 수       | 복잡도와 학습 시간 조정 |
| Hidden Unit 수 | 과소/과적합 조절 도구  |
| Activation    | 비선형성 선택       |

---

## 6. 복습 문제

### 문제 1. Dropout이 적용된 모델의 일반화 성능이 향상되는 이유는 무엇인가요?

🟩 **정답:** 학습 시 일부 뉴런을 무작위로 제거하여 과적합을 방지하고, 다양한 경로를 통해 학습되기 때문에 앙상블 효과가 발생합니다.

---

### 문제 2. 하이퍼파라미터 중 모델의 학습률이 너무 작을 경우 어떤 문제가 발생하나요?

🟩 **정답:** 수렴 속도가 매우 느려져 학습 시간이 비효율적으로 길어짐

---

## 7. 실습: Early Stopping과 Dropout을 이용한 모델 성능 향상

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 콜백
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop], verbose=1)

# 성능 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss with EarlyStopping")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 📘 **4장. DNN을 활용한 IoT 데이터 분류**


## 1. 분류(Classification) 문제 정의

* 입력: 다양한 센서 특성 (온도, 회전 속도, 토크, 해충 사진 등)
* 출력: 이벤트 발생 여부 (예: 기계 고장 여부, 해충 종류)

---

## 2. 분류 모델에서의 DNN 구성


| 항목         | 이진 분류               | 다중 분류 (one-hot encoding)               |
| ---------- | ------------------- | -------------------------- |
| 출력층 노드 수   | 1                   | N개 (해충 종류 수)               |
| 출력층 활성화 함수 | Sigmoid             | Softmax         |
| 손실 함수      | Binary Crossentropy | Categorical Crossentropy (one-hot encoding) |
| 예측 방식      | 0 or 1    (0~1 사이 확률)  | 각 라벨별로 0 또는 1              |
| 예시         | 기계 고장 여부            | 해충 종류    |

---

## 3. 성능 평가 지표

| 지표               | 정의                 | 특징                   |
| ---------------- | ------------------ | -------------------- |
| Accuracy         | 정확도 = (TP+TN)/(전체) | 클래스 불균형 시 왜곡 가능      |
| Precision        | 정밀도 = TP / (TP+FP) | 예측이 맞은 비율            |
| Recall           | 재현율 = TP / (TP+FN) | 실제 양성 중 맞춘 비율        |
| F1-score         | 정밀도와 재현율 조화 평균     | 불균형 클래스에 적합          |
| Confusion Matrix | 예측 vs 실제 분포표       | TP, TN, FP, FN 구조 확인 |

- 혼동 행렬은 다음과 같이 해석합니다:

| 항목 | 의미 |
|------|------|
| **True 1, Pred 1** | 올바르게 고장 예측 (True Positive) |
| **True 0, Pred 0** | 정상 장비 정확 예측 (True Negative) |
| **True 0, Pred 1** | 정상인데 고장으로 예측 (False Positive) |
| **True 1, Pred 0** | 고장인데 정상으로 예측 (False Negative) |
---

## 4. 실전 적용을 위한 고려 사항

* 입력 변수의 **스케일링 필수**
* 클래스 불균형이 심한 경우 **가중치 조정 또는 샘플링 기법** 고려
* 실시간 IoT 적용을 위한 **모델 경량화** 및 **배포 전략** 필요

---

## 5. 복습 문제

### 문제 1. 이진 분류 문제에서 출력층의 활성화 함수로 적절한 것은 무엇인가요?

🟩 **정답:** Sigmoid
📝 **설명:** Sigmoid는 0\~1 사이의 값을 출력하며, 확률로 해석할 수 있어 이진 분류에 적합합니다.

---
## 6. 실습: 다중 분류 문제

### (1) 문제
* **스마트 시티의 네트워크 트래픽 특성 기반 침입 탐지 및 공격 유형 분류**하는 문제.
* 데이터 셋: 공개 데이터인 [UNSW-NB15 네트워크 침입 탐지 데이터셋](https://figshare.com/articles/dataset/UNSW_NB15_training-set_csv/29149946?file=54850502)
- 사이트에서 **UNSW_NB15_training-set.csv** 파일을 다운로드하고,  Colab 환경에 업로드합니다.


#### 📘 `UNSW_NB15_training-set.csv` 파일의 주요 컬럼들


| 컬럼 이름     | 설명                                 |
| --------- | ---------------------------------- |
| `id`      | 각 행의 고유 식별자                        |
| `dur`     | 세션 지속 시간 (초 단위)                    |
| `proto`   | 사용된 전송 프로토콜 (예: TCP, UDP, ICMP 등)  |
| `service` | 목적지 서비스 유형 (예: HTTP, FTP, SMTP 등)  |
| `state`   | 트래픽 세션의 상태 코드 (예: FIN, REJ, INT 등) |
| `spkts`   | 소스에서 보낸 패킷 수                       |
| `dpkts`   | 목적지에서 보낸 패킷 수                      |
| `sbytes`  | 소스가 전송한 바이트 수                      |
| `dbytes`  | 목적지가 전송한 바이트 수                     |
| `rate`    | 평균 패킷 전송률                          |
| `sload`, `dload`   | 소스/목적지의 데이터 전송 속도 (bps)                |
| `stcpb`, `dtcpb`   | TCP 초기 시퀀스 번호                          |
| `tcprtt`           | TCP 왕복 시간 (RTT)                        |
| `synack`, `ackdat` | TCP 핸드셰이크 관련 시간 정보                     |
| `is_sm_ips_ports`  | 동일한 IP/포트를 사용했는지 여부                    |
| `attack_cat`       | 공격 유형 분류 (예: Fuzzers, Exploits, DoS 등) |
| `label`            | 이진 레이블 (0 = 정상, 1 = 공격)                |


* `attack_cat` 컬럼은 다중 클래스 레이블로 활용 가능
* `label` 컬럼은 이진 분류용 레이블
* 대부분의 수치형 컬럼은 **정규화 또는 스케일링 필요**
* `proto`, `service`, `state` 등은 **범주형 인코딩 필요**


#### 📘 공격 유형(`attack_cat` 컬럼): 10종류


| 인덱스 | 공격 유형 (`attack_cat`)        | 설명                                                                  |
| --- | --------------------------- | ------------------------------------------------------------------- |
| 0   | **Analysis**                | 포트 스캔, 패킷 캡처, IDS 회피 등 시스템 정보를 수집하거나 보안 취약점을 분석하기 위한 활동. 종종 정찰이나 침투 준비 단계에서 사용됩니다.         |
| 1   | **Backdoor**                | 시스템에 몰래 접근하는 경로를 만드는 공격 (ex: 원격 셸 생성). 시스템에 비정상적으로 접근하기 위해 설치된 비밀 경로를 통해 인증 없이 접근 가능한 공격입니다.               |
| 2   | **DoS (Denial of Service)** | 서비스 거부 공격. 과도한 트래픽으로 자원을 소모시켜 정상 서비스 방해. 서비스 거부 공격으로, 네트워크 또는 시스템을 마비시켜 정상 사용자가 서비스를 이용하지 못하도록 만듭니다.         |
| 3   | **Exploits**                | 보안 취약점을 이용하여 시스템 제어권을 획득하는 공격. 소프트웨어, 하드웨어, 또는 OS의 보안 취약점을 악용하여 권한 상승 또는 명령 실행 등을 유도하는 공격입니다.      |
| 4   | **Fuzzers**                 | 시스템 또는 네트워크에 비정상적인 입력값을 무작위로 삽입하여 충돌이나 취약점을 유발하는 공격. 다양한 입력 값을 무작위로 전송하여 시스템의 취약점을 발견하려는 기법. 종종 시스템 충돌이나 예외 발생을 유도합니다.   |
| 5   | **Generic**                 | 암호화 알고리즘의 일반적 약점을 활용하여 공격하는 방식. 예를 들어, 동일한 키나 알고리즘을 사용하는 시스템을 노립니다. |
| 6   | **Normal**                  | 정상적인 네트워크 트래픽으로, 공격이 아닌 일반적인 사용자 행위를 나타냅니다.                         |
| 7   | **Reconnaissance**          | 포트 스캐닝, 네트워크 매핑 등과 같은 정보 수집 행위. 공격을 위한 사전 조사에 해당합니다.                |
| 8   | **Shellcode**               | 쉘 명령을 실행하기 위한 바이너리 코드로, 주로 취약점을 악용해 시스템 명령어를 실행하도록 합니다.             |
| 9   | **Worms**                   | 자체 복제 및 확산 기능을 갖춘 악성코드 (네트워크 기반 감염). 네트워크를 통해 스스로 복제 및 확산되며 다른 시스템을 감염시키는 악성 프로그램입니다.                    |



* `attack_cat` 컬럼은 `LabelEncoder`를 사용해 `0~9`의 정수형 클래스로 변환됨.
* 이후 `to_categorical()` 처리해 **다중 클래스 분류의 타겟**으로 활용됩니다.
* `Normal`과 나머지 공격들을 구분하면 **이진 분류**, 전체를 구분하면 **다중 분류 문제**로 접근합니다.

---

### (2) 다중 분류에서의 딥러닝 구조

* 출력층 노드 수 = 클래스 수 (예: 10 가지의 공격 유형 → 10개 노드)
* 출력층 활성화 함수: `softmax`
* 손실 함수: `categorical_crossentropy`

---
### (3) 실습 코드

#### 1. 필요한 라이브러리와 데이터 로드

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 로드
df = pd.read_csv('/content/UNSW_NB15_training-set.csv')
df.head()

df.info()

df.nunique()

df.isnull().sum()

df.isin([np.inf, -np.inf]).sum()
```

---

#### 2. 특성과 타겟 분리

```python

# 특성과 타겟 분리: id와  label 불필요
X = df.drop(columns=['id', 'attack_cat', 'label'])
y = df['attack_cat']

# 결측치 처리
#X.replace([np.inf, -np.inf], np.nan, inplace=True)
#X.fillna(0, inplace=True)

# 범주형 변수 인코딩
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# 레이블 인코딩 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_
print(class_labels)

#  원-핫 인코딩
y_onehot = to_categorical(y_encoded)

```

---

#### 3. 훈련·검증 데이터 분할 및 스케일링

```python

# 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(
		X, y_onehot, test_size=0.2, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


```

---

#### 4. MLP 다중 분류 모델 구축

```python

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', 
		loss='categorical_crossentropy', 
		metrics=['accuracy'])
```

---

#### 5. 모델 학습 및 평가

```python


## EarlyStopping + ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("unsw_nb15_model.h5", save_best_only=True, monitor='val_loss')

# 학습
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
                    validation_split=0.2, 
		    callbacks=[early_stop, checkpoint], verbose=1)
# 예측 및 평가
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=class_labels))

# 혼동 행렬 시각화
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

---

#### 6. 학습 곡선 시각화

```python
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()
```

----
## 7. 다중분류 같지만 다중분류가 아닌 멀티 라벨 분류 문제

Ai4I 2020 Predictive Maintenance Dataset에서 고장의 **원인을 예측하는 다중 분류 문제** 


> 다중분류의 경우 출력층에 있는 뉴런 중 오직 한 개의 뉴런만 출력 '1'이 나옵니다.
> 그러나, 기계 고장 문제는 이 문제는 **Multi-label classification**입니다.
> 즉, 한 장비에 여러 개의 고장 유형이 동시에 발생할 수 있습니다.
> 한 장비에 여러 개의 노드가 1을 출력할 수 있으며 이는 **다중 고장 동시 발생 가능성**을 의미합니다.
> 이런 경우에는 모델의 구성 코드가 달라집니다.
---

### (1) 멀티 라벨 분류에서의 딥러닝 구조

* 출력층 노드 수 = 클래스 수 (예: 5개 고장 원인 → 5개 노드)
* 출력층 활성화 함수: `sigmoid`
* 손실 함수: `binary_crossentropy`


| 고장 유형 | 예측 값 (출력 노드) | 의미              |
| ----- | ------------ | --------------- |
| TWF   | 1            | 공구 마모로 인한 고장 발생 |
| HDF   | 0            | 히터 고장 없음        |
| ...   | ...          | ...             |

* 한 장비에 여러 개의 노드가 1을 출력할 수 있으며 이는 **다중 고장 동시 발생 가능성**을 의미합니다.

---

### (2) 실습 코드: 다중 고장 원인 분류

| 단계         | 내용                                    |
| ---------- | ------------------------------------- |
| **입력 데이터** | Type, 온도, 회전 속도, 토크, 마모 시간            |
| **출력 (y)** | 다중 이진 고장 유형 (TWF, HDF, PWF, OSF, RNF) |
| **모델 구조**  | 128 → 64 → 5 (sigmoid)                |
| **손실 함수**  | `binary_crossentropy` (다중 이진 타겟용)     |
| **활성화 함수** | ReLU + sigmoid 출력                     |
| **예측 해석**  | 각 고장 유형별 0.5 임계값 적용                   |
| **성능 평가**  | 각 고장 유형별 `classification_report` 출력   |



```python
# 1. 라이브러리 로드
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# 2. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(url)

# 3. 데이터 전처리
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
 # 고장 정보가 없는 행(모든 열이 0) 을 삭제
df = df[df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0]
df['Type'] = df['Type'].map({'H': 0, 'L': 1, 'M': 2})

X = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
X.columns = ['Type', 'AirTemp', 'ProcTemp', 'RotSpeed', 'Torque', 'ToolWear']

y = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

# 4. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y

y.sum()

y_test

y_test.sum()


```

```python
# 5. 모델 구성
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
    Dense(64, activation='relu'),
        Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate = 0.002), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 6. 콜백 설정 (EarlyStopping & ModelCheckpoint)
early_stop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# 7. 학습
history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

# 8. 모델 예측
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 9. 각 클래스별 평가 및 혼동 행렬 시각화
for i, col in enumerate(y.columns):
    print(f"\n[고장 유형: {col}]")
    print(classification_report(y_test[col], y_pred[:, i]))

    cm = confusion_matrix(y_test[col], y_pred[:, i])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix: {col}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# 10. 학습 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Multi-label Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

```

---

# 📘 **5장. DNN을 활용한 회귀**

## 1. 회귀 문제의 정의

* 입력: 센서 특성 (온도, 토크, 회전수 등)
* 출력: **연속값** (예: 고장 시점까지 남은 시간, 마모 시간 등)

---

## 2. 분류와 회귀의 주요 차이

| 항목         | 분류                       | 회귀                    |
| ---------- | ------------------------ | --------------------- |
| 출력값        | 범주 (0, 1, 다중 클래스)        | 연속형 수치                |
| 출력층 노드     | 1개 이상 (sigmoid, softmax) | 1개 (선형)               |
| 출력층 활성화 함수 | sigmoid/softmax          | **None (Linear)**     |
| 손실 함수      | crossentropy             | **MSE / MAE / Huber** |

---

## 3. 회귀 모델의 성능 평가 지표

| 지표                        | 수식                                 | 해석                   |   |               |
| ------------------------- | ---------------------------------- | -------------------- | - | ------------- |
| MSE (Mean Squared Error)  | $\frac{1}{n} \sum (y - \hat{y})^2$ | 오차 제곱 평균 (민감)        |   |               |
| MAE (Mean Absolute Error) | ( \frac{1}{n} \sum                 | y - \hat{y}          | ) | 절대 오차 평균 (강건) |
| RMSE                      | $\sqrt{MSE}$                       | 해석 용이                |   |               |
| R² Score                  | $1 - \frac{SS_{res}}{SS_{tot}}$    | 설명력 지표 (1에 가까울수록 좋음) |   |               |

---

## 4. 복습 문제

### 문제 1. 회귀 문제에서 출력층의 활성화 함수로 적절한 것은?

🟩 **정답:** 없음 (선형)
📝 **설명:** 회귀에서는 연속값을 그대로 출력하므로 비선형 함수가 필요 없습니다.

---

### 문제 2. MSE와 MAE의 차이점은 무엇인가요?

🟩 **정답:** MSE는 오차 제곱을 사용하여 큰 오차에 민감하고, MAE는 절대값을 사용하여 이상치에 강건합니다.

---

### 문제 3. 다음 중 R² Score가 1에 가까울수록 의미하는 바는?

🟩 **정답:** 모델이 데이터를 잘 설명하고 예측력이 높다는 뜻입니다.

---

## 5. 실습: **DNN 회귀 기반 수확량 예측**

### (1) 주요 변경

| 항목 | 선형 회귀 | MLP 회귀 |
|------|------------|------------|
| 모델 | `LinearRegression()` | `Keras Sequential` |
| 손실 함수 | MSE | MSE |
| 출력층 | 선형 | 선형 (`Dense(1)`) |
| 평가 | RMSE, R² | 동일 |

### (2) 적용 기능

| 기능 | 적용 내용 |
|------|-----------|
| **정규화** | `kernel_regularizer=l2(0.001)` |
| **Dropout** | 첫 번째 은닉층 30%, 두 번째 은닉층 20% |
| **EarlyStopping** | `patience=10`, `restore_best_weights=True` |

### (3) 실습 코드

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

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

# 5. 병합 및 결측치 제거
df = pd.merge(climate_daily, production_daily, on='Time')
df.dropna(inplace=True)

# 6. X, y 분리
X = df[numerical_cols]
y = df['ProdA']

# 7. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# 9. 모델 구성 (L2 정규화 + Dropout 포함)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),
          kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1)  # 회귀 문제: 선형 출력
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 10. EarlyStopping 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 11. 모델 학습
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0)
```

```python

# 12. 예측 및 평가
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" RMSE (MLP): {rmse:.2f}")
print(f" R² Score (MLP): {r2:.2f}")

# 13. 예측 vs 실제값 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Ground Truth')
plt.plot(y_pred, label='Predicted (MLP)', linestyle='--')
plt.title("Tomato Production Prediction (MLP + Reg + Dropout)")
plt.xlabel("Sample Index")
plt.ylabel("Production (kg/m²)")
plt.legend()
plt.grid()
plt.show()

# 14. 학습 곡선 시각화
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE History with EarlyStopping")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid()
plt.show()

```

---

>>>>>>> 6759c28b4288f0c35ae1f2157cef39a269f336a9
