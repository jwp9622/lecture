# https://shorturl.at/gLubM

---
# 📘 **5부. 딥러닝 기술과 활용**

---
#  **6장. 시계열 데이터와 딥러닝**

---

## **(1) 시계열 데이터와 예측 문제**

### ✅ 시계열 데이터란?

시계열(Time Series) 데이터는 **시간의 흐름에 따라 순차적으로 기록된 값들의 집합**입니다. 예를 들어, 다음과 같은 데이터들이 시계열입니다.

| 시간    | 온도(°C) | 습도(%) | 진동(m/s²) |
| ----- | ------ | ----- | -------- |
| 09:00 | 22.1   | 60.3  | 0.05     |
| 09:10 | 22.3   | 59.8  | 0.07     |
| 09:20 | 22.4   | 59.0  | 0.08     |

시계열 데이터는 **값의 순서가 중요**하며, 예측 모델은 과거 데이터를 바탕으로 미래의 값을 예측하는 것이 목표입니다.

---

### ✅ 시계열 예측의 주요 유형

| 유형                | 설명                     | 예시                        |
| ----------------- | ---------------------- | ------------------------- |
| **단일 입력 → 단일 출력** | 과거 1개 시점 → 미래 1개 시점 예측 | 현재 온도로 1시간 후 온도 예측        |
| **다중 입력 → 단일 출력** | 과거 여러 시점 → 미래 1개 예측    | 최근 24시간 온도로 내일 09시 온도 예측  |
| **다중 입력 → 다중 출력** | 과거 시계열 → 미래 시계열 예측     | 1일치 온도로 다음 1일 예측          |
| **다변량 시계열 예측**    | 여러 변수 사용               | 온도, 습도, CO₂ 모두 사용하여 온도 예측 |

---

### ✅ 전통적 시계열 예측 vs 딥러닝 기반 예측

| 항목      | 전통적 접근 (ARIMA 등) | 딥러닝 접근 (RNN/LSTM 등)    |
| ------- | ---------------- | ---------------------- |
| 특징      | 통계적, 수식 기반       | 데이터 기반, 학습 기반          |
| 특징량 추출  | 필요               | 불필요 (end-to-end)       |
| 비선형성 처리 | 어려움              | 가능                     |
| 다변량 입력  | 제한적              | 매우 용이                  |
| 예측 정확도  | 안정적 (단기)         | 우수 (비선형, 장기)           |
| 학습량     | 적음               | 많음                     |
| 예시      | ARIMA, Prophet   | LSTM, GRU, Transformer |

---

## **(2) AIoT에서의 시계열 데이터**

AIoT(Artificial Intelligence of Things)는 **IoT에서 수집된 시계열 데이터를 분석하여 지능형 예측 및 제어를 가능하게 하는 기술**입니다.

### ✅ 대표적인 AIoT 시계열 데이터 예시

| 센서 유형  | 예측 대상  | 설명                |
| ------ | ------ | ----------------- |
| 온도 센서  | 미래 온도  | 온실 내부 온도 제어       |
| CO₂ 센서 | 기류 이상  | 공기 순환 또는 정체 예측    |
| 진동 센서  | 기계 고장  | 이상 진동 감지 및 경고     |
| 전력 센서  | 에너지 소비 | 소비량 급증 예측 및 자동 절전 |

---

### ✅ 실시간 AIoT 데이터 흐름

```text
[센서 측정] → [시계열 저장] → [슬라이딩 윈도우] → [딥러닝 모델 입력] → [미래값 예측] → [제어]
```

예:

* 10분 간격으로 온도 센서 측정
* 24시간(144포인트)을 하나의 윈도우로 묶음
* 딥러닝 모델이 다음 10분 후 온도를 예측
* 온도가 급격히 올라갈 경우 냉방기 자동 작동

---


## **(3) 시계열 예측을 위한 딥러닝 모델: 순환 신경망(RNN)**

---

### 1) RNN

기존의 완전 연결 신경망(Dense Layer)은 입력의 순서 정보나 시간적 맥락을 반영하지 못합니다. 하지만 **시계열 데이터나 자연어처럼 과거의 상태가 현재에 영향을 미치는 데이터**에서는 시간적인 의존성이 중요합니다.

이를 해결하기 위해 **순환 신경망(Recurrent Neural Network, RNN)** 이 등장했습니다. RNN은 **이전 시점의 출력을 현재 시점의 입력으로 순환적으로 전달**함으로써 **시간의 흐름을 모델링**합니다.

- https://en.wikipedia.org/wiki/Recurrent_neural_network
---

#### ✅  RNN의 기본 구조
- https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Recurrent_neural_network_unfold.svg

- https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:RNN_architecture.png

RNN은 시계열 데이터를 다음과 같이 처리합니다:

* 입력: $x^{(t)}$ (시점 $t$의 입력)
* 은닉 상태: $h^{(t)}$ (이전 정보가 누적된 상태)
* 출력: $y^{(t)}$

```text
x(1) ──►[RNN Cell]──► h(1)
           │
x(2) ──►[RNN Cell]──► h(2)
           │
...     (반복 구조)
```



#### ✅ 시간 전개(Unfolding) 구조

RNN은 **하나의 RNN 셀을 여러 시간축으로 복사**하여 처리합니다. 이를 **시간에 따라 펼친다(unfold)** 고 표현합니다.

```text
x(1) → h(1)
        ↓
x(2) → h(2)
        ↓
x(3) → h(3)
```

각 시점의 은닉 상태는 이전 시점의 상태를 기반으로 계산됩니다:

$$
h^{(t)} = \tanh(W_h h^{(t-1)} + W_x x^{(t)} + b)
$$

* $W_h$: 은닉 상태에서의 가중치
* $W_x$: 입력 가중치
* $\tanh$: 활성화 함수 (기본 RNN의 경우)

---

#### ✅ RNN 연산 흐름

| 시점    | 계산 흐름                                            | 출력                      |
| ----- | ------------------------------------------------ | ----------------------- |
| $t=1$ | $h^{(1)} = \tanh(W_x x^{(1)} + W_h h^{(0)} + b)$ | $y^{(1)} = W_y h^{(1)}$ |
| $t=2$ | $h^{(2)} = \tanh(W_x x^{(2)} + W_h h^{(1)} + b)$ | $y^{(2)} = W_y h^{(2)}$ |
| ...   | ...                                              | ...                     |

---

#### ✅ RNN의 한계: 기울기 소실과 폭발

기존 RNN은 다음과 같은 문제점이 존재합니다:

| 문제                              | 설명                                     |
| ------------------------------- | -------------------------------------- |
| **기울기 소실 (Vanishing Gradient)** | 역전파 중 기울기가 점점 작아져 **장기 의존 관계 학습이 어려움** |
| **기울기 폭발 (Exploding Gradient)** | 기울기가 계속 커져 **학습이 발산하거나 불안정해짐**         |
| **장기 기억력 부족**                   | 정보가 시점이 지남에 따라 희미해져 장기 예측이 어려움         |

이러한 한계는 이후에 등장할 **LSTM과 GRU** 모델의 동기이자 이유가 됩니다.

---

### **(2) LSTM(Long Short-Term Memory)**


기존 RNN은 시계열 데이터를 잘 처리할 수 있지만, **장기 의존성 문제**(long-term dependency)를 학습하는 데 한계가 있습니다.
예를 들어, 긴 시퀀스에서 앞부분의 정보가 뒷부분 출력에 영향을 주어야 할 경우, **기울기 소실** 문제로 인해 학습이 되지 않거나 매우 불안정해집니다.

이를 해결하기 위해 **LSTM (Long Short-Term Memory)** 구조가 등장하였습니다.
LSTM은 **정보를 장기적으로 유지하고 필요할 때만 잊는 메커니즘**을 갖춘 RNN의 확장 구조입니다.

---

#### ✅ LSTM 셀의 구조

- https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Long_Short-Term_Memory.svg

```text
        ┌─────────────┐
 x_t → │ 입력 게이트 i_t ─┐
       └─────────────┘  │
                        ▼
                   ┌────────────┐
 h_{t-1} → ──────► │ 셀 상태 업데이트 (C_t) ├───► h_t (출력)
                   └────────────┘
                        ▲
       ┌─────────────┐  │
       │ 망각 게이트 f_t │
       └─────────────┘
```

LSTM은 내부적으로 다음과 같은 세 가지 게이트(gate)를 사용하여 정보를 조절합니다:

| 게이트 이름           | 역할                 | 수식                                          |
| ---------------- | ------------------ | ------------------------------------------- |
| **입력 게이트** $i_t$ | 현재 입력을 기억할지 결정     | $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$ |
| **망각 게이트** $f_t$ | 과거 기억을 유지할지 지울지 결정 | $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ |
| **출력 게이트** $o_t$ | 셀 상태를 출력할지 결정      | $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$ |

이 세 게이트를 통해 셀 상태(Cell State, $C_t$)를 조절하고, 은닉 상태($h_t$)를 출력합니다.

---

#### ✅ LSTM 셀의 전체 동작 흐름

1. **망각(forgot)**: 이전 셀 상태를 얼마나 유지할지 결정

   $$
   f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
   $$

2. **입력(input)**: 현재 입력을 얼마나 반영할지 결정

   $$
   i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
   $$

   $$
   \tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
   $$

3. **셀 상태 업데이트**:

   $$
   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
   $$

4. **출력(output)**: 다음 은닉 상태 결정

   $$
   o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
   $$

   $$
   h_t = o_t \odot \tanh(C_t)
   $$

> 여기서 $\odot$는 element-wise 곱, $\sigma$는 시그모이드 함수입니다.

---

#### ✅ LSTM 내부 구조



---

#### ✅ LSTM의 장점과 특징

| 항목               | 설명                               |
| ---------------- | -------------------------------- |
| **장기 기억력**       | 셀 상태를 통해 정보 보존 가능                |
| **유연한 정보 흐름 제어** | 게이트를 통해 입력/출력/기억을 선택적 제어         |
| **기울기 소실 문제 해결** | 안정적인 역전파 가능                      |
| **학습 안정성**       | RNN보다 학습 수렴률 높음                  |
| **대표적 사용 분야**    | 음성 인식, 기계 고장 예측, 날씨 예측, 시계열 예측 등 |

---

### **(3) GRU(Gated Recurrent Unit)의 구조와 특징**

\*\*GRU(Gated Recurrent Unit)\*\*는 2014년 Cho et al.이 LSTM의 대안으로 제안한 순환 신경망 구조입니다.
LSTM이 장기 기억을 유지하는 데 효과적이지만, 구조가 복잡하고 파라미터가 많다는 단점이 있었습니다.
GRU는 **LSTM의 성능을 유지하면서도 계산량을 줄이기 위해 게이트 수를 줄이고 구조를 단순화**한 모델입니다.

---

#### ✅ GRU의 셀 구조

GRU는 두 가지 주요 게이트를 사용합니다:

- https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Gated_Recurrent_Unit.svg


```text
          ┌────────────────────────┐
x_t ───►──│        업데이트 게이트 z_t     ├──────┐
          └────────────────────────┘      │
                                          ▼
          ┌────────────────────────┐    가중 평균
h_{t-1}──►│        리셋 게이트 r_t         ├───►  h_t
          └────────────────────────┘      ▲
                                          │
    x_t + (r_t * h_{t-1}) →  tanh →  ─────┘
                 = \tilde{h}_t
```

| 게이트                | 설명                 | 수식                                    |
| ------------------ | ------------------ | ------------------------------------- |
| **리셋 게이트** $r_t$   | 과거 정보를 얼마나 무시할지 결정 | $r_t = \sigma(W_r x_t + U_r h_{t-1})$ |
| **업데이트 게이트** $z_t$ | 이전 상태를 얼마나 유지할지 결정 | $z_t = \sigma(W_z x_t + U_z h_{t-1})$ |

그리고 새로운 은닉 상태 $\tilde{h}_t$ 를 계산하여, 최종 은닉 상태 $h_t$ 를 아래와 같이 구성합니다:

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

#### ✅ GRU의 연산 흐름

| 단계                     | 설명                      |
| ---------------------- | ----------------------- |
| ① 업데이트 게이트 $z_t$       | 현재 상태를 얼마나 새로 업데이트할지 결정 |
| ② 리셋 게이트 $r_t$         | 과거 정보를 얼마나 반영할지 결정      |
| ③ 새로운 상태 $\tilde{h}_t$ | 현재 입력과 리셋된 이전 상태로 계산    |
| ④ 최종 상태 $h_t$          | 이전 상태와 새로운 상태를 게이트로 혼합  |

---

#### ✅ GRU vs LSTM 비교

| 항목     | GRU               | LSTM            |
| ------ | ----------------- | --------------- |
| 게이트 수  | 2개 (업데이트, 리셋)     | 3개 (입력, 망각, 출력) |
| 셀 상태   | 없음 (은닉 상태만 사용)    | 셀 상태와 은닉 상태 분리  |
| 파라미터 수 | 적음                | 많음              |
| 계산 속도  | 빠름                | 느림              |
| 메모리 사용 | 적음                | 많음              |
| 표현력    | 충분 (실무에서도 널리 사용됨) | 매우 강력           |
| 학습 안정성 | 보통                | 매우 안정적          |



---

#### ✅ 모델 선택 기준
다음은 시계열 예측 문제에서 모델을 선택할 때 고려할 수 있는 기준입니다:

| 고려 기준                 | 추천 모델      |
| --------------------- | ---------- |
| 시퀀스 길이가 짧고 계산 자원이 제한됨 | RNN 또는 GRU |
| 장기 의존성 학습이 중요함        | LSTM       |
| 실시간 처리 및 모바일 디바이스 환경  | GRU        |
| 학습 안정성과 고성능이 필수       | LSTM       |
| 학습 시간과 메모리가 민감한 상황    | GRU        |
| 빠른 프로토타이핑, 단순 예측      | RNN        |

---
#### ✅ 실무 적용 예시

| 분야                     | 적용 모델       | 이유                  |
| ---------------------- | ----------- | ------------------- |
| 예: AIoT 센서 데이터 예측      | GRU         | 연산량이 적고 실시간 반응에 적합  |
| 예: 고장 진단 예측 (수백 시간 기준) | LSTM        | 장기 의존 관계 필요         |
| 예: 온도/습도 단기 예측         | RNN 또는 GRU  | 구조 단순하고 충분한 성능      |
| 예: ECG 등 생체신호 시계열 분석   | LSTM 또는 GRU | 특징 간 상관성 보존, 안정성 중요 |
| 예: 스마트워치 기반 실시간 감정 추론  | GRU         | 경량, 빠른 처리 필요        |

---
## 📘 **(4) LSTM 기반 시계열 예측 실습**



### ✅ETTh1 데이터셋

**ETTh1 (Electricity Transformer Temperature - Hourly 1)** 데이터셋은 전력 변압기 시스템의 시계열 동작 데이터를 기록한 데이터셋으로, **다중 변수 시계열 예측(Multivariate Time Series Forecasting)** 문제에 널리 사용됩니다.


#### 데이터셋 요약

| 항목       | 내용                            |
| -------- | ----------------------------- |
| 이름       | ETTh1.csv                     |
| 시간 간격    | 1시간                           |
| 기간       | 약 1년치                         |
| 샘플 수     | 약 8,000개                      |
| 주요 목적    | 시계열 예측 (ex. 향후 T1 예측)         |
| 주요 타깃 변수 | `OT` (유출 온도: Oil Temperature) |

---

#### Features

| Feature 이름 | 설명                                      | 단위 |
| ---------- | --------------------------------------- | -- |
| **date**   | 시간 정보 (YYYY-MM-DD HH)                   | -  |
| **HUFL**   | 고주파 유도 부하 (High Usage Frequency Load)   | kW |
| **HULL**   | 저주파 유도 부하 (Low Usage Frequency Load)    | kW |
| **MUFL**   | 중간 주파수 부하 (Medium Usage Frequency Load) | kW |
| **MULL**   | 중간 주파수 저부하 (Medium Usage Low Load)      | kW |
| **LUFL**   | 저주파 고부하 (Low Usage Frequency Load)      | kW |
| **LULL**   | 저주파 저부하 (Low Usage Low Load)            | kW |
| **OT**     | 변압기 오일 온도 (Oil Temperature) → 예측 대상     | °C |

> 참고: HUFL, MUFL, LUFL 등은 변전소나 송전 시스템에서 측정된 **부하(Load)** 또는 **에너지 사용량**을 시계열적으로 측정한 값입니다.

---
### ✅ 실습 목표

* **ETTh1 데이터셋 (Electricity Transformer Temperature for 1-hour level)**
* 시계열 데이터를 기반으로 LSTM 모델을 활용하여 앞으로 24시간 후의 OT 예측
* 시계열 데이터를 슬라이딩 윈도우로 재구성하여 딥러닝 입력 형식에 맞춤
* 모델 성능을 mse, r2-score로 평가


---

### ✅  실습 코드

```python

# 1. 라이브러리 로드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

 # 2. 데이터 다운로드 및 로드
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
filename = "ETTh1.csv"
urllib.request.urlretrieve(url, filename)

df = pd.read_csv(filename)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 3. 단일 피처 선택 (예: 'OT')
feature = 'OT'
data = df[[feature]]


# 4. 시계열 전체 시각화
plt.figure(figsize=(14, 4))
plt.plot(data.index[:500], data[feature].values[:500])
plt.title(f"ETTh1 Time Series - {feature}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 정규화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 6. 시퀀스 생성
def create_sequences(data, window_size):
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:i + window_size]
        y = data[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

window_size = 24
X, y = create_sequences(data_scaled, window_size)

# 7. 시퀀스 샘플 10개 시각화
fig, axs = plt.subplots(5, 2, figsize=(14, 10))
axs = axs.ravel()
for i in range(10):
    axs[i].plot(X[i].flatten(), label="X")
    axs[i].axhline(y=y[i], color='red', linestyle='--', label="y")
    axs[i].set_title(f"Sequence Sample {i}")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# 8. 학습/테스트 분할
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 9. LSTM 모델 정의 (xLSTM 스타일)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')


# 10. 학습
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 11. 예측
y_pred = model.predict(X_test)

# 12. 결과 복원 및 평가
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:300], label='True')
plt.plot(y_pred_inv[:300], label='Predicted')
plt.title("LSTM Forecast on ETTh1 (OT)")
plt.xlabel("Time Step")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 평가 지표
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

```
---

## 🧾 코드 주요 함수 설명

| 함수명                                   | 역할                    | 비고                      |
| ------------------------------------- | --------------------- | ----------------------- |
| `create_sequences(X, y, window_size)` | 시계열 데이터를 입력/타깃 쌍으로 변환 | LSTM용 시계열 윈도우 구성        |
| `train_test_split()`                  | 학습/테스트 데이터 분리         | sklearn 사용              |
| `StandardScaler().fit_transform()`    | 정규화                   | 평균 0, 표준편차 1            |
| `Sequential([...])`                   | CNN-LSTM 모델 구성        | Conv1D → MaxPool → LSTM |
| `model.fit()`                         | 모델 학습 수행              | Epoch=10                |

---
