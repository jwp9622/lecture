# - https://shorturl.at/8UmNg
---
# MLP 실습 주요 함수 설명서

Keras를 활용한 **MLP 실습 코드**에서 사용된 주요 함수 및 클래스에 대한 정리입니다.

## 📘 주요 함수/클래스 설명표

| 함수/클래스 | 입력 파라미터 | 반환값 | 설명 |
|-------------|----------------|---------|------|
| **`pd.read_csv(filepath)`** | `filepath` (str): 읽어올 CSV 파일 경로 | `DataFrame`: CSV로부터 생성된 데이터프레임 | CSV 파일을 읽어 pandas DataFrame으로 반환합니다. |
| **`LabelEncoder().fit_transform(y)`** | `y` (array-like): 문자열 레이블 시퀀스 | `np.ndarray`: 정수형 인코딩 배열 | 범주형 문자열을 정수 인덱스로 변환합니다. |
| **`to_categorical(y)`** | `y` (array-like): 정수형 클래스 인덱스 | `np.ndarray`: 원-핫 인코딩된 배열 | 정수 레이블을 다중 분류용 원-핫 벡터로 변환합니다. |
| **`train_test_split(X, y, test_size, stratify)`** | `X`, `y` (array-like): 특성과 타겟<br>`test_size` (float): 테스트셋 비율<br>`stratify` (array): 계층적 샘플링 기준 | `X_train, X_test, y_train, y_test` (tuple) | 학습용과 테스트용 데이터를 분할합니다. |
| **`StandardScaler().fit_transform(X)`** | `X` (array-like): 수치형 특성 행렬 | `np.ndarray`: 정규화된 특성 행렬 | 평균 0, 표준편차 1로 스케일 조정합니다. |
| **`Sequential([...])`** | 레이어 리스트 (`Dense`, `Dropout` 등) | `Model` 객체 | 순차적으로 구성된 Keras 모델 생성 |
| **`Dense(units, activation, input_shape)`** | `units` (int): 뉴런 개수<br>`activation` (str): 활성화 함수<br>`input_shape` (tuple): 입력 형상 (첫 레이어만) | `Layer` 객체 | 완전 연결 신경망 층 생성 |
| **`Dropout(rate)`** | `rate` (float): 무작위로 제거할 비율 (0~1) | `Layer` 객체 | 과적합 방지를 위해 일부 뉴런 출력을 무작위 제거 |
| **`model.compile(optimizer, loss, metrics)`** | `optimizer` (str/object): 최적화 알고리즘<br>`loss` (str): 손실함수<br>`metrics` (list): 평가 지표 | None | 모델 학습 전에 설정을 적용합니다. |
| **`model.fit(X, y, validation_split, epochs, batch_size, callbacks)`** | `X, y`: 학습 데이터<br>`validation_split`: 검증 데이터 비율<br>`epochs`: 반복 횟수<br>`batch_size`: 배치 크기<br>`callbacks`: 콜백 목록 | `History` 객체 | 모델 학습을 수행하고 학습 기록 반환 |
| **`EarlyStopping(monitor, patience, restore_best_weights)`** | `monitor` (str): 감시할 지표<br>`patience` (int): 개선되지 않아도 허용할 에폭 수<br>`restore_best_weights` (bool): 가장 좋은 모델 복원 여부 | 콜백 객체 | 검증 성능이 개선되지 않으면 학습 조기 종료 |
| **`ModelCheckpoint(filepath, save_best_only, monitor)`** | `filepath` (str): 저장 경로<br>`save_best_only` (bool): 최적 모델만 저장<br>`monitor` (str): 감시할 지표 | 콜백 객체 | 가장 성능이 좋은 모델을 자동 저장 |
| **`model.predict(X_test)`** | `X_test` (array-like): 테스트 입력 | `np.ndarray`: 예측 확률 배열 | 입력 샘플에 대한 각 클래스의 예측 확률 반환 |
| **`classification_report(y_true, y_pred, target_names)`** | `y_true`, `y_pred`: 실제/예측 라벨<br>`target_names`: 클래스 이름 리스트 | `str` or `dict` | 분류 정확도, 정밀도, 재현율, F1-score 리포트 |
| **`confusion_matrix(y_true, y_pred)`** | `y_true`, `y_pred`: 실제/예측 라벨 | `np.ndarray`: 혼동 행렬 (2D) | 클래스별 예측 정확도를 보여주는 행렬 생성 |
| **`sns.heatmap(data, annot, cmap, xticklabels, yticklabels)`** | `data`: 행렬 데이터<br>`annot`: 숫자 출력 여부<br>`cmap`: 색상 스타일<br>`xticklabels`, `yticklabels`: 라벨 | 시각화 출력 | 혼동 행렬을 히트맵으로 시각화 |
| **`plt.plot(data)` / `plt.show()`** | `data`: 시각화할 데이터 | 시각화 출력 | 선 그래프나 학습 곡선 등을 시각화 |
