import numpy as np # 침입자 import pandas as pd # 데이터 처리, CSV 파일 I/O (예: pd.read_csv)

입력 데이터 파일은 "../input/"으로 표시됩니다.
예를 들어, 이 코드를 실행하면 (실행을 클릭하거나 Shift+Enter를 눌러 입력) 아래의 모든 파일을 확인할 수 있습니다.
import os for dirname, _, filenames in os.walk('/kaggle/input'): for filename in filenames: print(os.path.join(dirname, filename))

"모두 저장 및 실행"을 사용하여 버전을 생성할 때 출력으로 인해 제한되는 현재 컨퍼런스(/kaggle/working/)에 최대 20GB를 더 이상 사용할 수 없습니다.
현재 세션 외부에 저장되지 않는 /kaggle/temp/에 임시 파일을 더 이상 사용할 수 없습니다.
!pip dlib 설치

cv2 가져오기 dlib 가져오기 numpy를 np로 가져오기 팬더를 pd로 가져오기

dlib에서 사전 학습된 얼굴 감지기 로드
감지기 = dlib.get_frontal_face_Detector()

얼굴 랜드마크 예측기
shape_predictor_68_face_landmarks.dat 파일의 올바른 해석으로 교체하세요
예측자 = dlib.shape_predictor("/kaggle/input/shape-predictor-68-face-landmarksdat/shape_predictor_68_face_landmarks.dat")

EAR(Eye Aspect Ratio)을 운동하는 함수
def eye_aspect_ratio(eye): x = [눈에 있는 점에 대한 point.x] y = [눈에 있는 점에 대한 point.y] A = np.linalg.norm(np.array([x[1] - x[5] , y[1] - y[5]])) B = np.linalg.norm(np.array([x[2] - x[4], y[2] - y[4]])) C = np.linalg.norm(np.array([x[0] - x[3], y[0] - y[3]])) 귀 = (A + B) / (2.0 * C) 반환 귀

PUC (Pupil to Eye Center distance)를 연습하는 함수
def Pupil_to_eye_center_distance(eye): x = [눈에 있는 점에 대한 point.x] y = [눈에 있는 점에 대한 point.y] d = np.linalg.norm(np.array([x[0] - x[3] , y[0] - y[3]])) d를 반환합니다.

MAR (Mouth Aspect Ratio)를 운동하는 함수
def Mouth_aspect_ratio(mouth): x = [입 안의 점에 대한 point.x] y = [입 안의 점에 대한 point.y] A = np.linalg.norm(np.array([x[13] - x[19] , y[13] - y[19]])) B = np.linalg.norm(np.array([x[14] - x[18], y[14] - y[18]])) C = np.linalg.norm(np.array([x[15] - x[17], y[15] - y[17]])) mar = (A + B + C) / (3.0 * np.linalg. norm(np.array([x[12] - x[16], y[12] - y[16]]))) mar를 반환합니다.

MOE(Mouth to Eye ratio)를 훈련하는 함수
def Mouth_to_eye_ratio(eye, Mouth): Ear = eye_aspect_ratio(eye) mar = Mouth_aspect_ratio(mouth) ifear == 0: # 0으로 할 수 있도록 하기 위해 return 0 moe = mar / 귀 return moe

프레임의 특징과 레이블을 추출하는 함수
def extract_features(프레임): 회색 = cv2.cvtColor(프레임, cv2.COLOR_BGR2GRAY)

# 회색조 프레임에서 얼굴 감지
faces = detector(gray)

features = []
labels = []

for face in faces:
    # 얼굴 랜드마크 예측
    shape = predictor(gray, face)

    # EAR, PUC, MAR, MOE 계산
    ear = eye_aspect_ratio(shape.parts()[36:42])
    puc = pupil_to_eye_center_distance(shape.parts()[36:42])
    mar = mouth_aspect_ratio(shape.parts()[48:68])
    moe = mouth_to_eye_ratio(shape.parts()[36:42], shape.parts()[48:68])

    # 졸음 감지를 위한 조건 정의
    # 예를 들어, EAR이 특정 임계값 이하이고 MAR 및 PUC가 임계값 이하인 경우 졸음으로 간주
    drowsy = 1 if ear < 0.2 or mar > 0.4 or puc < 70 or moe > 0.2 else 0

    # 특징과 레이블을 리스트에 추가
    features.append([ear, puc, mar, moe])
    labels.append(drowsy)

return features, labels
UTA 데이터셋 비디오가 포함된 상태
uta_dataset_video_path = "/kaggle/input/uta-reallife-drowsiness-dataset/Fold1_part1/Fold1_part1/01/0.mov"

영상 캡처 서비스
캡 = cv2.VideoCapture(uta_dataset_video_path)

all_features = [] all_labels = []

프레임 스킵 설정
Frame_skip = 5 # 매 5번째 프레임 처리 counter = 0

비디오에서 프레임 목록
while cap.isOpened(): ret, 프레임 = cap.read()

if not ret:
    break

counter += 1
if counter % frame_skip != 0:
    continue

# 현재 프레임에서 특징과 레이블 추출
features, labels = extract_features(frame)
all_features.extend(features)
all_labels.extend(labels)
영상을 캡처해 보세요
캡.릴리스()

특징과 목록을 DataFrame으로 변환
column_names = ["EAR", "PUC", "MAR", "MOE"] df_features = pd.DataFrame(all_features, columns=column_names) df_labels = pd.DataFrame({"drowsy": all_labels})

특징과 특징 DataFrame 설명
df = pd.concat([df_features, df_labels], 축=1)

추출된 특징과 라벨을 사용하는 DataFrame 출력
인쇄(df)

DataFrame을 CSV 파일로 저장
csv_file_path = "/kaggle/working/추출 기능.csv" df.to_csv(csv_file_path, index=False)

print("DataFrame이 저장되었습니다:", csv_file_path)

sklearn.model_selection에서 팬더를 pd로 가져오기 sklearn.naive_bayes에서 train_test_split 가져오기 sklearn.metrics에서 GaussianNB 가져오기 정확도_점수, 분류_보고서, 혼동_매트릭스 가져오기

데이터의 특징(X)과 목표를 달성하기(y)로 행사하기
X = df.drop('drowsy', axis=1) y = df['drowsy']

데이터를 학습하고 실험적으로 실험하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

나이브 베이즈 적용
naive_bayes_classifier = GaussianNB()

학습 세트에서 정밀기 학습
naive_bayes_classifier.fit(X_train, y_train)

테스트 설정에서 예측을 수행합니다.
y_pred_nb = naive_bayes_classifier.predict(X_test)

나이브 베이즈 모델의 성능 평가
정확도_nb = 정확도_점수(y_test, y_pred_nb) conf_matrix_nb = 혼동 매트릭스(y_test, y_pred_nb) classification_rep_nb = classification_report(y_test, y_pred_nb)

나이브 베이즈 결과 출력
print("\n나이브 베즈 꽉:", Accuracy_nb) print("\n나이브 뷰즈 충분히 감당:\n", conf_matrix_nb) print("\n나이브 베즈 정도범위:\n", classification_rep_nb)

sklearn.model_selection에서 pd로 팬더 가져오기 sklearn.tree에서 train_test_split 가져오기 sklearn.feature_selection에서 DecisionTreeClassifier 가져오기 sklearn.metrics에서 SelectKBest, chi2 가져오기 sklearn.model_selection에서 Accuracy_score, classification_report, 혼란 매트릭스 가져오기 sklearn.preprocessing에서 cross_val_score 가져오기 sklearn.preprocessing에서 가져오기 StandardScaler

X = df.drop('drowsy', axis=1) y = df['drowsy']

카이제곱 테스트를 사용하여 SelectKBest의 특징 선택
X_selected = SelectKBest(chi2, k=2).fit_transform(X, y)

데이터를 학습하고 실험적으로 실험하기
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

관점의 의사결정
dt_classifier = DecisionTreeClassifier(max_length=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

학습 세트에서 정밀기 학습
dt_classifier.fit(X_train, y_train)

테스트 설정에서 예측을 수행합니다.
y_pred_dt = dt_classifier.predict(X_test)

결정 트리의 성능 평가
정확도_dt = 정확도_점수(y_test, y_pred_dt) conf_matrix_dt = 혼동_행렬(y_test, y_pred_dt) classification_rep_dt = classification_report(y_test, y_pred_dt)

결정나무의 결과 출력
print("\n결정 트리 받음:", Accuracy_dt) print("\n결정 트리 할당량:\n", conf_matrix_dt) print("\n결정 트리 지정 범위:\n", classification_rep_dt)

결정 트리의 관계
cv_scores_dt = cross_val_score(dt_classifier, X_selected, y, cv=5) print("\n교차 득점 점수 (결정트리):", cv_scores_dt) print(" 평균 범위 대전(결정트리):", cv_scores_dt.mean() )

sklearn.model_selection에서 pd로 팬더 가져오기 sklearn.ensemble에서 train_test_split 가져오기 sklearn.feature_selection에서 RandomForestClassifier 가져오기 sklearn.metrics에서 SelectKBest, chi2 가져오기 sklearn.model_selection에서 Accuracy_score, classification_report, 혼란 매트릭스 가져오기 sklearn.preprocessing에서 cross_val_score 가져오기 sklearn.preprocessing에서 가져오기 StandardScaler

X = df.drop('drowsy', axis=1) y = df['drowsy']

카이제곱 테스트를 사용하여 SelectKBest의 특징 선택
X_selected = SelectKBest(chi2, k=2).fit_transform(X, y)

데이터를 학습하고 실험적으로 실험하기
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

리뷰 트리를 섞어서 포레스트
rf_classifier = RandomForestClassifier(n_estimators=100, max_length=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

학습 세트에서 무작위 포레스트 알아보기 학습
rf_classifier.fit(X_train, y_train)

테스트 설정에서 예측을 수행합니다.
y_pred_rf = rf_classifier.predict(X_test)

스피커 포레스트의 성능 평가
정확도_rf = 정확도_점수(y_test, y_pred_rf) conf_matrix_rf = 혼동_행렬(y_test, y_pred_rf) classification_rep_rf = classification_report(y_test, y_pred_rf)

무작위 포레스트의 결과
print("\n랜덤 포레스트 멋지다:", Accuracy_rf) print("\n랜덤 포레스트 꽉 차지:\n", conf_matrix_rf) print("\n랜덤 포레스트에 대하여:\n", classification_rep_rf)

무작위 포레스트의 연관성
cv_scores_rf = cross_val_score(rf_classifier, X_selected, y, cv=5) print("\n교차 자격 점수 (랜덤 포레스트):", cv_scores_rf) print(" 평균 분열시켜 (랜덤 포레스트):", cv_scores_rf.mean() )

sklearn.model_selection에서 pd로 팬더 가져오기 sklearn.neural_network에서 train_test_split 가져오기 sklearn.metrics에서 MLPClassifier 가져오기 sklearn.model_selection에서 Accuracy_score, classification_report, 혼란 매트릭스 가져오기 sklearn.preprocessing에서 cross_val_score 가져오기 sklearn.preprocessing에서 가져오기 StandardScaler

가정: df_normalized에 'drowsy' 열이 포함되어 있습니다.
X = df.drop('drowsy', axis=1) # 'drowsy' 열을 나머지 열에는 특징(X)으로 사용합니다. y = df['drowsy'] # 'drowsy' 열을 목표로(y)로 사용합니다.

데이터를 학습용과 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

표준화 및 규모화한 MLP 모델
scaler = StandardScaler() X_scaled = scaler.fit_transform(X) # 데이터를 표준화하고 크기를 조정합니다.

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.01, early_stopping=True, random_state=42)

축소된 손잡이 세트로 MLP 정밀한 손잡이로 설정됩니다.
mlp_classifier.fit(X_scaled, y)

테스트 설정을 예측합니다.
y_pred_mlp = mlp_classifier.predict(X_test)

MLP 모델의 성능을 높이 평가합니다.
정확도_mlp = 정확도_점수(y_test, y_pred_mlp) conf_matrix_mlp = 혼동_행렬(y_test, y_pred_mlp) classification_rep_mlp = classification_report(y_test, y_pred_mlp)

MLP 모델의 결과를 출력합니다.
print("\nMLP 변환:", Accuracy_mlp) print("\nMLP 덩어리 집계:\n", conf_matrix_mlp) print("\nMLP 규정 조정:\n", classification_rep_mlp)

MLP의 관계 검증을 수행합니다.
cv_scores_mlp = cross_val_score(mlp_classifier, X_scaled, y, cv=5) print("\n교차 적립 점수 (MLP):", cv_scores_mlp) print(" 평균 CV 입니다 (MLP):", cv_scores_mlp.mean())

pip 설치 텐서플로우

import cv2 import dlib import numpy as np import pandas as pd from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics import Accuracy_score, classification_report from tensorflow.keras import Sequential from tensorflow.keras.layers import Conv2D, Flatten, Dense , BatchNormalization, tensorflow.keras.optimizers에서 삭제 Adam 가져오기

데이터를 이전과 같이 로드합니다.
...
데이터를 특징(X)과 라벨(y)로 나눕니다.
X = df.iloc[:, :-1].values ​​y = df['drowsy'].values

데이터를 학습용과 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

데이터를 입력합니다.
scaler = StandardScaler() X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test)

CNN을 입력하기 위해 데이터 형식을 변경합니다. (각 샘플은 4개의 특징을 가정합니다.)
X_train = X_train.reshape(X_train.shape[0], 2, 2, 1) X_test = X_test.reshape(X_test.shape[0], 2, 2, 1)

CNN 모델을 생성합니다.
model = Sequential() model.add(Conv2D(32, kernel_size=(2, 2), 활성화='relu', input_shape=(2, 2, 1))) model.add(BatchNormalization()) model.add( Flatten()) model.add(Dense(128, activate='relu')) model.add(BatchNormalization()) model.add(Dropout(0.5)) # 드롭아웃 층을 추가합니다. 드롭아웃은 0.5입니다. model.add(Dense(64, activate='relu')) model.add(BatchNormalization()) model.add(Dense(1, activate='sigmoid')) # 이진을 위해서는 시그모이드 활성화를 하세요 형식입니다.

모델을 보내드립니다.
optimizer = Adam(learning_rate=0.00001) # Adam 옵티마이저의 학습률을 안내합니다. model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

모델 요약을 표시합니다.
모델.요약()

모델을 학습합니다.
model.fit(X_train, y_train, epochs=100, 배치_크기=32, 유효성 검사_data=(X_test, y_test))

테스트 세트의 모델을 평가합니다.
y_pred_prob = model.predict(X_test) y_pred = np.round(y_pred_prob) 정확도 = Accuracy_score(y_test, y_pred) classification_report_result = classification_report(y_test, y_pred)

결과를 출력합니다.
print("정확도:", 정확도) print("분류 보고서:\n", classification_report_result)
