# jiu1031

import numpy as np # 선형 대수
import pandas as pd # 데이터 처리, CSV 파일 I/O (예: pd.read_csv)

# 입력 데이터 파일은 읽기 전용 "../input/" 디렉토리에서 사용할 수 있습니다.
# 예를 들어, 이 코드를 실행하면 (run을 클릭하거나 Shift+Enter를 누르면) input 디렉토리 아래의 모든 파일을 나열합니다.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# "Save & Run All"을 사용하여 버전을 생성할 때 출력으로 보존되는 현재 디렉토리 (/kaggle/working/)에 최대 20GB를 쓸 수 있습니다.
# 또한 현재 세션 외부에 저장되지 않는 /kaggle/temp/에 임시 파일을 쓸 수 있습니다.

!pip install dlib

import cv2
import dlib
import numpy as np
import pandas as pd

# dlib에서 사전 학습된 얼굴 감지기 로드
detector = dlib.get_frontal_face_detector()

# 얼굴 랜드마크 예측기 로드
# shape_predictor_68_face_landmarks.dat 파일의 올바른 경로로 교체하세요
predictor = dlib.shape_predictor("/kaggle/input/shape-predictor-68-face-landmarksdat/shape_predictor_68_face_landmarks.dat")

# EAR (Eye Aspect Ratio)를 계산하는 함수
def eye_aspect_ratio(eye):
    x = [point.x for point in eye]
    y = [point.y for point in eye]
    A = np.linalg.norm(np.array([x[1] - x[5], y[1] - y[5]]))
    B = np.linalg.norm(np.array([x[2] - x[4], y[2] - y[4]]))
    C = np.linalg.norm(np.array([x[0] - x[3], y[0] - y[3]]))
    ear = (A + B) / (2.0 * C)
    return ear

# PUC (Pupil to Eye Center distance)를 계산하는 함수
def pupil_to_eye_center_distance(eye):
    x = [point.x for point in eye]
    y = [point.y for point in eye]
    d = np.linalg.norm(np.array([x[0] - x[3], y[0] - y[3]]))
    return d

# MAR (Mouth Aspect Ratio)를 계산하는 함수
def mouth_aspect_ratio(mouth):
    x = [point.x for point in mouth]
    y = [point.y for point in mouth]
    A = np.linalg.norm(np.array([x[13] - x[19], y[13] - y[19]]))
    B = np.linalg.norm(np.array([x[14] - x[18], y[14] - y[18]]))
    C = np.linalg.norm(np.array([x[15] - x[17], y[15] - y[17]]))
    mar = (A + B + C) / (3.0 * np.linalg.norm(np.array([x[12] - x[16], y[12] - y[16]])))
    return mar

# MOE (Mouth to Eye ratio)를 계산하는 함수
def mouth_to_eye_ratio(eye, mouth):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(mouth)
    if ear == 0:  # 0으로 나누는 것을 피하기 위해
        return 0
    moe = mar / ear
    return moe

# 프레임에서 특징과 레이블을 추출하는 함수
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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

# UTA 데이터셋 비디오가 포함된 디렉토리
uta_dataset_video_path = "/kaggle/input/uta-reallife-drowsiness-dataset/Fold1_part1/Fold1_part1/01/0.mov"

# 비디오 캡처 열기
cap = cv2.VideoCapture(uta_dataset_video_path)

all_features = []
all_labels = []

# 프레임 스킵 설정
frame_skip = 5  # 매 5번째 프레임 처리
counter = 0

# 비디오에서 프레임 읽기
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    counter += 1
    if counter % frame_skip != 0:
        continue

    # 현재 프레임에서 특징과 레이블 추출
    features, labels = extract_features(frame)
    all_features.extend(features)
    all_labels.extend(labels)

# 비디오 캡처 객체 해제
cap.release()

# 특징과 레이블 리스트를 DataFrame으로 변환
column_names = ["EAR", "PUC", "MAR", "MOE"]
df_features = pd.DataFrame(all_features, columns=column_names)
df_labels = pd.DataFrame({"drowsy": all_labels})

# 특징과 레이블 DataFrame 병합
df = pd.concat([df_features, df_labels], axis=1)

# 추출된 특징과 레이블을 가진 DataFrame 출력
print(df)

# DataFrame을 CSV 파일로 저장
csv_file_path = "/kaggle/working/Extract features.csv"
df.to_csv(csv_file_path, index=False)

print("DataFrame이 저장되었습니다:", csv_file_path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터의 특징(X)과 목표 변수(y)로 나누기
X = df.drop('drowsy', axis=1)
y = df['drowsy']

# 데이터를 학습용과 테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 나이브 베이즈 알고리즘 적용
naive_bayes_classifier = GaussianNB()

# 학습 세트에서 분류기 학습
naive_bayes_classifier.fit(X_train, y_train)

# 테스트 세트에서 예측 수행
y_pred_nb = naive_bayes_classifier.predict(X_test)

# 나이브 베이즈 모델의 성능 평가
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
classification_rep_nb = classification_report(y_test, y_pred_nb)

# 나이브 베이즈 결과 출력
print("\n나이브 베이즈 정확도:", accuracy_nb)
print("\n나이브 베이즈 혼동 행렬:\n", conf_matrix_nb)
print("\n나이브 베이즈 분류 보고서:\n", classification_rep_nb)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X = df.drop('drowsy', axis=1)
y = df['drowsy']

# chi-squared 테스트를 사용한 SelectKBest로 특징 선택
X_selected = SelectKBest(chi2, k=2).fit_transform(X, y)

# 데이터를 학습용과 테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 제한된 깊이의 결정 트리
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

# 학습 세트에서 분류기 학습
dt_classifier.fit(X_train, y_train)

# 테스트 세트에서 예측 수행
y_pred_dt = dt_classifier.predict(X_test)

# 결정 트리의 성능 평가
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt)

# 결정 트리의 결과 출력
print("결정 트리 정확도:", accuracy_dt)
print("\n결정 트리 혼동 행렬:\n", conf_matrix_dt)
print("\n결정 트리 분류 보고서:\n", classification_rep_dt)

# 결정 트리의 교차 검증
cv_scores_dt = cross_val_score(dt_classifier, X_selected, y, cv=5)
print("\n교차 검증 점수 (결정 트리):", cv_scores_dt)
print("평균 교차 검증 정확도 (결정 트리):", cv_scores_dt.mean())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X = df.drop('drowsy', axis=1)
y = df['drowsy']

# chi-squared 테스트를 사용한 SelectKBest로 특징 선택
X_selected = SelectKBest(chi2, k=2).fit_transform(X, y)

# 데이터를 학습용과 테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 제한된 트리 개수를 가진 랜덤 포레스트
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

# 학습 세트에서 랜덤 포레스트 분류기 학습
rf_classifier.fit(X_train, y_train)

# 테스트 세트에서 예측 수행
y_pred_rf = rf_classifier.predict(X_test)

# 랜덤 포레스트의 성능 평가
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# 랜덤 포레스트의 결과 출력
print("\n랜덤 포레스트 정확도:", accuracy_rf)
print("\n랜덤 포레스트 혼동 행렬:\n", conf_matrix_rf)
print("\n랜덤 포레스트 분류 보고서:\n", classification_rep_rf)

# 랜덤 포레스트의 교차 검증
cv_scores_rf = cross_val_score(rf_classifier, X_selected, y, cv=5)
print("\n교차 검증 점수 (랜덤 포레스트):", cv_scores_rf)
print("평균 교차 검증 정확도 (랜덤 포레스트):", cv_scores_rf.mean())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 가정: df_normalized에 'drowsy' 열이 포함되어 있다고 가정합니다.
X = df.drop('drowsy', axis=1)  # 'drowsy' 열을 제외한 나머지 열은 특징(X)으로 사용합니다.
y = df['drowsy']  # 'drowsy' 열을 목표 변수(y)로 사용합니다.

# 데이터를 학습용과 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화 및 스케일링된 MLP 모델
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 데이터를 정규화 및 스케일링합니다.

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.01, early_stopping=True, random_state=42)

# 스케일링된 학습 세트로 MLP 분류기를 학습시킵니다.
mlp_classifier.fit(X_scaled, y)

# 테스트 세트에서 예측을 수행합니다.
y_pred_mlp = mlp_classifier.predict(X_test)

# MLP 모델의 성능을 평가합니다.
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
classification_rep_mlp = classification_report(y_test, y_pred_mlp)

# MLP 모델의 결과를 출력합니다.
print("\nMLP 정확도:", accuracy_mlp)
print("\nMLP 혼동 행렬:\n", conf_matrix_mlp)
print("\nMLP 분류 보고서:\n", classification_rep_mlp)

# MLP의 교차 검증을 수행합니다.
cv_scores_mlp = cross_val_score(mlp_classifier, X_scaled, y, cv=5)
print("\n교차 검증 점수 (MLP):", cv_scores_mlp)
print("평균 CV 정확도 (MLP):", cv_scores_mlp.mean())

pip install tensorflow

import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# 데이터를 이전과 같이 로드합니다.
# ...

# 데이터를 특징(X)과 레이블(y)로 나눕니다.
X = df.iloc[:, :-1].values
y = df['drowsy'].values

# 데이터를 학습용과 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 입력 데이터를 표준화합니다.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CNN 입력을 위해 데이터를 형태를 변경합니다. (각 샘플이 4개의 특징을 가정합니다.)
X_train = X_train.reshape(X_train.shape[0], 2, 2, 1)
X_test = X_test.reshape(X_test.shape[0], 2, 2, 1)

# CNN 모델을 생성합니다.
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(2, 2, 1)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # 드롭아웃 층을 추가합니다. 드롭아웃 비율은 0.5입니다.
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))  # 이진 분류를 위한 시그모이드 활성화 함수를 가진 출력 층입니다.

# 모델을 컴파일합니다.
optimizer = Adam(learning_rate=0.00001)  # Adam 옵티마이저의 학습률을 지정합니다.
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약을 표시합니다.
model.summary()

# 모델을 학습합니다.
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 테스트 세트에서 모델을 평가합니다.
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# 결과를 출력합니다.
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
