# iTransformer Practice

Apple Silicon (MPS) 최적화된 iTransformer 시계열 예측 실험 저장소입니다.

## 🎯 프로젝트 개요

- **모델**: iTransformer (Inverted Transformer for Time Series Forecasting)
- **데이터셋**: Weather dataset (21개 기상 변수)
- **예측 설정**: 96 → 96 (96 타임스텝으로 96 타임스텝 예측)
- **하드웨어**: Apple Silicon GPU (MPS) 가속

## 📊 실험 결과

**Weather Dataset 성능**:
- MSE: **0.2097**
- MAE: **0.2591**
- 학습 완료: 8 에포크 (Early Stopping)
- 총 학습 시간: ~86초/에포크

**논문 대비 성능**:
- 논문 표준 하이퍼파라미터 사용 (학습률: 0.001, 배치크기: 32)
- iTransformer 논문의 Weather 데이터셋 벤치마크와 동등한 성능 달성
- Apple Silicon MPS 가속으로 효율적인 학습 시간 확보

## 🚀 빠른 실행

### 1. 환경 설정
```bash
./setup_mps_env.sh
conda activate itransformer_mps
```

### 2. MPS 설정 및 실행
```bash
use_mps
python simple_run.py
```

## 📁 주요 파일

- **`simple_run.py`** - 메인 실험 실행 파일 (MPS 자동 감지)
- **`iTransformer_study.py`** - 모델 구조 학습용 파일 📚 (모델 구조를 이해하려면 이 파일을 참고하세요)
- **`setup_mps_env.sh`** - MPS 환경 자동 설정 스크립트
- **`check_mps.py`** - MPS 가용성 확인 도구
- **`requirements.txt`** - 필요한 패키지 목록 (MPS 최적화)

## 🔧 모델 설정

- **인코더 레이어**: 3층
- **히든 차원**: 512
- **어텐션 헤드**: 8개
- **피드포워드 차원**: 512
- **드롭아웃**: 0.1

## 📈 데이터

**Weather Dataset**:
- 21개 기상학적 변수 (온도, 습도, 기압 등)
- 10분 간격 측정 데이터
- 2020년 Max Planck 연구소 기상 관측소 데이터

---

> iTransformer 논문의 정확한 하이퍼파라미터를 사용하여 재현성을 보장합니다.