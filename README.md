# KR-JP Culture Mining Dashboard

`culture_mining/kr_jp_final_weighted_practical.csv`를 읽어서 한일 텍스트를 시각화하는 Streamlit 앱입니다.

## 포함 기능

- 국가/교육단계/키워드 필터
- 한일 형태소 기반 워드클라우드 + 상위 키워드 빈도 차트
- 형태소 분해 샘플 테이블과 명사/의미어 품사 필터
- 다국어 BERT 감성분석
- 문장 임베딩 기반 t-SNE 시각화
- 국가별 특징 키워드 비교
- 교육단계별 키워드 히트맵/트리맵
- LDA 토픽 모델링
- 형태소 공기어 네트워크와 bigram/trigram 분석
- UMAP 문장 시각화
- 문서 길이 분포 분석
- 감성 히트맵과 키워드-감성 버블 차트
- 품사 비율, Sankey, Sunburst, 바이올린 플롯
- KMeans 클러스터 시각화와 대표 문장 추출
- TF-IDF 차이 워터폴, 단어 상관행렬, 네트워크 커뮤니티
- 토픽 거리 맵
- `sentence-transformers`가 없을 때 TF-IDF + SVD 폴백

## 실행

```bash
cd /Users/js/Documents/New\ project/culture_mining
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## 모델 관련

- 감성분석 기본 모델: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- 문장 임베딩 기본 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 최초 실행 시 Hugging Face 모델 다운로드가 필요할 수 있습니다.
- `umap-learn`이 없으면 UMAP 탭은 PCA 2D 투영으로 대체됩니다.
- `CULTURE_MINING_LIGHTWEIGHT=1`이면 무료 배포 환경용 경량 모드가 켜지고, 감성분석은 휴리스틱 우선, 임베딩은 `TF-IDF + SVD/PCA` 폴백을 우선 사용합니다.

## 메모

- 현재 CSV는 100,000건이라 t-SNE는 샘플링해서 계산합니다.
- `konlpy`, `fugashi`가 설치되면 형태소 단위로 분석하고, 없으면 정규식 기반으로 폴백합니다.
