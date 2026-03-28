from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

DATA_PATH = Path(__file__).with_name("kr_jp_final_weighted_practical.csv")
DEFAULT_SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TSNE_SAMPLES = 3000
LIGHTWEIGHT_MODE = os.getenv("CULTURE_MINING_LIGHTWEIGHT", "0").strip() == "1"
EMBEDDING_SAMPLE_LIMIT = 600 if LIGHTWEIGHT_MODE else 1500
TOPIC_SAMPLE_LIMIT = 2000 if LIGHTWEIGHT_MODE else 4000
TOKEN_SAMPLE_LIMIT = 3000 if LIGHTWEIGHT_MODE else 5000
COOCC_SAMPLE_LIMIT = 1500 if LIGHTWEIGHT_MODE else 3000
CLUSTER_SAMPLE_LIMIT = 500 if LIGHTWEIGHT_MODE else 1000
LEVEL_ORDER = [
    "유치원",
    "초등학교",
    "중학교",
    "고등학교",
    "대학교",
    "幼稚園",
    "小学校",
    "中学校",
    "高校",
    "大学",
]
LEVEL_DISPLAY = {
    "유치원": "유치원 / 幼稚園",
    "초등학교": "초등학교 / 小学校",
    "중학교": "중학교 / 中学校",
    "고등학교": "고등학교 / 高校",
    "대학교": "대학교 / 大学",
    "幼稚園": "幼稚園 / 유치원",
    "小学校": "小学校 / 초등학교",
    "中学校": "中学校 / 중학교",
    "高校": "高校 / 고등학교",
    "大学": "大学 / 대학교",
}

KR_STOPWORDS = {
    "그리고",
    "그래서",
    "너무",
    "정말",
    "진짜",
    "오늘",
    "우리",
    "제가",
    "저는",
    "그냥",
    "같다",
    "있다",
    "없다",
    "했다",
    "이다",
    "하다",
    "했다",
}
JP_STOPWORDS = {
    "そして",
    "とても",
    "すごく",
    "本当に",
    "今日",
    "私",
    "わたし",
    "です",
    "でした",
    "する",
    "した",
    "ある",
    "いる",
}
KR_NEGATIVE_CUES = {
    "힘들",
    "불안",
    "스트레스",
    "복잡",
    "걱정",
    "지치",
    "피곤",
    "아쉽",
    "싫",
    "부담",
    "긴장",
    "우울",
    "화나",
    "못했다",
    "안 됐",
}
KR_POSITIVE_CUES = {
    "행복",
    "재미있",
    "즐겁",
    "좋았",
    "기쁘",
    "뿌듯",
    "기대",
    "편안",
    "신났",
    "성장",
    "잘했",
}
JP_NEGATIVE_CUES = {
    "不安",
    "疲",
    "大変",
    "難し",
    "休みたい",
    "複雑",
    "心配",
    "つら",
    "苦し",
    "嫌",
    "悩",
    "ストレス",
}
JP_POSITIVE_CUES = {
    "楽しい",
    "嬉しい",
    "幸せ",
    "よかった",
    "安心",
    "期待",
    "面白い",
    "好き",
    "成長",
    "満足",
}

KR_CONTENT_POS = {"Noun", "Adjective", "Verb"}
JP_CONTENT_POS = {"名詞", "形容詞", "動詞"}
KR_NOUN_POS = {"Noun"}
JP_NOUN_POS = {"名詞"}
POS_LABELS = ["명사", "동사", "형용사", "부사", "감탄사"]
POS_DISPLAY = {
    "명사": "명사 / 名詞",
    "동사": "동사 / 動詞",
    "형용사": "형용사 / 形容詞",
    "부사": "부사 / 副詞",
    "감탄사": "감탄사 / 感動詞",
}
KR_POS_MAP = {
    "명사": {"Noun"},
    "동사": {"Verb"},
    "형용사": {"Adjective"},
    "부사": {"Adverb"},
    "감탄사": {"Exclamation"},
}
JP_POS_MAP = {
    "명사": {"名詞"},
    "동사": {"動詞"},
    "형용사": {"形容詞"},
    "부사": {"副詞"},
    "감탄사": {"感動詞"},
}
KR_PARTICLES = (
    "에서",
    "에게",
    "으로",
    "처럼",
    "보다",
    "까지",
    "부터",
    "하고",
    "과",
    "와",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "도",
    "만",
    "에",
    "의",
)
KR_VERB_ENDINGS = (
    "했습니다",
    "하였다",
    "합니다",
    "했다",
    "였다",
    "었다",
    "았다",
    "니다",
    "어요",
    "아요",
    "어서",
    "아서",
    "해서",
    "이다",
    "다",
)


class MorphToken(NamedTuple):
    surface: str
    lemma: str
    pos: str


def main() -> None:
    st.set_page_config(
        page_title="KR-JP Culture Mining Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("한일 텍스트 컬처마이닝 대시보드 / 日韓テキスト文化マイニングダッシュボード")
    st.caption("형태소 기반 워드클라우드, 다국어 BERT 감성분석, t-SNE 시각화를 한 번에 확인합니다. / 形態素ベースのワードクラウド、多言語BERT感情分析、t-SNE可視化をまとめて確認できます。")
    if LIGHTWEIGHT_MODE:
        st.info("경량 모드 활성화 / 軽量モード有効: 무료 배포 환경에 맞춰 감성분석은 휴리스틱, 임베딩은 경량 폴백을 우선 사용합니다. / 無料デプロイ環境向けに感情分析はヒューリスティック、埋め込みは軽量フォールバックを優先します。")

    df = load_data(DATA_PATH)
    filtered = render_sidebar(df)

    total_docs = len(filtered)
    avg_length = filtered["text"].str.len().mean() if total_docs else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("문서 수 / 文書数", f"{total_docs:,}")
    col2.metric("국가 수 / 国数", filtered["country"].nunique())
    col3.metric("평균 글자 수 / 平均文字数", f"{avg_length:.1f}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(
        [
            "데이터 개요 / データ概要",
            "한일 비교 / 日韓比較",
            "워드클라우드 / ワードクラウド",
            "키워드 비교 / キーワード比較",
            "히트맵·트리맵 / ヒートマップ・ツリーマップ",
            "감성분석 / 感情分析",
            "토픽 모델링 / トピックモデリング",
            "공기어·N그램 / 共起語・Nグラム",
            "t-SNE",
            "UMAP",
            "길이 분포 / 長さ分布",
            "고급 시각화 / 高度可視化",
        ]
    )

    with tab1:
        render_overview(filtered)

    with tab2:
        render_side_by_side_comparison(filtered)

    with tab3:
        render_wordcloud(filtered)

    with tab4:
        render_keyword_comparison(filtered)

    with tab5:
        render_heatmap_treemap(filtered)

    with tab6:
        render_sentiment(filtered)

    with tab7:
        render_topic_modeling(filtered)

    with tab8:
        render_cooccurrence_and_ngrams(filtered)

    with tab9:
        render_tsne(filtered)

    with tab10:
        render_umap(filtered)

    with tab11:
        render_length_distribution(filtered)

    with tab12:
        render_advanced_visualizations(filtered)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"id", "country", "level", "text"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"필수 컬럼이 없습니다 / 必須カラムがありません: {missing}")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["country"] = df["country"].astype(str)
    df["level"] = df["level"].astype(str)
    if "language" not in df.columns:
        df["language"] = df["country"].map({"KR": "ko", "JP": "ja"}).fillna("")
    df["language"] = df["language"].fillna("").astype(str).str.strip().str.lower()
    return df


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("분석 조건 / 分析条件")
    countries = st.sidebar.multiselect(
        "국가 / 国",
        options=sorted(df["country"].unique()),
        default=sorted(df["country"].unique()),
    )
    levels = st.sidebar.multiselect(
        "교육 단계 / 教育段階",
        options=sort_levels(df["level"].unique().tolist()),
        default=sort_levels(df["level"].unique().tolist()),
        format_func=lambda level: LEVEL_DISPLAY.get(level, level),
    )
    languages = st.sidebar.multiselect(
        "언어 / 言語",
        options=sorted(value for value in df["language"].unique().tolist() if value),
        default=sorted(value for value in df["language"].unique().tolist() if value),
    )
    st.sidebar.subheader("텍스트 키워드 / テキストキーワード")
    keyword_kr = st.sidebar.text_input("한국어 키워드 / 韓国語キーワード", "")
    keyword_jp = st.sidebar.text_input("일본어 키워드 / 日本語キーワード", "")
    selected_pos_labels = st.sidebar.multiselect(
        "포함할 품사 / 含む品詞",
        options=POS_LABELS,
        default=POS_LABELS,
        format_func=lambda label: POS_DISPLAY.get(label, label),
    )
    if not selected_pos_labels:
        selected_pos_labels = POS_LABELS
    sample_size = st.sidebar.slider(
        "t-SNE 최대 샘플 수 / t-SNE最大サンプル数",
        min_value=300,
        max_value=MAX_TSNE_SAMPLES,
        value=EMBEDDING_SAMPLE_LIMIT,
        step=100,
    )
    st.session_state["selected_pos_labels"] = selected_pos_labels
    st.session_state["tsne_sample_size"] = sample_size

    filtered = df[
        df["country"].isin(countries)
        & df["level"].isin(levels)
        & df["language"].isin(languages if languages else df["language"].unique())
    ].copy()
    if keyword_kr.strip():
        filtered = filtered[
            ~filtered["country"].eq("KR")
            | filtered["text"].str.contains(keyword_kr.strip(), case=False, na=False)
        ]
    if keyword_jp.strip():
        filtered = filtered[
            ~filtered["country"].eq("JP")
            | filtered["text"].str.contains(keyword_jp.strip(), case=False, na=False)
        ]

    st.sidebar.caption(f"선택된 문서 수 / 選択文書数: {len(filtered):,}")
    return filtered


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("분포 / 分布")
    st.caption(build_analyzer_status())
    level_order = sort_levels(df["level"].unique().tolist())
    col1, col2 = st.columns(2)

    with col1:
        country_counts = (
            df.groupby(["country", "level"]).size().reset_index(name="count")
        )
        fig = px.bar(
            country_counts,
            x="level",
            y="count",
            color="country",
            barmode="group",
            title="국가·교육단계별 문서 수 / 国・教育段階別文書数",
            category_orders={"level": level_order},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        length_df = df.assign(text_length=df["text"].str.len())
        fig = px.box(
            length_df,
            x="country",
            y="text_length",
            color="country",
            title="국가별 텍스트 길이 분포 / 国別テキスト長分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("샘플 문장 / サンプル文")
    sample_text_rows = build_balanced_sample_rows(
        df[["country", "language", "level", "text"]], per_country=10
    )
    st.dataframe(sample_text_rows, use_container_width=True)

    st.subheader("형태소 분해 샘플 / 形態素分解サンプル")
    sample_rows = build_balanced_sample_rows(
        df[["country", "language", "level", "text"]], per_country=5
    ).copy()
    selected_pos_labels = st.session_state.get("selected_pos_labels", POS_LABELS)
    sample_rows["morphemes"] = [
        format_morpheme_tokens(
            extract_analysis_tokens(row.text, row.country, selected_pos_labels, row.language)
        )
        for row in sample_rows.itertuples(index=False)
    ]
    st.dataframe(sample_rows, use_container_width=True)


def render_wordcloud(df: pd.DataFrame) -> None:
    st.subheader("워드클라우드 / ワードクラウド")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    top_n = st.slider("상위 키워드 수 / 上位キーワード数", min_value=20, max_value=500, value=120, step=20)
    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        freq = build_term_frequency(subset)
        with country_columns[country]:
            st.markdown(f"### {country}")
            if not freq:
                st.warning("토큰화 결과가 비어 있습니다. / トークン化結果が空です。")
                continue

            fig = build_wordcloud_figure(subset, top_n=top_n)
            if fig is not None:
                st.pyplot(fig)

            top_terms = pd.DataFrame(freq.most_common(top_n), columns=["term", "count"])
            display_n = min(top_n, len(top_terms))
            bar_fig = px.bar(
                top_terms.head(display_n),
                x="term",
                y="count",
                title=f"{country} 상위 키워드 빈도 / 上位キーワード頻度",
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.dataframe(top_terms.head(display_n), use_container_width=True)


def render_side_by_side_comparison(df: pd.DataFrame) -> None:
    st.subheader("한국어·일본어 결과 비교 / 韓国語・日本語結果比較")
    countries = set(df["country"].unique())
    if not {"KR", "JP"}.issubset(countries):
        st.warning("이 비교 화면은 KR과 JP가 모두 선택되어 있어야 합니다. / この比較画面はKRとJPの両方が選択されている必要があります。")
        return

    top_n = st.slider("비교 상위 키워드 수 / 比較上位キーワード数", 10, 200, 40, step=10, key="side_by_side_top_n")
    sentiment_sample = st.slider(
        "비교 감성분석 샘플 수 / 比較感情分析サンプル数", 100, 3000, 600, step=100, key="side_by_side_sentiment_sample"
    )

    left_df = df[df["country"] == "KR"].copy()
    right_df = df[df["country"] == "JP"].copy()
    left_col, right_col = st.columns(2)

    for col, country_label, subset in (
        (left_col, "KR", left_df),
        (right_col, "JP", right_df),
    ):
        with col:
            render_country_comparison_panel(
                subset,
                country_label=country_label,
                top_n=top_n,
                sentiment_sample=sentiment_sample,
            )


def render_country_comparison_panel(
    df: pd.DataFrame, country_label: str, top_n: int, sentiment_sample: int
) -> None:
    st.markdown(f"### {country_label}")
    if df.empty:
        st.info("데이터가 없습니다. / データがありません。")
        return

    token_df = build_token_dataframe(df)
    if token_df.empty or "term" not in token_df.columns:
        st.info("형태소 결과가 없습니다. / 形態素結果がありません。")
        return
    top_terms = token_df["term"].value_counts().head(top_n).rename_axis("term").reset_index(name="count")
    token_lengths = [
        len(
            extract_analysis_tokens(
                text,
                country_label,
                st.session_state.get("selected_pos_labels", POS_LABELS),
                language,
            )
        )
        for text, language in zip(df["text"].head(400), df["language"].head(400))
    ]
    col1, col2, col3 = st.columns(3)
    col1.metric("문서 수 / 文書数", f"{len(df):,}")
    col2.metric("평균 글자 수 / 平均文字数", f"{df['text'].str.len().mean():.1f}")
    col3.metric("평균 형태소 수 / 平均形態素数", f"{np.mean(token_lengths) if token_lengths else 0:.1f}")

    wc_fig = build_wordcloud_figure(df, top_n=top_n)
    if wc_fig is not None:
        st.pyplot(wc_fig)

    if not top_terms.empty:
        display_n = min(top_n, len(top_terms))
        fig = px.bar(
            top_terms.head(display_n),
            x="term",
            y="count",
            title=f"{country_label} 상위 키워드 / 上位キーワード",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_terms.head(display_n), use_container_width=True)

    sample_rows = df[["language", "level", "text"]].head(5).copy()
    sample_rows["morphemes"] = [
        format_morpheme_tokens(
            extract_analysis_tokens(
                text,
                country_label,
                st.session_state.get("selected_pos_labels", POS_LABELS),
                language,
            )
        )
        for text, language in zip(sample_rows["text"], sample_rows["language"])
    ]
    st.dataframe(sample_rows, use_container_width=True)

    sentiment_df, status = predict_sentiment(
        df.sample(min(sentiment_sample, len(df)), random_state=42).copy(),
        st.session_state.get("sentiment_model_name", DEFAULT_SENTIMENT_MODEL),
    )
    if sentiment_df is not None:
        counts = sentiment_df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.pie(counts, names="sentiment", values="count", title=f"{country_label} 감성 비율 / 感情比率")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(status)


def render_keyword_comparison(df: pd.DataFrame) -> None:
    st.subheader("국가별 키워드 비교 / 国別キーワード比較")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    top_n = st.slider("비교 키워드 수 / 比較キーワード数", 10, 40, 20, key="comparison_top_n")
    keyword_df = build_keyword_comparison(df, top_n)
    if keyword_df.empty:
        st.warning("비교할 키워드를 만들지 못했습니다. / 比較キーワードを作成できませんでした。")
        return
    countries = [country for country in ["KR", "JP"] if country in keyword_df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = keyword_df[keyword_df["country"] == country].copy()
        with country_columns[country]:
            st.markdown(f"### {country}")
            fig = px.bar(
                subset,
                x="score",
                y="term",
                orientation="h",
                title=f"{country} 특징 키워드 / 特徴キーワード",
                hover_data=["count", "share"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(subset, use_container_width=True)


def render_heatmap_treemap(df: pd.DataFrame) -> None:
    st.subheader("교육단계별 키워드 히트맵 / 教育段階別キーワードヒートマップ")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    top_n = st.slider("히트맵 키워드 수 / ヒートマップキーワード数", 10, 30, 15, key="heatmap_top_n")
    heatmap_df = build_level_term_heatmap(df, top_n)
    if heatmap_df.empty:
        st.warning("히트맵 데이터를 만들지 못했습니다. / ヒートマップデータを作成できませんでした。")
        return

    fig = px.imshow(
        heatmap_df.set_index("term"),
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="교육단계별 상위 키워드 빈도 / 教育段階別上位キーワード頻度",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("키워드 트리맵 / キーワードツリーマップ")
    treemap_df = build_treemap_data(df, top_n=50)
    fig = px.treemap(
        treemap_df,
        path=["country", "level", "term"],
        values="count",
        color="count",
        color_continuous_scale="Tealgrn",
        title="국가-교육단계-키워드 트리맵 / 国-教育段階-キーワードツリーマップ",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment(df: pd.DataFrame) -> None:
    st.subheader("다국어 BERT 감성분석 / 多言語BERT感情分析")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    model_name = st.text_input("감성분석 모델 / 感情分析モデル", value=DEFAULT_SENTIMENT_MODEL)
    st.session_state["sentiment_model_name"] = model_name
    sample_n = st.slider("감성분석 샘플 수 / 感情分析サンプル数", 100, 5000, 1000, step=100)
    sampled = df.sample(min(sample_n, len(df)), random_state=42).copy()

    with st.spinner("감성 라벨 예측 중... / 感情ラベル予測中..."):
        sentiment_df, status = predict_sentiment(sampled, model_name)

    st.caption(status)
    if sentiment_df is None:
        st.warning(
            "현재 환경에서 transformers/torch가 없어 BERT 감성분석을 실행할 수 없습니다. "
            "아래 requirements를 설치한 뒤 다시 실행하세요. / 現在の環境ではtransformers/torchがなく、BERT感情分析を実行できません。requirementsをインストールして再実行してください。"
        )
        return

    counts = (
        sentiment_df.groupby(["country", "sentiment"]).size().reset_index(name="count")
    )
    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="country",
        barmode="group",
        title="국가별 감성 분포 / 国別感情分布",
    )
    st.plotly_chart(fig, use_container_width=True)

    score_fig = px.box(
        sentiment_df,
        x="country",
        y="score",
        color="sentiment",
        title="감성 점수 분포 / 感情スコア分布",
    )
    st.plotly_chart(score_fig, use_container_width=True)
    st.dataframe(
        sentiment_df[["country", "level", "text", "sentiment", "score"]].head(30),
        use_container_width=True,
    )


def render_topic_modeling(df: pd.DataFrame) -> None:
    st.subheader("토픽 모델링 / トピックモデリング")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    topic_count = st.slider("토픽 수 / トピック数", 2, 8, 4, key="topic_count")
    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        with country_columns[country]:
            st.markdown(f"### {country}")
            topic_terms, topic_docs, status = build_topic_model(subset, topic_count)
            st.caption(status)
            if topic_terms.empty or topic_docs.empty:
                st.warning("토픽 모델을 만들지 못했습니다. / トピックモデルを作成できませんでした。")
                continue

            fig = px.bar(
                topic_terms,
                x="weight",
                y="term",
                color="topic",
                facet_col="topic",
                facet_col_wrap=2,
                orientation="h",
                title=f"{country} 토픽별 상위 키워드 / トピック別上位キーワード",
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.replace("topic=", "")))
            st.plotly_chart(fig, use_container_width=True)

            topic_share = topic_docs.groupby(["topic"]).size().reset_index(name="count")
            fig = px.bar(
                topic_share,
                x="topic",
                y="count",
                title=f"{country} 대표 토픽 분포 / 代表トピック分布",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(topic_terms, use_container_width=True)


def render_cooccurrence_and_ngrams(df: pd.DataFrame) -> None:
    st.subheader("형태소 공기어 네트워크 / 形態素共起語ネットワーク")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    col1, col2 = st.columns(2)
    with col1:
        min_edge = st.slider("최소 공기어 빈도 / 最小共起頻度", 2, 20, 5, key="coocc_min_edge")
    with col2:
        top_nodes = st.slider("최대 노드 수 / 最大ノード数", 10, 40, 20, key="coocc_top_nodes")

    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        with country_columns[country]:
            st.markdown(f"### {country}")
            cooccurrence_df = build_cooccurrence_edges(
                subset, min_edge=min_edge, top_nodes=top_nodes
            )
            if cooccurrence_df.empty:
                st.warning("공기어 네트워크를 만들기 위한 연결이 부족합니다. / 共起語ネットワークを作るための接続が不足しています。")
            else:
                fig = build_cooccurrence_figure(cooccurrence_df)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cooccurrence_df, use_container_width=True)

    st.subheader("N-gram 분석 / N-gram分析")
    ngram_size = st.radio(
        "N-gram 크기 / N-gramサイズ",
        options=[2, 3],
        horizontal=True,
        key="ngram_size",
        format_func=lambda value: f"{value}-gram",
    )
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        with country_columns[country]:
            st.markdown(f"### {country}")
            ngram_df = build_ngram_counts(subset, n=ngram_size, top_n=25)
            if ngram_df.empty:
                st.warning("N-gram 결과가 없습니다. / N-gram結果がありません。")
                continue

            fig = px.bar(
                ngram_df,
                x="count",
                y="ngram",
                orientation="h",
                title=f"{country} 상위 {ngram_size}-gram / 上位{ngram_size}-gram",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ngram_df, use_container_width=True)


def render_tsne(df: pd.DataFrame) -> None:
    st.subheader("문장 임베딩 t-SNE / 文埋め込み t-SNE")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    sample_size = min(st.session_state.get("tsne_sample_size", 1500), len(df))
    if sample_size < 6:
        st.warning("t-SNE는 최소 6개 이상의 문장이 필요합니다. / t-SNEには少なくとも6文以上必要です。")
        return
    model_name = st.text_input("임베딩 모델 / 埋め込みモデル", value=DEFAULT_EMBEDDING_MODEL)
    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        subset_sample_size = min(sample_size, len(subset))
        if subset_sample_size < 6:
            with country_columns[country]:
                st.markdown(f"### {country}")
                st.warning("t-SNE는 최소 6개 이상의 문장이 필요합니다. / t-SNEには少なくとも6文以上必要です。")
            continue
        sampled = subset.sample(subset_sample_size, random_state=42).copy()

        with country_columns[country]:
            st.markdown(f"### {country}")
            with st.spinner(f"{country} 문장 임베딩 및 t-SNE 계산 중... / {country} 文埋め込みとt-SNE計算中..."):
                plot_df, status = build_tsne_projection(sampled, model_name)

            st.caption(status)
            if plot_df is None:
                st.error("t-SNE 좌표를 만들지 못했습니다. / t-SNE座標を作成できませんでした。")
                continue

            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                symbol="level",
                color="level",
                hover_data=["text"],
                title=f"{country} 문장 의미 공간 t-SNE / 文意味空間 t-SNE",
                opacity=0.8,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_umap(df: pd.DataFrame) -> None:
    st.subheader("문장 임베딩 UMAP / 文埋め込み UMAP")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    sample_size = min(st.session_state.get("tsne_sample_size", 1500), len(df))
    if sample_size < 6:
        st.warning("UMAP은 최소 6개 이상의 문장이 필요합니다. / UMAPには少なくとも6文以上必要です。")
        return
    model_name = st.text_input("UMAP 임베딩 모델 / UMAP埋め込みモデル", value=DEFAULT_EMBEDDING_MODEL)
    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        subset_sample_size = min(sample_size, len(subset))
        if subset_sample_size < 6:
            with country_columns[country]:
                st.markdown(f"### {country}")
                st.warning("UMAP은 최소 6개 이상의 문장이 필요합니다. / UMAPには少なくとも6文以上必要です。")
            continue
        sampled = subset.sample(subset_sample_size, random_state=42).copy()

        with country_columns[country]:
            st.markdown(f"### {country}")
            with st.spinner(f"{country} 문장 임베딩 및 UMAP 계산 중... / {country} 文埋め込みとUMAP計算中..."):
                plot_df, status = build_umap_projection(sampled, model_name)

            st.caption(status)
            if plot_df is None:
                st.error("UMAP 좌표를 만들지 못했습니다. / UMAP座標を作成できませんでした。")
                continue

            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                symbol="level",
                color="level",
                hover_data=["text"],
                title=f"{country} 문장 의미 공간 UMAP / 文意味空間 UMAP",
                opacity=0.8,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_length_distribution(df: pd.DataFrame) -> None:
    st.subheader("문서 길이 분포 / 文書長分布")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    level_order = sort_levels(df["level"].unique().tolist())
    length_df = df.copy()
    length_df["char_length"] = length_df["text"].str.len()
    length_df["token_length"] = [
        len(
            extract_analysis_tokens(
                text,
                country,
                st.session_state.get("selected_pos_labels", POS_LABELS),
                language,
            )
        )
        for text, country, language in zip(length_df["text"], length_df["country"], length_df["language"])
    ]

    fig = px.histogram(
        length_df,
        x="char_length",
        color="country",
        nbins=40,
        barmode="overlay",
        title="국가별 문자 길이 분포 / 国別文字長分布",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        length_df,
        x="level",
        y="token_length",
        color="country",
        title="교육단계별 형태소 수 분포 / 教育段階別形態素数分布",
        category_orders={"level": level_order},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_advanced_visualizations(df: pd.DataFrame) -> None:
    st.subheader("고급 시각화 / 高度可視化")
    if df.empty:
        st.warning("필터 조건에 맞는 데이터가 없습니다. / フィルタ条件に合うデータがありません。")
        return

    sentiment_model = st.session_state.get("sentiment_model_name", DEFAULT_SENTIMENT_MODEL)
    selected_pos = st.session_state.get("selected_pos_labels", POS_LABELS)

    st.markdown("### 감성 히트맵 / 感情ヒートマップ")
    sentiment_sample = st.slider("고급 감성분석 샘플 수 / 高度感情分析サンプル数", 100, 3000, 800, step=100, key="advanced_sentiment_sample")
    sentiment_df, status = predict_sentiment(
        sample_documents(df, sentiment_sample),
        sentiment_model,
    )
    st.caption(status)
    if sentiment_df is not None:
        heatmap_df = build_sentiment_heatmap_data(sentiment_df)
        fig = px.imshow(
            heatmap_df.set_index("bucket"),
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="국가·교육단계별 감성 비율 / 国・教育段階別感情比率",
        )
        st.plotly_chart(fig, use_container_width=True)

        bubble_df = build_keyword_sentiment_bubble(sentiment_df)
        fig = px.scatter(
            bubble_df,
            x="frequency",
            y="sentiment_score",
            size="frequency",
            color="country",
            hover_name="term",
            title="키워드-감성 버블 차트 / キーワード感情バブルチャート",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("감성 모델이 없어서 감성 히트맵과 버블 차트는 건너뜁니다. / 感情モデルがないため、感情ヒートマップとバブルチャートはスキップします。")

    st.markdown("### 품사 비율 / 品詞比率")
    pos_ratio_df = build_pos_ratio_data(df)
    fig = px.bar(
        pos_ratio_df,
        x="country",
        y="ratio",
        color="pos_display",
        barmode="stack",
        title="국가별 품사 비율 / 国別品詞比率",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sankey / Sunburst")
    sankey_df = build_hierarchy_data(df)
    fig = build_sankey_figure(sankey_df)
    st.plotly_chart(fig, use_container_width=True)
    sunburst_df = build_sunburst_data(df, top_n=40)
    fig = px.sunburst(
        sunburst_df,
        path=["country", "level", "term"],
        values="count",
        title="국가-교육단계-키워드 Sunburst / 国-教育段階-キーワード Sunburst",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 바이올린 / TF-IDF 차이 / バイオリン / TF-IDF差")
    length_df = build_length_metrics(df)
    fig = px.violin(
        length_df,
        x="country",
        y="token_length",
        color="country",
        box=True,
        title="국가별 형태소 수 바이올린 플롯 / 国別形態素数バイオリンプロット",
    )
    st.plotly_chart(fig, use_container_width=True)
    tfidf_diff_df = build_tfidf_difference(df)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            x=tfidf_diff_df["term"],
            y=tfidf_diff_df["delta"],
            measure=["relative"] * len(tfidf_diff_df),
        )
    )
    fig.update_layout(title="TF-IDF 차이 워터폴 / TF-IDF差ウォーターフォール")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 클러스터 / 대표 문장 / クラスター / 代表文")
    cluster_count = st.slider("클러스터 수 / クラスター数", 2, 8, 4, key="cluster_count")
    countries = [country for country in ["KR", "JP"] if country in df["country"].unique()]
    if len(countries) == 2:
        left_col, right_col = st.columns(2)
        country_columns = {"KR": left_col, "JP": right_col}
    else:
        country_columns = {countries[0]: st.container()} if countries else {}

    for country in countries:
        subset = df[df["country"] == country].copy()
        with country_columns[country]:
            st.markdown(f"#### {country}")
            cluster_plot_df, cluster_centers_df, cluster_status = build_cluster_visualization(
                subset, cluster_count
            )
            st.caption(cluster_status)
            if cluster_plot_df is not None:
                fig = px.scatter(
                    cluster_plot_df,
                    x="x",
                    y="y",
                    color="cluster",
                    symbol="level",
                    hover_data=["text"],
                    title=f"{country} 문장 클러스터 시각화 / 文クラスタ可視化",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cluster_centers_df, use_container_width=True)

    st.markdown("### 단어 상관행렬 / 네트워크 커뮤니티 / 単語相関行列 / ネットワークコミュニティ")
    corr_df = build_term_correlation_matrix(df, top_n=15)
    if not corr_df.empty:
        fig = px.imshow(
            corr_df.set_index("term"),
            aspect="auto",
            color_continuous_scale="Blues",
            title="상위 단어 상관행렬 / 上位単語相関行列",
        )
        st.plotly_chart(fig, use_container_width=True)
    community_edges, community_nodes = build_community_network(df)
    if not community_edges.empty and not community_nodes.empty:
        fig = build_community_figure(community_edges, community_nodes)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 토픽 거리 맵 / トピック距離マップ")
    topic_count = st.slider("토픽 거리용 토픽 수 / トピック距離用トピック数", 2, 8, 4, key="topic_distance_count")
    topic_distance_df, topic_labels = build_topic_distance_map(df, topic_count)
    if topic_distance_df is not None:
        fig = px.scatter(
            topic_distance_df,
            x="x",
            y="y",
            text="topic",
            size="size",
            color="topic",
            title="토픽 거리 맵 / トピック距離マップ",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)


def build_term_frequency(df: pd.DataFrame) -> Counter[str]:
    tokens = tokenize_series(
        df["text"],
        df["country"],
        st.session_state.get("selected_pos_labels", POS_LABELS),
        df["language"] if "language" in df.columns else None,
    )
    return Counter(tokens)


def build_wordcloud_figure(df: pd.DataFrame, top_n: int):
    freq = build_term_frequency(df)
    if not freq:
        return None

    font_path = find_cjk_font(df["country"].unique().tolist())
    if not font_path:
        st.info("로컬 CJK 폰트를 찾지 못해 기본 폰트로 워드클라우드를 생성합니다. / ローカルCJKフォントが見つからないため、基本フォントでワードクラウドを生成します。")

    wc = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        colormap="viridis",
        font_path=font_path,
    ).generate_from_frequencies(dict(freq.most_common(top_n)))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def build_keyword_comparison(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    token_df = build_token_dataframe(df)
    if token_df.empty:
        return pd.DataFrame()

    overall_total = len(token_df)
    for country, group in token_df.groupby("country"):
        country_counts = group["term"].value_counts()
        other_counts = token_df[token_df["country"] != country]["term"].value_counts()
        country_total = len(group)
        other_total = max(overall_total - country_total, 1)
        terms = set(country_counts.head(top_n * 3).index) | set(other_counts.head(top_n * 3).index)

        scored = []
        for term in terms:
            country_share = country_counts.get(term, 0) / max(country_total, 1)
            other_share = other_counts.get(term, 0) / other_total
            score = country_share - other_share
            scored.append((term, country_counts.get(term, 0), country_share, score))

        for term, count, share, score in sorted(scored, key=lambda item: item[3], reverse=True)[:top_n]:
            records.append(
                {
                    "country": country,
                    "term": term,
                    "count": int(count),
                    "share": round(share, 4),
                    "score": round(score, 4),
                }
            )
    return pd.DataFrame(records)


def build_level_term_heatmap(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    token_df = build_token_dataframe(df)
    if token_df.empty:
        return pd.DataFrame()

    top_terms = token_df["term"].value_counts().head(top_n).index.tolist()
    subset = token_df[token_df["term"].isin(top_terms)].copy()
    subset["level_key"] = subset["country"] + " | " + subset["level"]
    heatmap = (
        subset.groupby(["term", "level_key"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ordered_level_keys = []
    for level in sort_levels(df["level"].unique().tolist()):
        for country in ["KR", "JP"]:
            key = f"{country} | {level}"
            if key in heatmap.columns:
                ordered_level_keys.append(key)
    remaining = [column for column in heatmap.columns if column not in {"term", *ordered_level_keys}]
    heatmap = heatmap[["term", *ordered_level_keys, *remaining]]
    return heatmap


def build_treemap_data(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    token_df = build_token_dataframe(df)
    if token_df.empty:
        return pd.DataFrame(columns=["country", "level", "term", "count"])
    treemap_df = (
        token_df.groupby(["country", "level", "term"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return treemap_df


def build_topic_model(df: pd.DataFrame, topic_count: int):
    sampled = sample_documents(df, limit=TOPIC_SAMPLE_LIMIT)
    documents = build_analysis_documents(sampled)
    if len(documents) < topic_count:
        return pd.DataFrame(), pd.DataFrame(), "토픽 모델링에 필요한 문서 수가 부족합니다. / トピックモデリングに必要な文書数が不足しています。"

    vectorizer = CountVectorizer(max_features=1500, min_df=5)
    try:
        matrix = vectorizer.fit_transform(documents["document"])
    except ValueError:
        return pd.DataFrame(), pd.DataFrame(), "토픽 모델링에 필요한 어휘가 부족합니다. / トピックモデリングに必要な語彙が不足しています。"

    lda = LatentDirichletAllocation(
        n_components=topic_count,
        learning_method="batch",
        random_state=42,
    )
    doc_topics = lda.fit_transform(matrix)
    feature_names = vectorizer.get_feature_names_out()

    topic_rows = []
    for topic_idx, weights in enumerate(lda.components_):
        top_indices = weights.argsort()[::-1][:10]
        for rank, term_idx in enumerate(top_indices, start=1):
            topic_rows.append(
                {
                    "topic": f"Topic {topic_idx + 1}",
                    "rank": rank,
                    "term": feature_names[term_idx],
                    "weight": float(weights[term_idx]),
                }
            )

    topic_docs = sampled[["country", "level", "text"]].copy()
    topic_docs["topic"] = [f"Topic {idx + 1}" for idx in doc_topics.argmax(axis=1)]
    return (
        pd.DataFrame(topic_rows),
        topic_docs,
        f"LDA 기반 토픽 {topic_count}개 추출 / LDAベースでトピック{topic_count}個抽出",
    )


def build_cooccurrence_edges(
    df: pd.DataFrame, min_edge: int = 5, top_nodes: int = 20
) -> pd.DataFrame:
    sampled = sample_documents(df, limit=COOCC_SAMPLE_LIMIT)
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()
    selected_pos = st.session_state.get("selected_pos_labels", POS_LABELS)

    for text, country, language in zip(sampled["text"], sampled["country"], sampled["language"]):
        terms = tokenize_text(text, country, selected_pos, language)
        unique_terms = list(dict.fromkeys(terms[:12]))
        node_counter.update(unique_terms)
        for i, source in enumerate(unique_terms):
            for target in unique_terms[i + 1 :]:
                if source == target:
                    continue
                edge_counter[tuple(sorted((source, target)))] += 1

    allowed_nodes = {term for term, _ in node_counter.most_common(top_nodes)}
    rows = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in edge_counter.items()
        if weight >= min_edge and source in allowed_nodes and target in allowed_nodes
    ]
    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight"])
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def build_cooccurrence_figure(edge_df: pd.DataFrame):
    graph = nx.Graph()
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight)

    positions = nx.spring_layout(graph, seed=42, weight="weight")
    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#94a3b8"),
        hoverinfo="none",
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        degree_weight = graph.degree(node, weight="weight")
        node_text.append(f"{node}<br>연결강도: {degree_weight:.1f}")
        node_size.append(12 + degree_weight * 0.4)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size, color=node_size, colorscale="Viridis", line_width=1),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="형태소 공기어 네트워크",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig


def build_ngram_counts(df: pd.DataFrame, n: int, top_n: int) -> pd.DataFrame:
    sampled = sample_documents(df, limit=TOPIC_SAMPLE_LIMIT)
    rows: list[dict[str, object]] = []
    selected_pos = st.session_state.get("selected_pos_labels", POS_LABELS)

    for country, group in sampled.groupby("country"):
        counter: Counter[str] = Counter()
        for text, language in zip(group["text"], group["language"]):
            terms = tokenize_text(text, country, selected_pos, language)
            ngrams = [" ".join(terms[idx : idx + n]) for idx in range(len(terms) - n + 1)]
            counter.update(ngrams)
        for ngram, count in counter.most_common(top_n):
            rows.append({"country": country, "ngram": ngram, "count": int(count)})
    return pd.DataFrame(rows)


def sample_documents(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    return df.sample(limit, random_state=42).copy()


def build_token_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    selected_pos = st.session_state.get("selected_pos_labels", POS_LABELS)
    rows: list[dict[str, str]] = []
    sampled = sample_documents(df, limit=TOKEN_SAMPLE_LIMIT)
    for country, language, level, text in zip(
        sampled["country"],
        sampled["language"],
        sampled["level"],
        sampled["text"],
    ):
        terms = tokenize_text(text, country, selected_pos, language)
        for term in terms:
            rows.append({"country": country, "level": level, "term": term})
    return pd.DataFrame(rows)


def build_analysis_documents(df: pd.DataFrame) -> pd.DataFrame:
    selected_pos = st.session_state.get("selected_pos_labels", POS_LABELS)
    documents = df[["country", "language", "level", "text"]].copy()
    documents["document"] = [
        " ".join(tokenize_text(text, country, selected_pos, language))
        for text, country, language in zip(documents["text"], documents["country"], documents["language"])
    ]
    documents = documents[documents["document"].str.strip() != ""].copy()
    return documents


def sort_levels(levels: list[str]) -> list[str]:
    unique_levels = list(dict.fromkeys(levels))
    order_index = {level: idx for idx, level in enumerate(LEVEL_ORDER)}
    return sorted(unique_levels, key=lambda level: (order_index.get(level, len(LEVEL_ORDER)), level))


def build_balanced_sample_rows(df: pd.DataFrame, per_country: int) -> pd.DataFrame:
    if df.empty:
        return df.head(0).copy()
    pieces = []
    for country in ["KR", "JP"]:
        subset = df[df["country"] == country]
        if subset.empty:
            continue
        pieces.append(subset.head(per_country))
    if not pieces:
        return df.head(min(len(df), per_country)).copy()
    return pd.concat(pieces, ignore_index=True)


def build_sentiment_heatmap_data(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    heatmap = (
        sentiment_df.groupby(["country", "level", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    heatmap["ratio"] = heatmap.groupby(["country", "level"])["count"].transform(
        lambda values: values / values.sum()
    )
    heatmap["bucket"] = heatmap["country"] + " | " + heatmap["level"]
    return (
        heatmap.pivot_table(index="bucket", columns="sentiment", values="ratio", fill_value=0)
        .reset_index()
    )


def build_keyword_sentiment_bubble(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    for row in sentiment_df.itertuples(index=False):
        terms = tokenize_text(
            row.text,
            row.country,
            st.session_state.get("selected_pos_labels", POS_LABELS),
            getattr(row, "language", ""),
        )
        for term in terms[:10]:
            rows.append(
                {
                    "country": row.country,
                    "term": term,
                    "sentiment_score": label_score.get(row.sentiment, 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["country", "term", "sentiment_score", "frequency"])
    return (
        frame.groupby(["country", "term"])
        .agg(sentiment_score=("sentiment_score", "mean"), frequency=("term", "size"))
        .reset_index()
        .sort_values("frequency", ascending=False)
        .head(60)
    )


def build_pos_ratio_data(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in sample_documents(df, TOPIC_SAMPLE_LIMIT).itertuples(index=False):
        tokens = tokenize_japanese(row.text) if row.country == "JP" else tokenize_korean(row.text)
        for token in tokens:
            label = map_pos_label(token.pos, row.country)
            if label:
                rows.append({"country": row.country, "pos_label": label})
    pos_df = pd.DataFrame(rows)
    if pos_df.empty:
        return pd.DataFrame(columns=["country", "pos_label", "ratio"])
    counts = pos_df.groupby(["country", "pos_label"]).size().reset_index(name="count")
    counts["ratio"] = counts.groupby("country")["count"].transform(lambda value: value / value.sum())
    counts["pos_display"] = counts["pos_label"].map(lambda label: POS_DISPLAY.get(label, label))
    return counts


def map_pos_label(pos: str, country: str) -> str | None:
    if country == "JP":
        mapping = {
            "名詞": "명사",
            "動詞": "동사",
            "形容詞": "형용사",
            "副詞": "부사",
            "連体詞": "관형사",
            "感動詞": "감탄사",
            "助詞": "조사",
        }
    else:
        mapping = {
            "Noun": "명사",
            "Verb": "동사",
            "Adjective": "형용사",
            "Adverb": "부사",
            "Determiner": "관형사",
            "Exclamation": "감탄사",
            "Josa": "조사",
        }
    if pos == "Fallback":
        return "명사"
    return mapping.get(pos)


def build_hierarchy_data(df: pd.DataFrame) -> pd.DataFrame:
    token_df = build_token_dataframe(df)
    top_terms = token_df["term"].value_counts().head(20).index.tolist()
    filtered = token_df[token_df["term"].isin(top_terms)].copy()
    return (
        filtered.groupby(["country", "level", "term"])
        .size()
        .reset_index(name="count")
    )


def build_sankey_figure(hierarchy_df: pd.DataFrame):
    labels: list[str] = []
    label_to_index: dict[str, int] = {}

    def ensure_label(value: str) -> int:
        if value not in label_to_index:
            label_to_index[value] = len(labels)
            labels.append(value)
        return label_to_index[value]

    source: list[int] = []
    target: list[int] = []
    value: list[int] = []

    level_group = hierarchy_df.groupby(["country", "level"])["count"].sum().reset_index()
    for row in level_group.itertuples(index=False):
        source.append(ensure_label(row.country))
        target.append(ensure_label(f"{row.country} | {row.level}"))
        value.append(int(row.count))
    for row in hierarchy_df.itertuples(index=False):
        source.append(ensure_label(f"{row.country} | {row.level}"))
        target.append(ensure_label(row.term))
        value.append(int(row.count))

    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=15, thickness=18),
            link=dict(source=source, target=target, value=value),
        )
    )
    fig.update_layout(title="country -> level -> term Sankey / 国 -> 教育段階 -> 用語 Sankey")
    return fig


def build_sunburst_data(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    token_df = build_token_dataframe(df)
    return (
        token_df.groupby(["country", "level", "term"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def build_length_metrics(df: pd.DataFrame) -> pd.DataFrame:
    length_df = sample_documents(df, TOPIC_SAMPLE_LIMIT).copy()
    length_df["token_length"] = [
        len(
            extract_analysis_tokens(
                text,
                country,
                st.session_state.get("selected_pos_labels", POS_LABELS),
                language,
            )
        )
        for text, country, language in zip(length_df["text"], length_df["country"], length_df["language"])
    ]
    return length_df


def build_tfidf_difference(df: pd.DataFrame) -> pd.DataFrame:
    documents = build_analysis_documents(sample_documents(df, TOPIC_SAMPLE_LIMIT))
    if documents.empty or documents["country"].nunique() < 2:
        return pd.DataFrame(columns=["term", "delta"])
    vectorizer = TfidfVectorizer(max_features=500, min_df=3)
    matrix = vectorizer.fit_transform(documents["document"])
    terms = vectorizer.get_feature_names_out()
    frame = pd.DataFrame(matrix.toarray(), columns=terms)
    frame["country"] = documents["country"].values
    kr_mean = frame[frame["country"] == "KR"].drop(columns=["country"]).mean()
    jp_mean = frame[frame["country"] == "JP"].drop(columns=["country"]).mean()
    delta = (kr_mean - jp_mean).sort_values(key=lambda value: value.abs(), ascending=False).head(15)
    return pd.DataFrame({"term": delta.index, "delta": delta.values})


def build_cluster_visualization(df: pd.DataFrame, cluster_count: int):
    sampled = sample_documents(df, CLUSTER_SAMPLE_LIMIT)
    embeddings, status = embed_texts(sampled["text"].tolist(), DEFAULT_EMBEDDING_MODEL)
    if embeddings is None or len(sampled) < cluster_count:
        return None, None, "클러스터링에 필요한 데이터가 부족합니다. / クラスタリングに必要なデータが不足しています。"
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    plot_df = sampled[["country", "level", "text"]].copy()
    plot_df["cluster"] = [f"Cluster {label + 1}" for label in labels]
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    representative_rows = []
    for cluster_id in range(cluster_count):
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_docs = sampled[cluster_mask]
        centroid = cluster_embeddings.mean(axis=0)
        similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).ravel()
        best_idx = int(similarities.argmax())
        representative_rows.append(
            {
                "cluster": f"Cluster {cluster_id + 1}",
                "country": cluster_docs.iloc[best_idx]["country"],
                "level": cluster_docs.iloc[best_idx]["level"],
                "text": cluster_docs.iloc[best_idx]["text"],
            }
        )
    return plot_df, pd.DataFrame(representative_rows), status


def build_term_correlation_matrix(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    documents = build_analysis_documents(sample_documents(df, COOCC_SAMPLE_LIMIT))
    if documents.empty:
        return pd.DataFrame()
    vectorizer = CountVectorizer(max_features=top_n, binary=True)
    try:
        matrix = vectorizer.fit_transform(documents["document"]).toarray()
    except ValueError:
        return pd.DataFrame()
    if matrix.shape[1] < 2:
        return pd.DataFrame()
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    terms = vectorizer.get_feature_names_out()
    corr_df = pd.DataFrame(corr, columns=terms)
    corr_df.insert(0, "term", terms)
    return corr_df


def build_community_network(df: pd.DataFrame):
    edge_df = build_cooccurrence_edges(df, min_edge=3, top_nodes=20)
    if edge_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    graph = nx.Graph()
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    node_rows = []
    for idx, community in enumerate(communities, start=1):
        for node in community:
            node_rows.append({"node": node, "community": f"C{idx}"})
    return edge_df, pd.DataFrame(node_rows)


def build_community_figure(edge_df: pd.DataFrame, node_df: pd.DataFrame):
    graph = nx.Graph()
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight)
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    node_map = node_df.set_index("node")["community"].to_dict()
    plot_rows = []
    for node, (x, y) in positions.items():
        plot_rows.append({"node": node, "x": x, "y": y, "community": node_map.get(node, "C0")})
    plot_df = pd.DataFrame(plot_rows)
    return px.scatter(plot_df, x="x", y="y", color="community", text="node", title="네트워크 커뮤니티 / ネットワークコミュニティ")


def build_topic_distance_map(df: pd.DataFrame, topic_count: int):
    topic_terms, _, _ = build_topic_model(df, topic_count)
    if topic_terms.empty:
        return None, []
    topic_matrix = (
        topic_terms.pivot_table(index="topic", columns="term", values="weight", fill_value=0)
        .sort_index()
    )
    if len(topic_matrix) < 2:
        coords = np.column_stack([np.arange(len(topic_matrix)), np.zeros(len(topic_matrix))])
    else:
        distances = pdist(topic_matrix.values, metric="cosine")
        coords = PCA(n_components=2, random_state=42).fit_transform(squareform(distances))
    plot_df = pd.DataFrame(
        {
            "topic": topic_matrix.index,
            "x": coords[:, 0],
            "y": coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(topic_matrix)),
            "size": topic_matrix.sum(axis=1).values,
        }
    )
    return plot_df, topic_matrix.index.tolist()

def tokenize_series(
    texts: pd.Series,
    countries: pd.Series,
    selected_pos_labels: list[str],
    languages: pd.Series | None = None,
) -> list[str]:
    tokens: list[str] = []
    language_values = languages.tolist() if languages is not None else [None] * len(texts)
    for text, country, language in zip(texts.tolist(), countries.tolist(), language_values):
        tokens.extend(tokenize_text(text, country, selected_pos_labels, language))
    return tokens


def tokenize_text(
    text: str,
    country: str,
    selected_pos_labels: list[str] | None = None,
    language: str | None = None,
) -> list[str]:
    analysis_country = detect_text_language(text, country, language)
    raw_tokens = extract_analysis_tokens(text, analysis_country, selected_pos_labels or POS_LABELS)
    stopwords = JP_STOPWORDS if analysis_country == "JP" else KR_STOPWORDS
    return [
        token.lemma
        for token in raw_tokens
        if len(token.lemma) > 1 and token.lemma not in stopwords and not token.lemma.isdigit()
    ]


def extract_analysis_tokens(
    text: str,
    country: str,
    selected_pos_labels: list[str],
    language: str | None = None,
) -> list[MorphToken]:
    analysis_country = detect_text_language(text, country, language)
    raw_tokens = tokenize_japanese(text) if analysis_country == "JP" else tokenize_korean(text)
    allowed_pos = select_allowed_pos(analysis_country, selected_pos_labels)
    return [
        token
        for token in raw_tokens
        if (token.pos in allowed_pos or token.pos == "Fallback") and len(token.lemma) > 1
    ]


def select_allowed_pos(country: str, selected_pos_labels: list[str]) -> set[str]:
    pos_labels = selected_pos_labels or POS_LABELS
    if country == "JP":
        return set().union(*(JP_POS_MAP[label] for label in pos_labels))
    return set().union(*(KR_POS_MAP[label] for label in pos_labels))


def detect_text_language(text: str, fallback_country: str, language: str | None = None) -> str:
    if language in {"ko", "kr"}:
        return "KR"
    if language in {"ja", "jp"}:
        return "JP"
    if re.search(r"[ぁ-んァ-ン一-龥]", text):
        return "JP"
    if re.search(r"[가-힣]", text):
        return "KR"
    return fallback_country


def format_morpheme_tokens(tokens: list[MorphToken], limit: int = 12) -> str:
    preview = [f"{token.lemma}/{token.pos}" for token in tokens[:limit]]
    return ", ".join(preview) if preview else "-"


@lru_cache(maxsize=1)
def get_okt():
    try:
        from konlpy.tag import Okt
        import jpype

        try:
            return Okt()
        except Exception:
            jvm_path = find_valid_jvm_path(jpype.getDefaultJVMPath())
            if jvm_path:
                return Okt(jvmpath=jvm_path)
            return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_fugashi():
    try:
        import fugashi

        return fugashi.Tagger()
    except Exception:
        return None


def find_valid_jvm_path(default_path: str | None) -> str | None:
    candidates = []
    if default_path:
        candidates.append(default_path)
    candidates.extend(
        [
            "/Library/Java/JavaVirtualMachines/jdk-15.0.2.jdk/Contents/Home/lib/server/libjvm.dylib",
            "/Library/Java/JavaVirtualMachines/jdk-15.0.2.jdk/Contents/Home/lib/libjli.dylib",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def build_analyzer_status() -> str:
    kr_engine = "KoNLPy Okt" if get_okt() else "규칙기반 경량 분석 / ルールベース軽量分析"
    jp_engine = "fugashi + UniDic" if get_fugashi() else "정규식 폴백 / 正規表現フォールバック"
    return f"현재 형태소 분석기 / 現在の形態素分析器: KR={kr_engine}, JP={jp_engine}"


def tokenize_korean(text: str) -> list[MorphToken]:
    okt = get_okt()
    if okt:
        return [
            MorphToken(surface=token, lemma=token, pos=pos)
            for token, pos in okt.pos(text, stem=True)
        ]
    return heuristic_korean_tokens(text)


def tokenize_japanese(text: str) -> list[MorphToken]:
    tagger = get_fugashi()
    if tagger:
        tokens = []
        for word in tagger(text):
            surface = word.surface.strip()
            if surface:
                lemma = getattr(word.feature, "lemma", surface) or surface
                pos = getattr(word.feature, "pos1", "Unknown") or "Unknown"
                if lemma == "*":
                    lemma = surface
                tokens.append(MorphToken(surface=surface, lemma=lemma, pos=pos))
        return tokens
    return [
        MorphToken(surface=token, lemma=token, pos="Fallback")
        for token in re.findall(r"(?:[一-龥]+[ぁ-ん]*)|(?:[ぁ-ん]{2,})|(?:[ァ-ンー]{2,})", text)
    ]


def heuristic_korean_tokens(text: str) -> list[MorphToken]:
    raw_tokens = re.findall(r"[가-힣]{2,}", text)
    tokens: list[MorphToken] = []
    for surface in raw_tokens:
        lemma, pos = normalize_korean_fallback(surface)
        tokens.append(MorphToken(surface=surface, lemma=lemma, pos=pos))
    return tokens


def normalize_korean_fallback(surface: str) -> tuple[str, str]:
    token = strip_korean_particle(surface)
    lemma, pos = lemmatize_korean_predicate(token)
    if len(lemma) < 2:
        return surface, "Fallback"
    return lemma, pos


def strip_korean_particle(token: str) -> str:
    for suffix in KR_PARTICLES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def lemmatize_korean_predicate(token: str) -> tuple[str, str]:
    for suffix in KR_VERB_ENDINGS:
        if token.endswith(suffix) and len(token) - len(suffix) >= 1:
            stem = token[: -len(suffix)]
            if not stem:
                break
            if stem.endswith(("있", "없", "좋", "즐겁", "재미있", "행복", "신나", "어렵")):
                return stem + "다", "Adjective"
            return stem + "다", "Verb"
    return token, "Noun"


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline(model_name: str):
    if LIGHTWEIGHT_MODE:
        return None
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=256,
        )
    except Exception:
        return None


def predict_sentiment(df: pd.DataFrame, model_name: str):
    classifier = get_sentiment_pipeline(model_name)
    if classifier is None:
        heuristic_df = predict_sentiment_heuristic(df)
        if heuristic_df is not None:
            reason = (
                "경량 모드라 휴리스틱 감성분석 사용 / 軽量モードのためヒューリスティック感情分析を使用"
                if LIGHTWEIGHT_MODE
                else "transformers 또는 torch가 없어 휴리스틱 감성분석 사용 / transformersまたはtorchがないためヒューリスティック感情分析を使用"
            )
            return heuristic_df, reason
        return None, "transformers 또는 torch가 없어 감성분석을 건너뜁니다. / transformersまたはtorchがないため感情分析をスキップします。"

    texts = df["text"].tolist()
    outputs = classifier(texts, batch_size=16)
    sentiment_df = df.copy()
    raw_sentiments = [normalize_label(output["label"]) for output in outputs]
    raw_scores = [float(output["score"]) for output in outputs]
    adjusted = [
        calibrate_sentiment_label(text, country, language, sentiment, score)
        for text, country, language, sentiment, score in zip(
            sentiment_df["text"],
            sentiment_df["country"],
            sentiment_df["language"],
            raw_sentiments,
            raw_scores,
        )
    ]
    sentiment_df["sentiment"] = [label for label, _ in adjusted]
    sentiment_df["score"] = [score for _, score in adjusted]
    return sentiment_df, f"{model_name} 모델로 {len(sentiment_df):,}건 예측 / {model_name}モデルで{len(sentiment_df):,}件予測"


def normalize_label(label: str) -> str:
    value = label.lower()
    if "positive" in value or value.endswith("5 stars") or value.endswith("4 stars"):
        return "positive"
    if "negative" in value or value.endswith("1 star") or value.endswith("2 stars"):
        return "negative"
    return "neutral"


def predict_sentiment_heuristic(df: pd.DataFrame):
    if df.empty:
        return df.copy()
    sentiment_df = df.copy()
    labels = []
    scores = []
    for row in sentiment_df.itertuples(index=False):
        analysis_country = detect_text_language(row.text, row.country, getattr(row, "language", None))
        negative_cues = JP_NEGATIVE_CUES if analysis_country == "JP" else KR_NEGATIVE_CUES
        positive_cues = JP_POSITIVE_CUES if analysis_country == "JP" else KR_POSITIVE_CUES
        lower_text = row.text.lower()
        negative_hits = sum(1 for cue in negative_cues if cue.lower() in lower_text)
        positive_hits = sum(1 for cue in positive_cues if cue.lower() in lower_text)
        if negative_hits > positive_hits:
            labels.append("negative")
            scores.append(min(0.55 + 0.08 * negative_hits, 0.95))
        elif positive_hits > negative_hits:
            labels.append("positive")
            scores.append(min(0.55 + 0.08 * positive_hits, 0.95))
        else:
            labels.append("neutral")
            scores.append(0.5)
    sentiment_df["sentiment"] = labels
    sentiment_df["score"] = scores
    return sentiment_df


def calibrate_sentiment_label(
    text: str,
    country: str,
    language: str,
    sentiment: str,
    score: float,
) -> tuple[str, float]:
    analysis_country = detect_text_language(text, country, language)
    negative_cues = JP_NEGATIVE_CUES if analysis_country == "JP" else KR_NEGATIVE_CUES
    positive_cues = JP_POSITIVE_CUES if analysis_country == "JP" else KR_POSITIVE_CUES
    lower_text = text.lower()

    negative_hits = sum(1 for cue in negative_cues if cue.lower() in lower_text)
    positive_hits = sum(1 for cue in positive_cues if cue.lower() in lower_text)

    if negative_hits >= 1 and positive_hits == 0 and sentiment == "positive":
        return "negative", max(score, 0.7)
    if positive_hits >= 1 and negative_hits == 0 and sentiment == "negative":
        return "positive", max(score, 0.7)
    if negative_hits >= 2 and sentiment == "neutral":
        return "negative", max(score, 0.65)
    if positive_hits >= 2 and sentiment == "neutral":
        return "positive", max(score, 0.65)
    return sentiment, score


@st.cache_resource(show_spinner=False)
def get_sentence_transformer(model_name: str):
    if LIGHTWEIGHT_MODE:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        return None


def build_tsne_projection(df: pd.DataFrame, model_name: str):
    embeddings, status = embed_texts(df["text"].tolist(), model_name)
    if embeddings is None:
        return None, status

    perplexity = min(30, max(5, math.floor(len(df) / 20)))
    perplexity = min(perplexity, len(df) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
    )
    coords = tsne.fit_transform(embeddings)

    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    return plot_df, status


def build_umap_projection(df: pd.DataFrame, model_name: str):
    embeddings, status = embed_texts(df["text"].tolist(), model_name)
    if embeddings is None:
        return None, status

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(df) - 1),
            min_dist=0.1,
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        projection_status = "UMAP 투영 사용 / UMAP投影を使用"
    except Exception:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        projection_status = "umap-learn이 없어 PCA 2D 투영으로 대체 / umap-learnがないためPCA 2D投影で代替"

    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    return plot_df, f"{status} / {projection_status}"


def embed_texts(texts: list[str], model_name: str):
    embedder = get_sentence_transformer(model_name)
    if embedder is not None:
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings), f"{model_name} 문장 임베딩 사용 / {model_name}文埋め込み使用"

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] < 2:
        return None, "임베딩을 만들기 위한 유효 어휘 수가 부족합니다. / 埋め込み生成に必要な有効語彙数が不足しています。"
    svd_dim = min(50, matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=svd_dim, random_state=42)
    embeddings = reducer.fit_transform(matrix)
    return embeddings, "sentence-transformers가 없어 TF-IDF + SVD로 대체 / sentence-transformersがないためTF-IDF + SVDで代替"


def find_cjk_font(countries: list[str] | None = None) -> str | None:
    countries = countries or []
    has_kr = "KR" in countries
    has_jp = "JP" in countries

    if has_jp and not has_kr:
        candidates = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        ]
    elif has_kr and not has_jp:
        candidates = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


if __name__ == "__main__":
    main()
