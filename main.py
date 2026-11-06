import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="흡착률 예측 분석", layout="wide")

URL = "https://raw.githubusercontent.com/dg2025519-create/suhangpyeongga/refs/heads/main/DataSet.csv"
TARGET = "Removal"
LEAK = "Max Adsorp Capacity"   # 누수 변수: 표시만 하고 분석/모델에서는 제외
NUM_FEATURES = ["pH", "Adsorbent Dose (g/L)", "Concentration (mg/L)", "Temperature", "Time (min)"]
ANALYSIS_COLS = NUM_FEATURES + [TARGET]
SHORT = {"pH": "pH", "Adsorbent Dose (g/L)": "Dose", "Concentration (mg/L)": "Conc",
         "Temperature": "Temp", "Time (min)": "Time"}
REV = {v: k for k, v in SHORT.items()}


@st.cache_data
def load_data():
    df = pd.read_csv(URL, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Type"] = df["Type"].str.strip()
    return df


df = load_data()
X_all = pd.concat([df[NUM_FEATURES], pd.get_dummies(df["Type"], prefix="Type").astype(int)], axis=1)
y_all = df[TARGET]
type_cols = [c for c in X_all.columns if c.startswith("Type_")]


def strength(r):
    a = abs(r)
    if a >= 0.7: return "강한"
    if a >= 0.4: return "보통 수준의"
    if a >= 0.2: return "약한"
    return "거의 없는"


def direction(r):
    return "양(+)" if r > 0 else "음(−)"


def explain(text):
    st.info("💡 **해석**\n\n" + text)


st.sidebar.title("📌 메뉴")
page = st.sidebar.radio("페이지 선택", ["1. 데이터 보기", "2. 상관관계 분석",
                                        "3. 최적 변수 조합 탐색", "4. 조합·모델 비교"])
st.title("🧪 실험조건으로 흡착률(Removal)을 예측할 수 있을까?")

# ================= 1. 데이터 =================
if page == "1. 데이터 보기":
    st.header("📄 데이터 확인")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", df.shape[0])
    c2.metric("열 수", df.shape[1])
    c3.metric("결측치", int(df.isna().sum().sum()))
    c4.metric("흡착제 종류", df["Type"].nunique())

    types = df["Type"].unique().tolist()
    sel = st.multiselect("Type 필터", types, default=types)
    st.dataframe(df[df["Type"].isin(sel)], use_container_width=True, height=400)

    st.subheader("기초 통계량 (분석에 사용하는 변수)")
    st.dataframe(df[ANALYSIS_COLS].describe().T, use_container_width=True)

    st.subheader("Type별 Removal 평균")
    st.dataframe(df.groupby("Type")[TARGET].agg(["mean", "std", "count"]), use_container_width=True)

    st.info("ℹ️ `Max Adsorp Capacity`는 Removal·농도·흡착제 양으로 계산되는 값이라 "
            "상관관계 분석과 예측 모델에서 제외했습니다(데이터 누수 방지). "
            "`Concentration`은 오염물질의 초기 농도(mg/L)입니다.")

# ================= 2. 상관관계 =================
elif page == "2. 상관관계 분석":
    st.header("📊 상관관계 분석")
    st.caption("누수 변수(Max Adsorp Capacity)는 제외하고, 실험조건 5개와 Removal만 분석합니다.")
    tabs = st.tabs(["히트맵", "타깃과의 상관", "조건별 변화", "산점도", "Type별 비교", "분포"])

    with tabs[0]:
        method = st.radio("상관계수 종류", ["pearson", "spearman"], horizontal=True)
        corr = df[ANALYSIS_COLS].corr(method=method)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)
        ct = corr[TARGET].drop(TARGET)
        top = ct.abs().idxmax()
        feat_corr = corr.loc[NUM_FEATURES, NUM_FEATURES].where(~np.eye(len(NUM_FEATURES), dtype=bool))
        pair = feat_corr.abs().stack().idxmax()
        explain(
            f"- 색이 **빨강**이면 양의 상관(함께 증가), **파랑**이면 음의 상관(한쪽이 늘면 다른 쪽은 감소), 흰색에 가까우면 관계가 약합니다.\n"
            f"- Removal과 가장 관련이 큰 변수는 **{top}** (r = {ct[top]:.2f}, {strength(ct[top])} {direction(ct[top])}상관)입니다.\n"
            f"- 입력 변수끼리 가장 상관이 큰 쌍은 **{pair[0]} – {pair[1]}** (r = {feat_corr.loc[pair]:.2f})입니다.\n"
            f"- 이 데이터는 한 번에 한 변수만 바꾼 실험이 모여 있어 변수끼리 서로 독립적으로 설계되었고, "
            f"그래서 전체 상관계수가 전반적으로 낮게 나옵니다. **상관이 낮다고 영향이 없다는 뜻은 아니며**, "
            f"'조건별 변화' 탭에서 변수별 효과를 확인해야 합니다.\n"
            f"- 피어슨은 직선 관계, 스피어만은 순위 관계(곡선이어도 한 방향이면 높게 나옴)를 봅니다. 두 값 차이가 크면 비선형 관계라는 단서입니다."
        )

    with tabs[1]:
        corr_t = df[NUM_FEATURES].corrwith(df[TARGET]).sort_values()
        fig = px.bar(x=corr_t.values, y=corr_t.index, orientation="h", color=corr_t.values,
                     color_continuous_scale="RdBu_r", range_color=[-1, 1],
                     labels={"x": "Removal과의 상관계수", "y": "변수"})
        st.plotly_chart(fig, use_container_width=True)
        lines = [f"- **{k}**: r = {v:.2f} → {strength(v)} {direction(v)}상관" for k, v in corr_t.iloc[::-1].items()]
        explain("막대가 오른쪽(+)이면 값이 클수록 Removal이 높고, 왼쪽(−)이면 값이 클수록 Removal이 낮은 경향입니다.\n\n"
                + "\n".join(lines) +
                "\n\n단순 상관계수는 **직선 관계만** 보여줍니다. 농도처럼 곡선으로 줄어드는 관계나 "
                "시간처럼 일정 시점 이후 평평해지는 관계는 과소평가될 수 있습니다.")

    with tabs[2]:
        st.markdown("실험 시리즈별로 **한 변수만 바꾼 실험**을 골라 보여줍니다. 같은 조건이 반복된 경우 평균을 사용합니다.")
        series_fixed = {
            "pH": {"Adsorbent Dose (g/L)": 1, "Concentration (mg/L)": 10, "Temperature": 25, "Time (min)": 300},
            "Adsorbent Dose (g/L)": {"pH": 5, "Concentration (mg/L)": 10, "Temperature": 25, "Time (min)": 300},
            "Concentration (mg/L)": {"pH": 5, "Adsorbent Dose (g/L)": 2, "Temperature": 25, "Time (min)": 300},
            "Temperature": {"pH": 5, "Adsorbent Dose (g/L)": 2, "Concentration (mg/L)": 100, "Time (min)": 300},
            "Time (min)": {"pH": 5, "Adsorbent Dose (g/L)": 2, "Concentration (mg/L)": 100, "Temperature": 25},
        }
        var = st.selectbox("변화시킨 변수", NUM_FEATURES, index=2)
        fixed = series_fixed[var]
        mask = np.ones(len(df), dtype=bool)
        for c, v in fixed.items():
            mask &= (df[c] == v)
        sub = df[mask].groupby(["Type", var])[TARGET].mean().reset_index().sort_values(var)
        st.caption("고정 조건: " + ", ".join(f"{k}={v}" for k, v in fixed.items()))
        fig = px.line(sub, x=var, y=TARGET, color="Type", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        lines = []
        for t, g in sub.groupby("Type"):
            g = g.sort_values(var)
            first, last = g.iloc[0], g.iloc[-1]
            peak = g.loc[g[TARGET].idxmax()]
            lines.append(f"- **{t}**: {var} {first[var]:g} → {last[var]:g}일 때 Removal {first[TARGET]:.1f}% → {last[TARGET]:.1f}% "
                         f"({last[TARGET]-first[TARGET]:+.1f}%p), 최대값 {peak[TARGET]:.1f}% ({var} = {peak[var]:g})")
        hint = {
            "pH": "pH가 산성(3)에서 중성~염기성으로 갈수록 Removal이 높아지는지 확인하세요. 표면 전하와 오염물질 상태가 바뀌기 때문입니다.",
            "Adsorbent Dose (g/L)": "흡착제를 늘리면 흡착 자리가 늘어 Removal이 오르지만, 어느 양부터는 증가폭이 작아지는(수확 체감) 모양이 나타납니다.",
            "Concentration (mg/L)": "오염물질 농도가 높아질수록 흡착제의 자리가 포화되어 Removal(%)이 떨어지는 것이 일반적입니다.",
            "Temperature": "온도가 오를 때 Removal이 오르면 흡열 과정(온도가 높을수록 유리)임을 시사합니다.",
            "Time (min)": "초반에 빠르게 오르다 일정 시간 이후 평평해지면 흡착 평형에 도달했다는 뜻입니다.",
        }[var]
        explain("\n".join(lines) + f"\n\n{hint}\n\n세 종류 모두 비슷한 방향으로 움직이면 Type과 무관한 공통 경향이고, "
                "선이 서로 많이 떨어져 있으면 Type이 Removal 수준 자체에 영향을 준다는 뜻입니다.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X축", NUM_FEATURES)
        yv = c2.selectbox("Y축", [TARGET] + NUM_FEATURES, index=0)
        fig = px.scatter(df, x=x, y=yv, color="Type", trendline="ols" if x != yv else None)
        st.plotly_chart(fig, use_container_width=True)
        if x != yv:
            r_all = df[x].corr(df[yv])
            per = [f"{t}: r = {g[x].corr(g[yv]):.2f}" for t, g in df.groupby("Type")]
            explain(f"- 전체 데이터의 상관계수는 **r = {r_all:.2f}** ({strength(r_all)} {direction(r_all)}상관)입니다.\n"
                    f"- Type별 상관계수: {' / '.join(per)}\n"
                    f"- 점이 추세선 주위에 모여 있으면 직선 관계가 뚜렷하고, 넓게 퍼져 있거나 곡선 모양이면 "
                    f"다른 조건(농도, 용량 등)의 영향이 섞였거나 비선형 관계입니다.\n"
                    f"- 같은 X값에서 점이 세로로 흩어져 있다면 그 차이는 **다른 실험조건** 때문입니다.")
        else:
            explain("X축과 Y축이 같은 변수입니다. 서로 다른 변수를 선택해 주세요.")

        st.subheader("산점도 행렬")
        st.plotly_chart(px.scatter_matrix(df, dimensions=ANALYSIS_COLS, color="Type", height=800),
                        use_container_width=True)
        explain("모든 변수 쌍의 관계를 한 번에 보는 그래프입니다. 맨 아래 줄(또는 맨 오른쪽 열)이 Removal과의 관계입니다.\n\n"
                "- 입력 변수는 **몇 개의 값만 가지는 점**(예: 온도 25/35/45/55)이라 세로 또는 가로 줄무늬로 보입니다. "
                "실험을 정해진 수준으로 설계했기 때문입니다.\n"
                "- 줄무늬 안에서 Removal이 넓게 퍼져 있으면 그 변수 하나로는 설명이 부족하고, 여러 조건의 조합이 필요하다는 뜻입니다.")

    with tabs[4]:
        st.plotly_chart(px.box(df, x="Type", y=TARGET, color="Type", points="all"), use_container_width=True)
        stat = df.groupby("Type")[TARGET].agg(["mean", "median", "std", "min", "max"])
        best_t, worst_t = stat["mean"].idxmax(), stat["mean"].idxmin()
        lines = [f"- **{t}**: 평균 {r['mean']:.1f}%, 중앙값 {r['median']:.1f}%, 표준편차 {r['std']:.1f}, 범위 {r['min']:.1f}~{r['max']:.1f}"
                 for t, r in stat.iterrows()]
        explain("상자는 중간 50% 구간, 상자 안 선은 중앙값, 점은 개별 실험입니다.\n\n" + "\n".join(lines) +
                f"\n\n평균 Removal은 **{best_t}**가 가장 높고 **{worst_t}**가 가장 낮으며 차이는 {stat['mean'].max()-stat['mean'].min():.1f}%p입니다. "
                "표준편차가 크고 범위가 겹치므로, Type 차이보다 **실험조건(특히 농도)에 따른 변동이 더 큽니다.**")

        g = st.selectbox("그룹 변수(막대)", NUM_FEATURES, index=2)
        agg = df.groupby(["Type", g])[TARGET].mean().reset_index()
        st.plotly_chart(px.bar(agg, x=g, y=TARGET, color="Type", barmode="group"), use_container_width=True)
        gm = df.groupby(g)[TARGET].mean()
        explain(f"{g} 값별로 Type의 평균 Removal을 비교한 그래프입니다.\n\n"
                f"- 전체 평균이 가장 높은 구간은 **{g} = {gm.idxmax():g}** ({gm.max():.1f}%), 가장 낮은 구간은 "
                f"**{g} = {gm.idxmin():g}** ({gm.min():.1f}%)입니다.\n"
                f"- 같은 {g}에서 막대 높이가 Type마다 다르면 그 조건에서 흡착제 성능 차이가 있다는 뜻입니다.\n"
                f"- 각 {g} 값에 해당하는 실험 수가 다르므로(일부 값은 다른 조건 실험이 섞임) 평균은 참고용으로 보세요.")

    with tabs[5]:
        v = st.selectbox("변수 선택", ANALYSIS_COLS, key="hist")
        st.plotly_chart(px.histogram(df, x=v, color="Type", nbins=30, marginal="box", barmode="overlay"),
                        use_container_width=True)
        s = df[v]
        sk = s.skew()
        shape = "오른쪽으로 긴 꼬리(큰 값이 드물게 존재)" if sk > 0.5 else "왼쪽으로 긴 꼬리(작은 값이 드물게 존재)" if sk < -0.5 else "비교적 좌우 대칭"
        explain(f"- **{v}**: 평균 {s.mean():.2f}, 중앙값 {s.median():.2f}, 최소 {s.min():g}, 최대 {s.max():g}, 왜도 {sk:.2f} → {shape}.\n"
                f"- 평균과 중앙값 차이가 크면 치우친 분포입니다.\n"
                + ("- Removal이 높은 쪽(90% 이상)에 몰려 있다면, 저농도 조건의 실험이 많기 때문입니다. "
                   "모델이 높은 값은 잘 맞추고 낮은 값(고농도 조건)은 데이터가 적어 덜 정확할 수 있습니다.\n"
                   if v == TARGET else
                   "- 입력 변수는 설계된 몇 개 값에 몰려 있어 막대가 띄엄띄엄 나타납니다. 특정 값의 막대가 높은 것은 그 조건이 기준 조건으로 여러 번 반복되었기 때문입니다.\n")
                + "- 박스플롯의 점이 상자 밖에 있으면 이상치 후보입니다.")

# ================= 3. 최적 변수 조합 탐색 =================
elif page == "3. 최적 변수 조합 탐색":
    st.header("🔎 최적 변수 조합 탐색 (선형회귀)")
    st.markdown("실험조건 5개 중 **1~5개를 고르는 모든 조합(31개)** 으로 선형회귀 모델을 학습해, R²가 가장 높은 조합을 찾습니다.")

    c1, c2 = st.columns(2)
    use_type = c1.checkbox("Type(흡착제 종류) 항상 포함", value=True)
    metric_name = c2.radio("정렬 기준", ["R² (5-fold CV)", "R² (Type 제외 검증)"])

    @st.cache_data(show_spinner=False)
    def search(use_type):
        kf = KFold(5, shuffle=True, random_state=42)
        rows = []
        for k in range(1, len(NUM_FEATURES) + 1):
            for combo in combinations(NUM_FEATURES, k):
                cols = list(combo) + (type_cols if use_type else [])
                X = X_all[cols]
                cvp = cross_val_predict(LinearRegression(), X, y_all, cv=kf)
                lo = pd.Series(index=df.index, dtype=float)
                for t in df["Type"].unique():
                    idx = df["Type"] == t
                    m = LinearRegression().fit(X[~idx], y_all[~idx])
                    lo[idx] = m.predict(X[idx])
                rows.append({
                    "변수 조합": " + ".join(SHORT[c] for c in combo),
                    "변수 수": k,
                    "R² (5-fold CV)": r2_score(y_all, cvp),
                    "R² (Type 제외 검증)": r2_score(y_all, lo),
                    "MAE (CV)": mean_absolute_error(y_all, cvp),
                })
        return pd.DataFrame(rows)

    res = search(use_type).sort_values(metric_name, ascending=False).reset_index(drop=True)
    res.index = res.index + 1

    best = res.iloc[0]
    st.success(f"🏆 {metric_name} 기준 최고 조합: **{best['변수 조합']}** "
               f"(변수 {int(best['변수 수'])}개, R² = {best[metric_name]:.3f}, MAE = {best['MAE (CV)']:.2f}%p)")

    st.subheader("조합별 성능 순위")
    st.dataframe(res.style.format({"R² (5-fold CV)": "{:.3f}", "R² (Type 제외 검증)": "{:.3f}", "MAE (CV)": "{:.2f}"}),
                 use_container_width=True, height=450)

    top = res.head(15).iloc[::-1]
    fig = px.bar(top, x=metric_name, y="변수 조합", orientation="h", color="변수 수",
                 color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
    second = res.iloc[1]
    explain(f"상위 15개 조합입니다. 막대가 길수록 예측력이 높고, 색은 변수 개수입니다.\n\n"
            f"- 1위 **{best['변수 조합']}** ({best[metric_name]:.3f}), 2위 **{second['변수 조합']}** ({second[metric_name]:.3f})\n"
            f"- 상위권에 공통으로 들어가는 변수가 **예측에 꼭 필요한 변수**입니다.\n"
            f"- 1위와 하위권의 R² 차이는 {best[metric_name] - res[metric_name].iloc[-1]:.3f}입니다.")

    by_k = res.groupby("변수 수")[metric_name].max().reset_index()
    fig = px.line(by_k, x="변수 수", y=metric_name, markers=True)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    simple = res[res[metric_name] >= best[metric_name] - 0.01].sort_values(["변수 수", metric_name], ascending=[True, False]).iloc[0]
    explain(f"변수 개수별로 달성 가능한 최고 R²입니다. 곡선이 평평해지는 지점부터는 변수를 추가해도 이득이 작습니다.\n\n"
            f"- 변수 1개 최고 R²: {by_k[metric_name].iloc[0]:.3f} → 5개: {by_k[metric_name].iloc[-1]:.3f}\n"
            f"- 최고 R²와 차이가 0.01 이내이면서 변수가 가장 적은 조합: **{simple['변수 조합']}** "
            f"({int(simple['변수 수'])}개, R² = {simple[metric_name]:.3f}) → 실험을 줄이고 싶다면 이 조합이 효율적입니다.")

    contrib = []
    for full, s in SHORT.items():
        has = res["변수 조합"].str.split(" \\+ ").apply(lambda l: s in l)
        contrib.append({"변수": s, "포함 시 평균 R²": res.loc[has, metric_name].mean(),
                        "미포함 시 평균 R²": res.loc[~has, metric_name].mean()})
    ct = pd.DataFrame(contrib)
    ct["차이"] = ct["포함 시 평균 R²"] - ct["미포함 시 평균 R²"]
    ct = ct.sort_values("차이", ascending=False)
    st.plotly_chart(px.bar(ct, x="변수", y="차이", color="차이", color_continuous_scale="RdBu",
                           labels={"차이": "포함 시 − 미포함 시 평균 R²"}), use_container_width=True)
    explain(f"각 변수가 들어간 조합과 빠진 조합의 평균 R² 차이입니다. 클수록 그 변수가 성능을 끌어올립니다.\n\n"
            f"- 기여도 1위 **{ct.iloc[0]['변수']}** ({ct.iloc[0]['차이']:+.3f}), "
            f"꼴찌 **{ct.iloc[-1]['변수']}** ({ct.iloc[-1]['차이']:+.3f})\n"
            f"- 차이가 0에 가깝거나 음수면 이 데이터에서는 예측에 도움이 되지 않는 변수입니다.")

    st.caption("⚠️ 이 데이터는 한 변수씩 바꾼 실험이라, 변수를 빼면 같은 입력값에 서로 다른 Removal이 대응되어 R²가 크게 떨어질 수 있습니다. "
               "또 31개 조합 중 최고를 고르는 것이므로 점수가 다소 낙관적입니다. 'Type 제외 검증' 기준 순위도 함께 확인하세요.")

# ================= 4. 조합·모델 비교 =================
else:
    st.header("⚖️ 조합 × 모델 비교 (베이스라인 포함)")
    st.markdown("세 가지 변수 조합(A/B/C)을 **베이스라인, 선형회귀, 다항회귀, 랜덤포레스트, 그래디언트부스팅**으로 비교합니다.")

    with st.expander("📌 베이스라인이란? 왜 필요한가?", expanded=False):
        st.markdown(
            "**베이스라인 = 변수를 전혀 쓰지 않고 학습 데이터의 평균 Removal만 내놓는 모델**입니다.\n\n"
            "- 점수의 **기준점**: 베이스라인보다 못한 모델은 쓸모가 없습니다.\n"
            "- **착시 방지**: 고제거 실험이 많으면 무조건 '고제거'라고 찍어도 정확도가 높게 나옵니다.\n"
            "- '실험조건으로 예측 가능한가?'의 직접 근거: 변수를 쓴 모델이 평균 찍기보다 오차를 얼마나 줄였는지가 답입니다."
        )
    with st.expander("📌 정확도·재현율은 어떻게 계산하나?", expanded=False):
        st.markdown(
            "Removal은 연속값이라, **기준값 이상이면 '고제거', 미만이면 '저제거'** 로 나눠 분류 지표를 계산합니다. "
            "실제값과 모델의 예측값을 같은 기준으로 나눈 뒤 비교합니다.\n\n"
            "- **정확도** = 맞게 구분한 수 / 전체\n"
            "- **재현율** = 실제 고제거 중 고제거로 예측한 비율 (고제거를 놓치지 않는 정도)\n"
            "- **정밀도** = 고제거로 예측한 것 중 실제 고제거 비율, **F1** = 정밀도와 재현율의 조화평균"
        )

    COMBOS = {
        "A: pH+Dose+Conc+Temp": ["pH", "Dose", "Conc", "Temp"],
        "B: pH+Dose+Conc+Temp+Time": ["pH", "Dose", "Conc", "Temp", "Time"],
        "C: pH+Dose+Conc": ["pH", "Dose", "Conc"],
    }
    MODEL_NAMES = ["베이스라인(평균)", "선형회귀", "다항회귀(2차)", "랜덤포레스트", "그래디언트부스팅"]

    def make_model(name):
        if name == "베이스라인(평균)":
            return DummyRegressor(strategy="mean")
        if name == "선형회귀":
            return LinearRegression()
        if name == "다항회귀(2차)":
            return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression())
        if name == "랜덤포레스트":
            return RandomForestRegressor(n_estimators=200, random_state=42)
        return GradientBoostingRegressor(random_state=42)

    c1, c2, c3 = st.columns(3)
    use_type = c1.checkbox("Type(흡착제 종류) 포함", value=True)
    val_method = c2.radio("검증 방법", ["5-fold CV", "Type 제외 검증"])
    thr = c3.slider("고제거 기준값 (%)", 50, 99, 90, 1)
    extra = type_cols if use_type else []

    @st.cache_data(show_spinner="모델을 학습하는 중입니다...")
    def run_all(use_type):
        kf = KFold(5, shuffle=True, random_state=42)
        ex = type_cols if use_type else []
        out = {}
        for cname, vars_ in COMBOS.items():
            X = X_all[[REV[n] for n in vars_] + ex]
            for mname in MODEL_NAMES:
                cvp = cross_val_predict(make_model(mname), X, y_all, cv=kf)
                lop = np.zeros(len(df))
                for t in df["Type"].unique():
                    idx = (df["Type"] == t).values
                    lop[idx] = make_model(mname).fit(X[~idx], y_all[~idx]).predict(X[idx])
                out[(cname, mname)] = (cvp, lop)
        return out

    preds = run_all(use_type)
    k = 0 if val_method == "5-fold CV" else 1

    y_bin = (y_all >= thr).astype(int)
    pos_rate = y_bin.mean()
    maj = max(pos_rate, 1 - pos_rate)
    st.caption(f"기준값 {thr}% 이상 = 고제거. 고제거 비율 {pos_rate*100:.1f}% ({int(y_bin.sum())}/{len(y_bin)}개). "
               f"무조건 다수 클래스로 찍을 때의 정확도: {maj*100:.1f}%")

    rows = []
    for (cname, mname), pr in preds.items():
        p = pr[k]
        pb = (p >= thr).astype(int)
        rows.append({
            "조합": cname, "모델": mname,
            "R²": r2_score(y_all, p),
            "MAE": mean_absolute_error(y_all, p),
            "RMSE": np.sqrt(mean_squared_error(y_all, p)),
            "정확도": accuracy_score(y_bin, pb),
            "재현율": recall_score(y_bin, pb, zero_division=0),
            "정밀도": precision_score(y_bin, pb, zero_division=0),
            "F1": f1_score(y_bin, pb, zero_division=0),
        })
    res = pd.DataFrame(rows)
    base = res[res["모델"] == "베이스라인(평균)"].set_index("조합")
    res["MAE 개선율(%)"] = res.apply(
        lambda r: (1 - r["MAE"] / base.loc[r["조합"], "MAE"]) * 100, axis=1)

    st.subheader(f"성능 비교표 ({val_method})")
    fmt = {"R²": "{:.3f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "정확도": "{:.3f}", "재현율": "{:.3f}",
           "정밀도": "{:.3f}", "F1": "{:.3f}", "MAE 개선율(%)": "{:+.1f}"}
    st.dataframe(res.set_index(["조합", "모델"]).style.format(fmt), use_container_width=True)

    nb = res[res["모델"] != "베이스라인(평균)"]
    bi = nb["R²"].idxmax()
    br = res.loc[bi]
    bb = base.loc[br["조합"]]
    beat = nb.groupby("모델")["R²"].mean().sort_values(ascending=False)
    explain(f"- 베이스라인(평균 찍기)의 R²는 **{base['R²'].mean():.3f}**(0 근처면 정상), MAE는 약 **{base['MAE'].mean():.2f}%p**입니다. 이것이 넘어야 할 기준선입니다.\n"
            f"- 가장 좋은 조합·모델은 **{br['조합']} / {br['모델']}** (R² = {br['R²']:.3f}, MAE = {br['MAE']:.2f}%p)로, "
            f"같은 조합의 베이스라인보다 오차가 **{br['MAE 개선율(%)']:.1f}%** 줄었습니다.\n"
            f"- 세 조합 평균 R² 순위: " + " > ".join(f"{m}({v:.2f})" for m, v in beat.items()) + "\n"
            f"- 정확도는 베이스라인도 {base['정확도'].mean():.2f}가 나옵니다. 모델의 정확도가 이 값(또는 다수 클래스 비율 {maj:.2f})과 비슷하면 실질적 개선이 없는 것입니다.\n"
            f"- 재현율이 높아도 정밀도가 낮으면 '고제거'를 남발하는 모델이므로 F1도 함께 보세요.\n"
            + ("- Type 제외 검증은 새로운 흡착제에 대한 일반화 능력을 보므로, 5-fold CV보다 점수가 낮은 것이 일반적입니다."
               if k == 1 else
               "- 이 데이터는 비슷한 조건의 반복이 많아 5-fold CV가 다소 낙관적입니다. 'Type 제외 검증'으로도 꼭 확인하세요."))

    st.subheader("R² 비교")
    st.plotly_chart(px.bar(res, x="모델", y="R²", color="조합", barmode="group"), use_container_width=True)
    explain("막대가 높을수록 좋습니다. 베이스라인(0 근처)보다 낮은 막대는 평균 찍기보다 못한 모델입니다. "
            "선형회귀보다 다항회귀·트리 모델이 높다면 변수 사이에 **비선형 관계**가 있다는 뜻입니다.")

    st.subheader("정확도 · 재현율 비교")
    cl = res.melt(id_vars=["조합", "모델"], value_vars=["정확도", "재현율"], var_name="지표", value_name="값")
    st.plotly_chart(px.bar(cl, x="모델", y="값", color="조합", barmode="group", facet_row="지표"),
                    use_container_width=True)
    explain(f"- 기준값 {thr}%에서 베이스라인은 항상 한쪽 클래스로만 예측하므로, 재현율이 0 또는 1이 됩니다(고제거를 전부 놓치거나 전부 '고제거'라고 찍음).\n"
            f"- 모델의 정확도가 다수 클래스 비율({maj:.2f})보다 충분히 높고 재현율도 높아야 의미 있는 모델입니다.\n"
            f"- 기준값을 바꿔 보세요. 기준을 높이면(예: 95%) 고제거 실험이 줄어 더 까다로운 문제가 됩니다.")

    st.subheader("상세 확인: 실제값 vs 예측값 · 혼동행렬")
    s1, s2 = st.columns(2)
    sc_name = s1.selectbox("조합", list(COMBOS), key="sel_c")
    sm_name = s2.selectbox("모델", MODEL_NAMES, index=3, key="sel_m")
    p = preds[(sc_name, sm_name)][k]
    g1, g2 = st.columns(2)
    fig = px.scatter(x=y_all, y=p, color=df["Type"],
                     labels={"x": "실제 Removal", "y": "예측 Removal", "color": "Type"})
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="y=x",
                             line=dict(dash="dash", color="red")))
    fig.add_hline(y=thr, line_dash="dot", line_color="gray")
    fig.add_vline(x=thr, line_dash="dot", line_color="gray")
    g1.plotly_chart(fig, use_container_width=True)
    cm = confusion_matrix(y_bin, (p >= thr).astype(int), labels=[0, 1])
    fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     x=["예측: 저제거", "예측: 고제거"], y=["실제: 저제거", "실제: 고제거"])
    g2.plotly_chart(fig2, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    err = np.abs(y_all - p)
    explain(f"**{sc_name} / {sm_name}** ({val_method})\n\n"
            f"- 평균 오차 {err.mean():.2f}%p, 최대 오차 {err.max():.1f}%p\n"
            f"- 왼쪽: 점이 빨간 점선(y=x)에 가까울수록 정확합니다. 회색 점선 기준 오른쪽 위·왼쪽 아래 칸의 점이 올바른 분류입니다.\n"
            f"- 오른쪽 혼동행렬: 고제거를 맞힌 것 {tp}개, 놓친 것 {fn}개(재현율 {tp/max(tp+fn,1):.2f}), "
            f"저제거를 고제거로 잘못 예측한 것 {fp}개.\n"
            f"- 예측이 0~100%를 벗어난 개수: {int(((p < 0) | (p > 100)).sum())}개 (선형·다항회귀의 한계)")

