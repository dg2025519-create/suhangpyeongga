import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
                                        "3. 최적 변수 조합 탐색", "4. 조합별 모델 비교",
                                        "5. 농도 log 변환 모델"])
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

# ================= 4. 조합별 모델 비교 =================
elif page == "4. 조합별 모델 비교":
    st.header("⚖️ 조합별 선형회귀 모델 비교")
    st.markdown("세 가지 입력 변수 조합으로 각각 선형회귀 모델을 만들어 성능을 비교합니다.")

    COMBOS = {
        "A: pH+Dose+Conc+Temp": ["pH", "Dose", "Conc", "Temp"],
        "B: pH+Dose+Conc+Temp+Time": ["pH", "Dose", "Conc", "Temp", "Time"],
        "C: pH+Dose+Conc": ["pH", "Dose", "Conc"],
    }
    use_type = st.checkbox("Type(흡착제 종류) 포함", value=True)
    extra = type_cols if use_type else []

    def cols_of(names):
        return [REV[n] for n in names] + extra

    kf = KFold(5, shuffle=True, random_state=42)
    rows, cvps, models = [], {}, {}
    for name, vars_ in COMBOS.items():
        cols = cols_of(vars_)
        X = X_all[cols]
        cvp = cross_val_predict(LinearRegression(), X, y_all, cv=kf)
        lo = pd.Series(index=df.index, dtype=float)
        for t in df["Type"].unique():
            idx = df["Type"] == t
            lo[idx] = LinearRegression().fit(X[~idx], y_all[~idx]).predict(X[idx])
        m = LinearRegression().fit(X, y_all)
        cvps[name], models[name] = cvp, m
        rows.append({"조합": name, "변수 수": len(cols),
                     "R² (학습)": m.score(X, y_all),
                     "R² (5-fold CV)": r2_score(y_all, cvp),
                     "R² (Type 제외 검증)": r2_score(y_all, lo),
                     "MAE (CV)": mean_absolute_error(y_all, cvp),
                     "RMSE (CV)": np.sqrt(mean_squared_error(y_all, cvp))})
    res = pd.DataFrame(rows).set_index("조합")

    st.subheader("성능 비교표")
    st.dataframe(res.style.format({"변수 수": "{:.0f}", "R² (학습)": "{:.3f}", "R² (5-fold CV)": "{:.3f}",
                                   "R² (Type 제외 검증)": "{:.3f}", "MAE (CV)": "{:.2f}", "RMSE (CV)": "{:.2f}"})
                 .highlight_max(subset=["R² (5-fold CV)", "R² (Type 제외 검증)"], color="#c6efce")
                 .highlight_min(subset=["MAE (CV)", "RMSE (CV)"], color="#c6efce"),
                 use_container_width=True)
    bname = res["R² (5-fold CV)"].idxmax()
    wname = res["R² (5-fold CV)"].idxmin()
    gap = res["R² (학습)"] - res["R² (5-fold CV)"]
    explain(f"- 5-fold CV 기준 최고 조합은 **{bname}** (R² = {res.loc[bname,'R² (5-fold CV)']:.3f}, MAE = {res.loc[bname,'MAE (CV)']:.2f}%p), "
            f"최저는 **{wname}** (R² = {res.loc[wname,'R² (5-fold CV)']:.3f})입니다. 차이는 {res['R² (5-fold CV)'].max()-res['R² (5-fold CV)'].min():.3f}입니다.\n"
            f"- 학습 R²와 CV R²의 차이(과적합 정도)는 최대 {gap.max():.3f}입니다. 선형회귀는 단순해서 보통 차이가 작습니다.\n"
            f"- Type 제외 검증 R²가 CV보다 낮으면 새로운 흡착제에는 덜 정확하다는 뜻입니다.\n"
            f"- R²가 낮다면 농도·용량처럼 **곡선(비선형)** 관계를 직선으로 근사하기 때문일 수 있습니다.")

    long = res.reset_index().melt(id_vars="조합", value_vars=["R² (학습)", "R² (5-fold CV)", "R² (Type 제외 검증)"],
                                  var_name="검증방법", value_name="R²")
    st.plotly_chart(px.bar(long, x="조합", y="R²", color="검증방법", barmode="group"), use_container_width=True)
    dtime = res.loc["B: pH+Dose+Conc+Temp+Time", "R² (5-fold CV)"] - res.loc["A: pH+Dose+Conc+Temp", "R² (5-fold CV)"]
    dtemp = res.loc["A: pH+Dose+Conc+Temp", "R² (5-fold CV)"] - res.loc["C: pH+Dose+Conc", "R² (5-fold CV)"]
    explain(f"- A에 Time을 추가한 B의 CV R² 변화: **{dtime:+.3f}** → "
            + ("Time은 성능 향상에 기여합니다.\n" if dtime > 0.01 else "Time을 넣어도 거의 달라지지 않아 이 데이터에서는 불필요한 변수입니다.\n")
            + f"- C에 Temp를 추가한 A의 CV R² 변화: **{dtemp:+.3f}** → "
            + ("Temp는 성능 향상에 기여합니다." if dtemp > 0.01 else "Temp를 넣어도 거의 달라지지 않습니다."))

    st.subheader("실제값 vs 예측값 (5-fold CV 예측)")
    cols3 = st.columns(3)
    for i, name in enumerate(COMBOS):
        fig = px.scatter(x=y_all, y=cvps[name], color=df["Type"],
                         labels={"x": "실제 Removal", "y": "예측 Removal", "color": "Type"}, title=name)
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="y=x",
                                 line=dict(dash="dash", color="red")))
        fig.update_layout(showlegend=(i == 2), title_font_size=13)
        cols3[i].plotly_chart(fig, use_container_width=True)
    errs = {n: np.abs(y_all - cvps[n]) for n in COMBOS}
    explain("점이 빨간 점선(y=x)에 가까울수록 정확합니다.\n\n"
            + "\n".join(f"- **{n}**: 평균 오차 {e.mean():.2f}%p, 최대 오차 {e.max():.1f}%p" for n, e in errs.items())
            + "\n- 예측값이 100%를 넘거나 0% 아래로 나오면 직선 모델의 한계입니다(Removal은 0~100% 범위).\n"
              "- 낮은 Removal 구간에서 점이 선에서 멀다면 고농도 조건을 직선으로 잘 설명하지 못한다는 뜻입니다.")

    st.subheader("회귀계수 (표준화)")
    coef_rows = []
    for name, vars_ in COMBOS.items():
        cols = cols_of(vars_)
        Xs = (X_all[cols] - X_all[cols].mean()) / X_all[cols].std()
        ms = LinearRegression().fit(Xs, y_all)
        for c, b in zip(cols, ms.coef_):
            coef_rows.append({"조합": name, "변수": SHORT.get(c, c), "표준화 계수": b})
    coef_df = pd.DataFrame(coef_rows)
    st.plotly_chart(px.bar(coef_df, x="변수", y="표준화 계수", color="조합", barmode="group"),
                    use_container_width=True)
    bcoef = coef_df[coef_df["조합"] == bname].copy()
    bcoef = bcoef[~bcoef["변수"].str.startswith("Type_")]
    bcoef["abs"] = bcoef["표준화 계수"].abs()
    bcoef = bcoef.sort_values("abs", ascending=False)
    explain(f"변수를 같은 스케일(표준편차 1)로 맞춘 뒤의 계수라 **막대 길이로 영향력을 비교**할 수 있습니다. 양수면 값이 클수록 Removal 증가, 음수면 감소입니다.\n\n"
            f"- 최고 조합 **{bname}** 에서 영향이 가장 큰 변수는 **{bcoef.iloc[0]['변수']}** ({bcoef.iloc[0]['표준화 계수']:+.2f}), "
            f"그다음은 **{bcoef.iloc[1]['변수']}** ({bcoef.iloc[1]['표준화 계수']:+.2f})입니다.\n"
            f"- 조합이 바뀌어도 계수의 부호와 크기가 비슷하면 모델이 안정적입니다.")

    st.subheader("원 단위 회귀식 (최고 조합)")
    bm = models[bname]
    bcols = cols_of(COMBOS[bname])
    eq = pd.DataFrame({"변수": [SHORT.get(c, c) for c in bcols], "계수": bm.coef_})
    eq.loc[len(eq)] = ["절편", bm.intercept_]
    st.dataframe(eq.style.format({"계수": "{:.4f}"}), use_container_width=True, hide_index=True)
    explain("변수가 1단위 늘 때 Removal(%)이 얼마나 변하는지를 보여줍니다(다른 변수는 고정). "
            "예: Conc 계수가 −0.1이면 오염물질 농도가 10 mg/L 늘 때 Removal이 약 1%p 감소합니다.")

    st.subheader("직접 예측해 보기")
    pc = st.columns(4)
    pname = pc[0].selectbox("사용할 조합", list(COMBOS))
    typ = pc[1].selectbox("Type", sorted(df["Type"].unique()), disabled=not use_type)
    vals = {}
    vars_ = COMBOS[pname]
    ranges = {"pH": (3.0, 10.0, 5.0, 0.5), "Dose": (0.4, 6.0, 2.0, 0.1), "Conc": (10, 500, 100, 10),
              "Temp": (25, 55, 25, 5), "Time": (30, 600, 300, 30)}
    sc = st.columns(len(vars_))
    for i, n in enumerate(vars_):
        lo_, hi_, d_, st_ = ranges[n]
        vals[REV[n]] = sc[i].slider(n, lo_, hi_, d_, st_)
    row = dict(vals)
    for c in extra:
        row[c] = int(c == f"Type_{typ}")
    pcols = cols_of(vars_)
    pred = models[pname].predict(pd.DataFrame([row])[pcols])[0]
    pm = res.loc[pname, "MAE (CV)"]
    st.metric(f"예측 Removal ({pname})", f"{pred:.1f}%")
    explain(f"선택한 조건에서 **{pname}** 모델의 예측값은 **{pred:.1f}%**이고, CV 평균 오차({pm:.1f}%p)를 고려하면 "
            f"약 {max(pred-pm,0):.1f}~{min(pred+pm,100):.1f}% 범위로 볼 수 있습니다.\n\n"
            + ("⚠️ 예측값이 0~100%를 벗어났습니다. 선형회귀가 범위 밖 값을 만들 수 있으니 실제로는 0 또는 100%로 해석하세요."
               if pred < 0 or pred > 100 else
               "선형회귀는 직선 관계를 가정하므로, 실험하지 않은 조합이나 범위 밖 값의 예측은 신뢰도가 낮습니다."))

# ================= 5. 농도 log 변환 모델 =================
else:
    st.header("📐 농도 log 변환 모델 (추가 실험)")
    st.markdown("원래 계획에는 없던 **추가 분석**입니다. 농도(Conc)는 Removal과 곡선 관계라서, "
                "`log(Conc)`로 바꾸면 선형회귀가 더 잘 맞는지 4페이지의 세 조합으로 비교합니다.")

    COMBOS = {
        "A: pH+Dose+Conc+Temp": ["pH", "Dose", "Conc", "Temp"],
        "B: pH+Dose+Conc+Temp+Time": ["pH", "Dose", "Conc", "Temp", "Time"],
        "C: pH+Dose+Conc": ["pH", "Dose", "Conc"],
    }
    use_type = st.checkbox("Type(흡착제 종류) 포함", value=True, key="log_type")
    extra = type_cols if use_type else []

    # log 변환 데이터 (log1p 아님: 농도는 10 이상이라 자연로그 사용)
    X_log = X_all.copy()
    X_log["Concentration (mg/L)"] = np.log(X_log["Concentration (mg/L)"])

    st.subheader("농도 변환 전/후 비교")
    cc = st.columns(2)
    d1 = df.assign(구분="원래 농도")
    cc[0].plotly_chart(px.scatter(df, x="Concentration (mg/L)", y=TARGET, color="Type",
                                  title="원래 Conc vs Removal"), use_container_width=True)
    cc[1].plotly_chart(px.scatter(x=X_log["Concentration (mg/L)"], y=y_all, color=df["Type"],
                                  labels={"x": "log(Conc)", "y": TARGET, "color": "Type"},
                                  title="log(Conc) vs Removal"), use_container_width=True)
    r_raw = df["Concentration (mg/L)"].corr(y_all)
    r_log = X_log["Concentration (mg/L)"].corr(y_all)
    explain(f"- Removal과의 상관계수: 원래 Conc **{r_raw:.3f}**, log(Conc) **{r_log:.3f}**\n"
            + ("- log 변환 후 상관의 크기가 커져 관계가 더 직선에 가까워졌습니다.\n" if abs(r_log) > abs(r_raw)
               else "- log 변환으로 직선성이 좋아지지 않았습니다.\n")
            + "- 농도가 10~500으로 크게 퍼져 있어 작은 값 구간이 몰려 보이는데, log를 취하면 간격이 고르게 펴집니다.")

    kf = KFold(5, shuffle=True, random_state=42)

    def evaluate(Xdf, cols):
        X = Xdf[cols]
        cvp = cross_val_predict(LinearRegression(), X, y_all, cv=kf)
        lo = pd.Series(index=df.index, dtype=float)
        for t in df["Type"].unique():
            idx = df["Type"] == t
            lo[idx] = LinearRegression().fit(X[~idx], y_all[~idx]).predict(X[idx])
        m = LinearRegression().fit(X, y_all)
        return m, cvp, {"R² (학습)": m.score(X, y_all),
                        "R² (5-fold CV)": r2_score(y_all, cvp),
                        "R² (Type 제외 검증)": r2_score(y_all, lo),
                        "MAE (CV)": mean_absolute_error(y_all, cvp),
                        "RMSE (CV)": np.sqrt(mean_squared_error(y_all, cvp))}

    rows, cvp_log, models_log, cvp_raw = [], {}, {}, {}
    for name, vars_ in COMBOS.items():
        cols = [REV[n] for n in vars_] + extra
        _, cr, sr = evaluate(X_all, cols)
        ml, cl, sl = evaluate(X_log, cols)
        cvp_raw[name], cvp_log[name], models_log[name] = cr, cl, ml
        rows.append({"조합": name, "변환": "원래 Conc", **sr})
        rows.append({"조합": name, "변환": "log(Conc)", **sl})
    res = pd.DataFrame(rows)

    st.subheader("성능 비교표")
    fmt = {c: "{:.3f}" for c in ["R² (학습)", "R² (5-fold CV)", "R² (Type 제외 검증)"]}
    fmt.update({"MAE (CV)": "{:.2f}", "RMSE (CV)": "{:.2f}"})
    st.dataframe(res.set_index(["조합", "변환"]).style.format(fmt), use_container_width=True)

    piv = res.pivot(index="조합", columns="변환", values="R² (5-fold CV)")
    piv["개선폭"] = piv["log(Conc)"] - piv["원래 Conc"]
    bn = piv["log(Conc)"].idxmax()
    explain("\n".join(f"- **{n}**: CV R² {r['원래 Conc']:.3f} → {r['log(Conc)']:.3f} ({r['개선폭']:+.3f})"
                      for n, r in piv.iterrows())
            + f"\n\n- log 변환 후 가장 좋은 조합은 **{bn}** (CV R² = {piv.loc[bn,'log(Conc)']:.3f})입니다.\n"
            + ("- 모든 조합에서 log 변환이 성능을 높였습니다." if (piv["개선폭"] > 0).all()
               else "- 일부 조합에서는 log 변환의 효과가 없거나 오히려 낮아졌습니다."))

    long = res.melt(id_vars=["조합", "변환"], value_vars=["R² (5-fold CV)", "R² (Type 제외 검증)"],
                    var_name="검증방법", value_name="R²")
    st.plotly_chart(px.bar(long, x="조합", y="R²", color="변환", barmode="group", facet_col="검증방법"),
                    use_container_width=True)
    explain("파란/빨간 막대 쌍의 높이 차이가 log 변환의 효과입니다. "
            "Type 제외 검증에서도 개선되면 새로운 흡착제에도 통하는 개선입니다.")

    st.subheader(f"실제값 vs 예측값 ({bn}, 5-fold CV)")
    cc = st.columns(2)
    for i, (title, p) in enumerate([("원래 Conc", cvp_raw[bn]), ("log(Conc)", cvp_log[bn])]):
        fig = px.scatter(x=y_all, y=p, color=df["Type"],
                         labels={"x": "실제 Removal", "y": "예측 Removal", "color": "Type"}, title=title)
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="y=x",
                                 line=dict(dash="dash", color="red")))
        cc[i].plotly_chart(fig, use_container_width=True)
    e_raw, e_log = np.abs(y_all - cvp_raw[bn]), np.abs(y_all - cvp_log[bn])
    explain(f"- 평균 오차: 원래 Conc **{e_raw.mean():.2f}%p**, log(Conc) **{e_log.mean():.2f}%p**\n"
            f"- 100%를 넘는 예측 수: 원래 {int((cvp_raw[bn] > 100).sum())}개, log {int((cvp_log[bn] > 100).sum())}개\n"
            "- 점이 점선에 더 가깝게 모이면 log 변환이 곡선 관계를 잘 반영한 것입니다.")

    st.subheader("원 단위 회귀식 (log 변환, 최고 조합)")
    bm = models_log[bn]
    bcols = [REV[n] for n in COMBOS[bn]] + extra
    names = [("log(Conc)" if c == "Concentration (mg/L)" else SHORT.get(c, c)) for c in bcols]
    eq = pd.DataFrame({"변수": names + ["절편"], "계수": list(bm.coef_) + [bm.intercept_]})
    st.dataframe(eq.style.format({"계수": "{:.4f}"}), use_container_width=True, hide_index=True)
    ck = eq.loc[eq["변수"] == "log(Conc)", "계수"].iloc[0]
    explain(f"log(Conc) 계수는 **{ck:.2f}**입니다. 농도가 2배가 되면 log 값이 약 0.69 늘어 "
            f"Removal이 약 **{ck*np.log(2):+.1f}%p** 변한다는 뜻입니다(다른 변수 고정).")

    st.subheader("직접 예측해 보기 (log 모델)")
    pc = st.columns(2)
    pname = pc[0].selectbox("사용할 조합", list(COMBOS), key="log_pick")
    typ = pc[1].selectbox("Type", sorted(df["Type"].unique()), disabled=not use_type, key="log_typ")
    ranges = {"pH": (3.0, 10.0, 5.0, 0.5), "Dose": (0.4, 6.0, 2.0, 0.1), "Conc": (10, 500, 100, 10),
              "Temp": (25, 55, 25, 5), "Time": (30, 600, 300, 30)}
    vars_ = COMBOS[pname]
    sc = st.columns(len(vars_))
    row = {}
    for i, n in enumerate(vars_):
        a, b, d, s_ = ranges[n]
        row[REV[n]] = sc[i].slider(n, a, b, d, s_, key=f"log_{n}")
    row["Concentration (mg/L)"] = np.log(row["Concentration (mg/L)"])
    for c in extra:
        row[c] = int(c == f"Type_{typ}")
    pcols = [REV[n] for n in vars_] + extra
    pred = models_log[pname].predict(pd.DataFrame([row])[pcols])[0]
    st.metric(f"예측 Removal ({pname}, log 모델)", f"{pred:.1f}%")
    explain("농도는 입력값을 자동으로 log 변환해서 계산합니다. "
            + ("⚠️ 예측값이 0~100%를 벗어났으니 실제로는 0 또는 100%로 해석하세요." if pred < 0 or pred > 100
               else "학습 범위 안의 조건에서만 신뢰할 수 있습니다."))
