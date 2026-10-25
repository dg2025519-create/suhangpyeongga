import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="흡착률 예측 분석", layout="wide")

URL = "https://raw.githubusercontent.com/dg2025519-create/suhangpyeongga/refs/heads/main/DataSet.csv"
TARGET = "Removal"
LEAK = "Max Adsorp Capacity"   # 누수 변수: 표시만 하고 분석/모델에서는 제외
NUM_FEATURES = ["pH", "Adsorbent Dose (g/L)", "Concentration (mg/L)", "Temperature", "Time (min)"]
ANALYSIS_COLS = NUM_FEATURES + [TARGET]


@st.cache_data
def load_data():
    df = pd.read_csv(URL, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Type"] = df["Type"].str.strip()
    return df


df = load_data()
X_all = pd.concat([df[NUM_FEATURES], pd.get_dummies(df["Type"], prefix="Type").astype(int)], axis=1)
y_all = df[TARGET]
FEATURES = X_all.columns.tolist()


def get_models():
    return {
        "선형회귀": LinearRegression(),
        "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=42),
        "그래디언트부스팅": GradientBoostingRegressor(random_state=42),
    }


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
page = st.sidebar.radio("페이지 선택", ["1. 데이터 보기", "2. 상관관계 분석", "3. 예측 모델", "4. 결론 및 직접 예측"])
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
            "상관관계 분석과 예측 모델에서 제외했습니다(데이터 누수 방지).")

# ================= 2. 상관관계 =================
elif page == "2. 상관관계 분석":
    st.header("📊 상관관계 분석")
    st.caption("누수 변수(Max Adsorp Capacity)는 제외하고, 실험조건 5개와 Removal만 분석합니다.")
    tabs = st.tabs(["히트맵", "타깃과의 상관", "조건별 변화", "산점도", "Type별 비교", "분포"])

    # ---- 히트맵
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

    # ---- 타깃과의 상관
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

    # ---- 조건별 변화
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
            "Concentration (mg/L)": "농도가 높아질수록 흡착제의 자리가 포화되어 Removal(%)이 떨어지는 것이 일반적입니다.",
            "Temperature": "온도가 오를 때 Removal이 오르면 흡열 과정(온도가 높을수록 유리)임을 시사합니다.",
            "Time (min)": "초반에 빠르게 오르다 일정 시간 이후 평평해지면 흡착 평형에 도달했다는 뜻입니다.",
        }[var]
        explain("\n".join(lines) + f"\n\n{hint}\n\n세 종류 모두 비슷한 방향으로 움직이면 Type과 무관한 공통 경향이고, "
                "선이 서로 많이 떨어져 있으면 Type이 Removal 수준 자체에 영향을 준다는 뜻입니다.")

    # ---- 산점도
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

    # ---- Type별 비교
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

    # ---- 분포
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

# ================= 3. 모델 =================
elif page == "3. 예측 모델":
    st.header("🤖 머신러닝 예측 모델")
    st.markdown(f"**입력 변수:** {', '.join(FEATURES)}  \n**타깃:** {TARGET}")

    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

    kf = KFold(5, shuffle=True, random_state=42)
    groups = df["Type"]
    rows, preds = [], {}
    for name, m in get_models().items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name] = p
        cvp = cross_val_predict(get_models()[name], X_all, y_all, cv=kf)
        lo_pred = pd.Series(index=df.index, dtype=float)
        for t in groups.unique():
            idx = groups == t
            mm = get_models()[name].fit(X_all[~idx], y_all[~idx])
            lo_pred[idx] = mm.predict(X_all[idx])
        rows.append({"모델": name,
                     "R² (테스트)": r2_score(y_te, p),
                     "MAE": mean_absolute_error(y_te, p),
                     "RMSE": np.sqrt(mean_squared_error(y_te, p)),
                     "R² (5-fold CV)": r2_score(y_all, cvp),
                     "R² (Type 제외 검증)": r2_score(y_all, lo_pred)})
    res = pd.DataFrame(rows).set_index("모델")
    st.subheader("모델 성능 비교")
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    long = res.reset_index().melt(id_vars="모델", value_vars=["R² (테스트)", "R² (5-fold CV)", "R² (Type 제외 검증)"],
                                  var_name="검증방법", value_name="R²")
    st.plotly_chart(px.bar(long, x="모델", y="R²", color="검증방법", barmode="group"), use_container_width=True)

    best = res["R² (5-fold CV)"].idxmax()
    cv_best = res.loc[best, "R² (5-fold CV)"]
    ty_best = res.loc[best, "R² (Type 제외 검증)"]
    lin_cv = res.loc["선형회귀", "R² (5-fold CV)"]
    explain(f"- R²는 1에 가까울수록 잘 맞춘다는 뜻이고, 0 이하면 평균값으로 찍는 것보다 못하다는 뜻입니다. MAE·RMSE는 평균적인 오차 크기(%p)입니다.\n"
            f"- 5-fold CV 기준 최고 모델은 **{best}** (R² = {cv_best:.3f}, 테스트 MAE = {res.loc[best,'MAE']:.2f}%p)입니다.\n"
            f"- 선형회귀의 CV R²는 {lin_cv:.3f}로, 트리 기반 모델과 비교해 "
            + ("차이가 작아 관계가 대체로 직선에 가깝습니다.\n" if cv_best - lin_cv < 0.1 else
               "차이가 커서 변수 사이의 **비선형·상호작용 관계**가 중요함을 보여줍니다.\n")
            + f"- **Type 제외 검증** R²는 {ty_best:.3f}로 CV({cv_best:.3f})보다 "
            + ("낮습니다. 즉 이미 본 흡착제의 새 조건은 잘 예측하지만 새로운 흡착제는 그보다 덜 정확합니다.\n" if ty_best < cv_best - 0.02 else
               "비슷합니다. 흡착제 종류가 달라도 비슷한 방식으로 예측되어 일반화가 양호합니다.\n")
            + "- 테스트 R²는 표본이 적어(분할에 따라) 흔들리므로 CV 값을 더 신뢰하세요.")

    st.subheader(f"실제값 vs 예측값 ({best}, 5-fold CV 예측)")
    cvp = cross_val_predict(get_models()[best], X_all, y_all, cv=kf)
    fig = px.scatter(x=y_all, y=cvp, color=df["Type"], labels={"x": "실제 Removal", "y": "예측 Removal", "color": "Type"})
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="y=x", line=dict(dash="dash", color="red")))
    st.plotly_chart(fig, use_container_width=True)
    err = np.abs(y_all - cvp)
    low = y_all < 70
    explain(f"- 점이 빨간 점선(y=x)에 가까울수록 정확한 예측입니다.\n"
            f"- 평균 절대 오차는 **{err.mean():.2f}%p**, 최대 오차는 {err.max():.1f}%p입니다.\n"
            f"- Removal이 70% 미만인 구간({int(low.sum())}개)의 평균 오차는 {err[low].mean():.2f}%p, "
            f"70% 이상 구간은 {err[~low].mean():.2f}%p입니다. "
            + ("낮은 Removal 구간에서 오차가 더 큰 것은 그 영역의 데이터가 적기 때문입니다." if err[low].mean() > err[~low].mean()
               else "낮은 구간에서도 오차가 크지 않아 전 구간에서 고르게 예측합니다."))

    st.subheader("변수 중요도 (Permutation Importance)")
    final = get_models()[best].fit(X_tr, y_tr)
    pi = permutation_importance(final, X_te, y_te, n_repeats=30, random_state=42)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "중요도(R² 감소량)", "y": "변수"}),
                    use_container_width=True)
    order = imp.sort_values(ascending=False)
    explain(f"해당 변수를 무작위로 섞었을 때 예측력(R²)이 얼마나 떨어지는지를 나타냅니다. 클수록 예측에 중요한 변수입니다.\n\n"
            f"- 1위 **{order.index[0]}** ({order.iloc[0]:.3f}), 2위 **{order.index[1]}** ({order.iloc[1]:.3f}), "
            f"3위 **{order.index[2]}** ({order.iloc[2]:.3f})\n"
            f"- 값이 0에 가깝거나 음수인 변수는 이 모델에서 예측에 거의 기여하지 않습니다.\n"
            f"- 중요도는 선택한 테스트 비율에 따라 바뀔 수 있으므로 순위의 큰 흐름만 참고하세요.")

# ================= 4. 결론 =================
else:
    st.header("🎯 결론 및 직접 예측")
    kf = KFold(5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    cvp = cross_val_predict(model, X_all, y_all, cv=kf)
    r2 = r2_score(y_all, cvp)
    mae = mean_absolute_error(y_all, cvp)

    c1, c2 = st.columns(2)
    c1.metric("랜덤포레스트 5-fold CV R²", f"{r2:.3f}")
    c2.metric("평균 절대 오차(MAE)", f"{mae:.2f} %p")

    if r2 >= 0.8:
        st.markdown("✅ **예측 가능**: 실험조건(pH, 흡착제 양, 농도, 온도, 시간, 종류)만으로 Removal을 높은 정확도로 예측할 수 있습니다.")
    elif r2 >= 0.5:
        st.markdown("⚠️ **부분적으로 예측 가능**: 경향은 잡지만 오차가 있습니다.")
    else:
        st.markdown("❌ **예측 어려움**: 현재 변수만으로는 설명력이 부족합니다.")
    st.caption("⚠️ 데이터가 약 100행이고 한 변수씩 바꾼 실험이라 비슷한 조건이 많아 교차검증 점수가 다소 낙관적일 수 있습니다. "
               "3페이지의 'Type 제외 검증'도 함께 참고하세요.")

    st.subheader("실험조건을 입력해 Removal 예측")
    final = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_all, y_all)
    c = st.columns(3)
    typ = c[0].selectbox("Type", sorted(df["Type"].unique()))
    ph = c[1].slider("pH", 3.0, 10.0, 5.0, 0.5)
    dose = c[2].slider("Adsorbent Dose (g/L)", 0.4, 6.0, 2.0, 0.1)
    conc = c[0].slider("Concentration (mg/L)", 10, 500, 100, 10)
    temp = c[1].slider("Temperature (℃)", 25, 55, 25, 5)
    tm = c[2].slider("Time (min)", 30, 600, 300, 30)

    row = {"pH": ph, "Adsorbent Dose (g/L)": dose, "Concentration (mg/L)": conc,
           "Temperature": temp, "Time (min)": tm}
    for col in FEATURES:
        if col.startswith("Type_"):
            row[col] = int(col == f"Type_{typ}")
    pred = final.predict(pd.DataFrame([row])[FEATURES])[0]
    st.metric("예측 Removal (%)", f"{pred:.1f}")
    explain(f"입력한 조건(**{typ}**, pH {ph}, 용량 {dose} g/L, 농도 {conc} mg/L, {temp}℃, {tm}분)에서 모델이 예측한 Removal은 "
            f"**{pred:.1f}%**이고, 교차검증 평균 오차({mae:.1f}%p)를 감안하면 대략 {max(pred-mae,0):.1f}~{min(pred+mae,100):.1f}% 범위로 볼 수 있습니다.\n\n"
            "랜덤포레스트는 학습 데이터 범위 밖을 외삽하지 못하고, 실험에서 측정하지 않은 조합(예: pH 10 + 농도 500)은 "
            "가까운 다른 조건의 값으로 추정하므로 신뢰도가 낮습니다.")
