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
LEAK = "Max Adsorp Capacity"   # 누수 변수 (Removal로 계산되는 값)
NUM_FEATURES = ["pH", "Adsorbent Dose (g/L)", "Concentration (mg/L)", "Temperature", "Time (min)"]


@st.cache_data
def load_data():
    df = pd.read_csv(URL, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Type"] = df["Type"].str.strip()
    return df


df = load_data()

# 모델 입력 구성: 숫자 변수 + Type 원-핫
X_all = pd.concat([df[NUM_FEATURES], pd.get_dummies(df["Type"], prefix="Type").astype(int)], axis=1)
y_all = df[TARGET]
FEATURES = X_all.columns.tolist()


def get_models():
    return {
        "선형회귀": LinearRegression(),
        "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=42),
        "그래디언트부스팅": GradientBoostingRegressor(random_state=42),
    }


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

    sel = st.multiselect("Type 필터", df["Type"].unique().tolist(), default=df["Type"].unique().tolist())
    st.dataframe(df[df["Type"].isin(sel)], use_container_width=True, height=400)

    st.subheader("기초 통계량")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Type별 평균")
    st.dataframe(df.groupby("Type")[[TARGET, LEAK]].agg(["mean", "std", "count"]), use_container_width=True)

    st.info("ℹ️ `Max Adsorp Capacity`는 Removal·농도·흡착제 양으로 계산되는 값이라 "
            "예측 입력 변수에서 제외했습니다(데이터 누수 방지).")

# ================= 2. 상관관계 =================
elif page == "2. 상관관계 분석":
    st.header("📊 상관관계 분석")
    tabs = st.tabs(["히트맵", "타깃과의 상관", "조건별 변화(선 그래프)", "산점도", "Type별 비교", "분포"])

    num_all = NUM_FEATURES + [TARGET, LEAK]

    with tabs[0]:
        method = st.radio("상관계수 종류", ["pearson", "spearman"], horizontal=True)
        corr = df[num_all].corr(method=method)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)

    with tabs[1]:
        corr_t = df[NUM_FEATURES + [LEAK]].corrwith(df[TARGET]).sort_values()
        fig = px.bar(x=corr_t.values, y=corr_t.index, orientation="h", color=corr_t.values,
                     color_continuous_scale="RdBu_r", range_color=[-1, 1],
                     labels={"x": "상관계수", "y": "변수"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("전체 데이터를 섞어 계산한 값이라, 한 변수만 바꾼 실험들이 섞여 있어 단순 상관은 약하게 나올 수 있습니다. "
                   "아래 '조건별 변화' 탭에서 변수별 효과를 확인하세요.")

    with tabs[2]:
        st.markdown("기준 조건(pH 5, 용량 2 g/L, 농도 100 mg/L, 25℃, 300분) 중 **한 변수만 바꾼 실험**을 추려서 보여줍니다.")
        base = {"pH": 5, "Adsorbent Dose (g/L)": 2, "Concentration (mg/L)": 100,
                "Temperature": 25, "Time (min)": 300}
        var = st.selectbox("변화시킨 변수", NUM_FEATURES, index=2)
        others = [c for c in NUM_FEATURES if c != var]
        mask = np.ones(len(df), dtype=bool)
        for c in others:
            mask &= (df[c] == base[c])
        sub = df[mask].sort_values(var)
        ycol = st.radio("Y축", [TARGET, LEAK], horizontal=True)
        fig = px.line(sub, x=var, y=ycol, color="Type", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("pH/용량 실험은 농도 10 mg/L 조건이라 기준(농도 100)과 달라 일부 변수에서는 점이 적을 수 있습니다.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X축", NUM_FEATURES)
        yv = c2.selectbox("Y축", [TARGET, LEAK, *NUM_FEATURES], index=0)
        fig = px.scatter(df, x=x, y=yv, color="Type", trendline="ols" if x != yv else None)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("산점도 행렬")
        st.plotly_chart(px.scatter_matrix(df, dimensions=NUM_FEATURES + [TARGET], color="Type", height=800),
                        use_container_width=True)

    with tabs[4]:
        v = st.selectbox("비교할 값", [TARGET, LEAK])
        st.plotly_chart(px.box(df, x="Type", y=v, color="Type", points="all"), use_container_width=True)
        g = st.selectbox("그룹 변수(막대)", NUM_FEATURES, index=0)
        agg = df.groupby(["Type", g])[v].mean().reset_index()
        st.plotly_chart(px.bar(agg, x=g, y=v, color="Type", barmode="group"), use_container_width=True)

    with tabs[5]:
        v = st.selectbox("변수 선택", NUM_FEATURES + [TARGET, LEAK], key="hist")
        st.plotly_chart(px.histogram(df, x=v, color="Type", nbins=30, marginal="box", barmode="overlay"),
                        use_container_width=True)

# ================= 3. 모델 =================
elif page == "3. 예측 모델":
    st.header("🤖 머신러닝 예측 모델")
    st.markdown(f"**입력 변수:** {', '.join(FEATURES)}  \n**타깃:** {TARGET}")

    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

    kf = KFold(5, shuffle=True, random_state=42)
    groups = df["Type"]
    rows, preds, fitted = [], {}, {}
    for name, m in get_models().items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name], fitted[name] = p, m
        cvp = cross_val_predict(get_models()[name], X_all, y_all, cv=kf)

        # Type 하나를 통째로 제외하고 예측 (새로운 흡착제에 대한 일반화)
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
    st.caption("Type 제외 검증: 한 종류의 껍질 데이터를 전부 빼고 학습한 뒤 그 종류를 예측 → 새로운 흡착제에 대한 일반화 능력.")

    long = res.reset_index().melt(id_vars="모델", value_vars=["R² (테스트)", "R² (5-fold CV)", "R² (Type 제외 검증)"],
                                  var_name="검증방법", value_name="R²")
    st.plotly_chart(px.bar(long, x="모델", y="R²", color="검증방법", barmode="group"), use_container_width=True)

    best = res["R² (5-fold CV)"].idxmax()
    st.success(f"교차검증 기준 최고 모델: **{best}** (R² = {res.loc[best, 'R² (5-fold CV)']:.3f})")

    st.subheader(f"실제값 vs 예측값 ({best}, 5-fold CV 예측)")
    cvp = cross_val_predict(get_models()[best], X_all, y_all, cv=kf)
    fig = px.scatter(x=y_all, y=cvp, color=df["Type"], labels={"x": "실제 Removal", "y": "예측 Removal", "color": "Type"})
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="y=x", line=dict(dash="dash", color="red")))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("변수 중요도 (Permutation Importance)")
    final = get_models()[best].fit(X_tr, y_tr)
    pi = permutation_importance(final, X_te, y_te, n_repeats=30, random_state=42)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "중요도(R² 감소량)", "y": "변수"}),
                    use_container_width=True)

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
    st.caption("⚠️ 데이터가 약 100행이고 한 변수씩 바꾼 실험이라 비슷한 조건이 많습니다. "
               "교차검증 점수는 다소 낙관적일 수 있으니 3페이지의 'Type 제외 검증'도 함께 참고하세요.")

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
    st.caption("학습 데이터 범위 밖의 값은 신뢰하기 어렵습니다(랜덤포레스트는 범위 밖을 외삽하지 못합니다).")
