# -*- coding: utf-8 -*-
"""
역행렬 시뮬레이터 (수학-정보 융합 프로젝트 기초)
당곡고등학교 2026 수학정보 융합캠프
실행:  streamlit run app.py
"""
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="역행렬 시뮬레이터", page_icon="🔢", layout="wide")

# ---------------------------------------------------------------- 공통 함수
FIG_BG = "#ffffff"

def det2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

def inv2(A):
    """공식으로 직접 구현한 2x2 역행렬 (교과서 방식)"""
    d = det2(A)
    if abs(d) < 1e-12:
        return None
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / d

def matrix_input(label, default, key):
    """2x2 행렬 입력 위젯"""
    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    a = c1.number_input("a", value=float(default[0][0]), step=0.5, key=key + "a")
    b = c2.number_input("b", value=float(default[0][1]), step=0.5, key=key + "b")
    c = c1.number_input("c", value=float(default[1][0]), step=0.5, key=key + "c")
    d = c2.number_input("d", value=float(default[1][1]), step=0.5, key=key + "d")
    return np.array([[a, b], [c, d]], dtype=float)

def show_matrix(A, name="A"):
    r = lambda v: f"{v:.3g}"
    st.latex(
        rf"{name}=\begin{{pmatrix}} {r(A[0,0])} & {r(A[0,1])} \\ "
        rf"{r(A[1,0])} & {r(A[1,1])} \end{{pmatrix}}"
    )

# 'F' 모양 도형 (글자의 방향/뒤집힘을 눈으로 확인하기 좋음)
SHAPES = {
    "글자 F": np.array([
        [0.0, 0.0], [0.30, 0.0], [0.30, 0.40], [0.68, 0.40], [0.68, 0.58],
        [0.30, 0.58], [0.30, 0.82], [0.90, 0.82], [0.90, 1.00], [0.0, 1.00],
    ]),
    "단위정사각형": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "삼각형": np.array([[0, 0], [1, 0], [0, 1]]),
    "집": np.array([[0, 0], [1, 0], [1, 0.6], [0.5, 1.0], [0, 0.6]]),
}

def plot_shapes(P, Q, lim=3.0, label1="원래 도형", label2="변환된 도형"):
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=FIG_BG)
    def draw(pts, color, alpha, lab):
        poly = np.vstack([pts, pts[0]])
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, label=lab)
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=2)
    draw(P, "#94a3b8", 0.35, label1)
    if Q is not None:
        draw(Q, "#2563eb", 0.45, label2)
    ax.axhline(0, color="#334155", lw=0.9)
    ax.axvline(0, color="#334155", lw=0.9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    return fig

def plot_basis_grid(M, lim=4.0, show_grid=True, show_para=True, show_inv=False):
    """기저벡터가 어디로 가는지 + 격자 전체의 변형을 보여준다"""
    fig, ax = plt.subplots(figsize=(5.6, 5.6), facecolor=FIG_BG)
    K = int(lim) + 3

    for k in range(-K, K + 1):                       # 변환 전 격자 (연회색)
        ax.plot([k, k], [-K, K], color="#d4d4d8", lw=0.8, zorder=1)
        ax.plot([-K, K], [k, k], color="#d4d4d8", lw=0.8, zorder=1)

    if show_grid:                                    # 변환 후 격자 (파랑)
        for k in range(-K, K + 1):
            for P in ([[k, -K], [k, K]], [[-K, k], [K, k]]):
                Q = np.array(P, float) @ M.T
                ax.plot(Q[:, 0], Q[:, 1], color="#60a5fa", lw=0.9, zorder=2)

    u, v = M[:, 0], M[:, 1]
    if show_para:                                    # det = 평행사변형 넓이
        poly = np.array([[0, 0], u, u + v, v])
        ax.fill(poly[:, 0], poly[:, 1], color="#a78bfa", alpha=0.35, zorder=3)

    for tip in [(1, 0), (0, 1)]:                     # 원래 기저벡터 i, j (점선)
        ax.annotate("", xy=tip, xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#9ca3af", ls="--"))
    ax.annotate("", xy=tuple(u), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="->", lw=3.0, color="#dc2626"))
    ax.annotate("", xy=tuple(v), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="->", lw=3.0, color="#16a34a"))
    ax.text(u[0] * 1.08, u[1] * 1.08, "u=(a,c)", color="#dc2626", fontsize=11, weight="bold", zorder=6)
    ax.text(v[0] * 1.08, v[1] * 1.08, "v=(b,d)", color="#16a34a", fontsize=11, weight="bold", zorder=6)

    if show_inv:
        Mi = inv2(M)
        if Mi is not None:
            for k in range(-K, K + 1):
                for P in ([[k, -K], [k, K]], [[-K, k], [K, k]]):
                    Q = np.array(P, float) @ Mi.T
                    ax.plot(Q[:, 0], Q[:, 1], color="#f59e0b", lw=0.8, ls=":", zorder=2)

    ax.axhline(0, color="#1f2937", lw=1.0, zorder=3)
    ax.axvline(0, color="#1f2937", lw=1.0, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks(range(-int(lim), int(lim) + 1))
    ax.set_yticks(range(-int(lim), int(lim) + 1))
    ax.tick_params(labelsize=8)
    return fig


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

# ---------------------------------------------------------------- 헤더
st.title("🔢 역행렬 시뮬레이터")
st.caption("수학–정보 융합 프로젝트(기초) · 역행렬과 파이썬 · 당곡고등학교")

tabs = st.tabs([
    "0. 기저벡터와 행렬식",
    "1. 역행렬 계산기",
    "2. 일차변환 실험실",
    "3. 연립일차방정식",
    "4. 이미지 변환",
    "5. 커널 필터 (AI)",
    "6. CNN 단계 체험 (AI)",
    "7. 행렬 암호",
])

# ================================================================ 0. 기저벡터
with tabs[0]:
    st.header("행렬은 기저벡터를 어디로 보내는가")
    st.caption("행렬 A는 î=(1,0), ĵ=(0,1) 을 새 기저벡터 u=(a,c), v=(b,d) 로 보냅니다. "
               "그 둘이 만드는 평행사변형의 넓이가 곧 행렬식입니다.")

    if "bg" not in st.session_state:
        st.session_state.bg = dict(a=1.0, b=0.5, c=0.0, d=1.0)

    def set_bg(a, b, c, d):
        st.session_state.bg = dict(a=a, b=b, c=c, d=d)

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("**프리셋**")
        p1, p2 = st.columns(2)
        p1.button("항등 E", use_container_width=True, on_click=set_bg, args=(1., 0., 0., 1.))
        p2.button("기울임체", use_container_width=True, on_click=set_bg, args=(1., .8, 0., 1.))
        p1.button("회전 53°", use_container_width=True, on_click=set_bg, args=(.6, -.8, .8, .6))
        p2.button("x축 대칭", use_container_width=True, on_click=set_bg, args=(1., 0., 0., -1.))
        p1.button("닮음 1.5배", use_container_width=True, on_click=set_bg, args=(1.5, 0., 0., 1.5))
        p2.button("det = 0", use_container_width=True, on_click=set_bg, args=(1., 2., 2., 4.))

        st.divider()
        bg = st.session_state.bg
        a = st.slider("a  (u의 x성분)", -3.0, 3.0, bg["a"], 0.1, key="sl_a")
        c = st.slider("c  (u의 y성분)", -3.0, 3.0, bg["c"], 0.1, key="sl_c")
        b = st.slider("b  (v의 x성분)", -3.0, 3.0, bg["b"], 0.1, key="sl_b")
        d = st.slider("d  (v의 y성분)", -3.0, 3.0, bg["d"], 0.1, key="sl_d")

        st.divider()
        show_grid = st.checkbox("변환된 격자 보기", True)
        show_para = st.checkbox("평행사변형 보기", True)
        show_inv  = st.checkbox("역행렬 격자도 겹쳐 보기", False)

    M = np.array([[a, b], [c, d]], float)
    dM = det2(M)

    with right:
        st.pyplot(plot_basis_grid(M, show_grid=show_grid,
                                  show_para=show_para, show_inv=show_inv))

    m1, m2, m3 = st.columns(3)
    m1.metric("det A = ad − bc", f"{dM:.3f}")
    m2.metric("평행사변형 넓이", f"{abs(dM):.3f}")
    m3.metric("방향", "유지" if dM > 1e-9 else ("뒤집힘" if dM < -1e-9 else "붕괴"))

    if abs(dM) < 1e-9:
        st.error("**det = 0** — 두 기저벡터가 일직선 위에 놓였습니다. "
                 "격자 전체가 하나의 직선으로 붕괴했고, 넓이는 0입니다. "
                 "여러 점이 같은 자리로 뭉개졌으므로 **되돌릴 수 없습니다.**")
    elif dM < 0:
        st.warning(f"**det < 0** — u에서 v로 가는 회전 방향이 반대가 되었습니다. "
                   f"평면이 뒤집혔고, 넓이는 {abs(dM):.3f}배입니다.")
    else:
        st.success(f"넓이가 **{dM:.3f}배**가 되었습니다. 역행렬은 이것을 다시 "
                   f"**1/{dM:.3f} = {1/dM:.3f}배**로 되돌립니다 — 역행렬 공식에 1/det가 붙는 이유입니다.")

    with st.expander("💡 슬라이더로 직접 확인해 보기"):
        st.markdown("""
1. **a만 키워 보세요.** 빨간 화살표 u만 길어지고 평행사변형이 옆으로 늘어납니다. → a는 î의 행선지
2. **b를 키워 보세요.** 초록 화살표 v가 오른쪽으로 눕습니다. 이게 바로 **기울임체**입니다.
   이때 넓이는? → 밑변도 높이도 그대로라 **변하지 않습니다** (det = 1)
3. **c를 조금씩 키워 두 화살표를 겹쳐 보세요.** det가 0을 지나는 순간 격자가 한 줄로 붕괴합니다.
4. **c를 더 키워 보세요.** det가 음수가 되며 격자가 뒤집힙니다.
5. **'역행렬 격자도 겹쳐 보기'** 를 켜면 주황 점선이 나타납니다. 파란 격자를 원래대로 되돌리는 변환입니다.

> 📌 이 자료는 열벡터 규약을 따릅니다. 행렬의 **1열 (a,c)** 가 u, **2열 (b,d)** 가 v 입니다.
> 책자와 같은 규약이니 수학 시간에 배운 그대로입니다.
        """)

    with st.expander("🔍 왜 하필 ad − bc 일까?"):
        st.markdown("""
평행사변형 넓이를 직접 계산해 보면 나옵니다. u=(a,c), v=(b,d) 라 할 때,
u와 v를 감싸는 큰 직사각형에서 주변 삼각형들의 넓이를 빼면 됩니다.

$$\text{큰 직사각형} - \text{삼각형들} = ad - bc$$

**심화 미션 G**로 직접 유도해 보세요. 종이와 연필만 있으면 됩니다.
(참고: 공돌이의 수학정리노트 『행렬식의 기하학적 의미』)
        """)


# ================================================================ 1. 역행렬
with tabs[1]:
    st.header("행렬식과 역행렬")
    left, right = st.columns([1, 1.4])
    with left:
        A = matrix_input("행렬 A 입력", [[3, 1], [2, 1]], "t1")
        st.divider()
        st.markdown("**빠른 예시**")
        st.caption("아래 값을 직접 입력해 보세요")
        st.code("역행렬 있음: 3 1 / 2 1\n역행렬 없음: 2 4 / 1 2\n회전(60°): 0.5 -0.866 / 0.866 0.5")
    with right:
        d = det2(A)
        show_matrix(A, "A")
        st.latex(rf"\det A = ad-bc = {A[0,0]:.3g}\times{A[1,1]:.3g}-{A[0,1]:.3g}\times{A[1,0]:.3g} = {d:.4g}")
        Ainv = inv2(A)
        if Ainv is None:
            st.error("**det A = 0 → 역행렬이 존재하지 않습니다.**\n\n"
                     "이 변환은 평면 전체를 하나의 직선(또는 점)으로 납작하게 눌러 버립니다. "
                     "눌린 정보는 되돌릴 수 없기 때문에 역변환이 불가능합니다.")
        else:
            st.success("det A ≠ 0 → 역행렬이 존재합니다.")
            show_matrix(Ainv, "A^{-1}")
            st.markdown("**검산: $A A^{-1}$**")
            show_matrix(A @ Ainv, "AA^{-1}")
            st.info(f"|det A| = {abs(d):.4g} → 이 변환은 도형의 넓이를 **{abs(d):.4g}배**로 만듭니다."
                    + ("  (det < 0 이므로 **좌우가 뒤집힙니다**)" if d < 0 else ""))

    with st.expander("💡 생각해 보기"):
        st.markdown("""
- a, b, c, d를 조금씩 바꾸면서 det A가 0이 되는 순간을 찾아보세요. 그때 두 행 `(a,b)`와 `(c,d)`는 어떤 관계인가요?
- 회전변환 행렬 $\\begin{pmatrix}\\cos\\theta & -\\sin\\theta\\\\ \\sin\\theta & \\cos\\theta\\end{pmatrix}$ 의 행렬식은 $\\theta$와 상관없이 항상 얼마인가요? 왜 그럴까요?
- 대칭변환의 행렬식은 왜 음수일까요?
        """)

# ================================================================ 2. 일차변환
with tabs[2]:
    st.header("일차변환 실험실 — 워드프로세서 속 행렬 찾기")
    st.caption("책자 30쪽 『워드프로세서에서 찾는 일차변환』 활동을 직접 시뮬레이션합니다.")

    c0, c1, c2 = st.columns([1, 1, 1.6])
    with c0:
        shape_name = st.selectbox("도형 선택", list(SHAPES.keys()))
        preset = st.selectbox("변환 프리셋", [
            "직접 입력", "기울임체(전단)", "굵게(가로 확대)", "회전", "x축 대칭",
            "y축 대칭", "원점 대칭", "y=x 대칭", "닮음(확대/축소)", "찌그러뜨리기(det=0)",
        ])
    with c1:
        if preset == "기울임체(전단)":
            k = st.slider("기울기 k", -1.5, 1.5, 0.4, 0.05)
            M = np.array([[1, k], [0, 1]], float)
        elif preset == "굵게(가로 확대)":
            k = st.slider("가로 배율", 0.2, 3.0, 1.5, 0.1)
            M = np.array([[k, 0], [0, 1]], float)
        elif preset == "회전":
            th = st.slider("회전각 θ (도)", -180, 180, 45, 5)
            r = np.radians(th)
            M = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        elif preset == "x축 대칭":
            M = np.array([[1, 0], [0, -1]], float)
        elif preset == "y축 대칭":
            M = np.array([[-1, 0], [0, 1]], float)
        elif preset == "원점 대칭":
            M = np.array([[-1, 0], [0, -1]], float)
        elif preset == "y=x 대칭":
            M = np.array([[0, 1], [1, 0]], float)
        elif preset == "닮음(확대/축소)":
            k = st.slider("닮음비 k", -2.0, 3.0, 1.5, 0.1)
            M = np.array([[k, 0], [0, k]], float)
        elif preset == "찌그러뜨리기(det=0)":
            M = np.array([[1, 2], [2, 4]], float)
        else:
            M = matrix_input("변환 행렬", [[1, 0.4], [0, 1]], "t2")
        apply_inv = st.checkbox("역행렬 $M^{-1}$을 한 번 더 적용 (되돌리기)")

    P = SHAPES[shape_name]
    Q = P @ M.T                       # 각 점에 M을 곱함 (행벡터라 전치)
    Minv = inv2(M)
    if apply_inv and Minv is not None:
        Q = Q @ Minv.T
    with c2:
        show_matrix(M, "M")
        st.latex(rf"\det M = {det2(M):.4g}")
        st.pyplot(plot_shapes(P, Q))

    a0, a1 = polygon_area(P), polygon_area(Q)
    m1, m2, m3 = st.columns(3)
    m1.metric("원래 넓이", f"{abs(a0):.4g}")
    m2.metric("변환 후 넓이", f"{abs(a1):.4g}")
    m3.metric("넓이 배율", f"{(abs(a1/a0) if abs(a0)>1e-9 else 0):.4g}", help="|det M| 과 비교해 보세요")

    if apply_inv:
        if Minv is None:
            st.error("det M = 0 이라 역행렬이 없습니다 → **되돌릴 수 없습니다.** 도형이 선으로 뭉개진 것을 확인하세요.")
        else:
            st.success("$M^{-1}M = E$ 이므로 원래 도형으로 정확히 돌아옵니다. (두 도형이 겹쳐 보입니다)")

    with st.expander("💡 탐구 과제"):
        st.markdown("""
1. **기울임체(전단) 행렬** $\\begin{pmatrix}1&k\\\\0&1\\end{pmatrix}$ 의 행렬식은 k와 상관없이 1입니다.
   글자는 분명 기울어졌는데 **넓이는 왜 변하지 않을까요?**
2. 회전 45° 후 회전 −45°를 하면 제자리로 옵니다. 두 행렬의 곱을 계산해 확인해 보세요.
3. `찌그러뜨리기(det=0)`을 고르고 '되돌리기'를 눌러 보세요. 왜 실패할까요?
4. 대칭변환 행렬을 두 번 적용하면? $M^2 = E$ 인 행렬을 다른 것도 찾아보세요.
        """)

# ================================================================ 3. 연립방정식
with tabs[3]:
    st.header("연립일차방정식을 역행렬로 풀기")
    st.latex(r"\begin{cases} ax+by=p \\ cx+dy=q \end{cases} \iff AX=B \iff X=A^{-1}B")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        ex = st.selectbox("예제 불러오기", [
            "직접 입력", "오리와 양 (책자 24쪽)", "해가 무수히 많음", "해가 없음",
        ])
        defaults = {
            "오리와 양 (책자 24쪽)": (1, 1, 15, 2, 4, 40),
            "해가 무수히 많음": (1, 2, 3, 2, 4, 6),
            "해가 없음": (1, 2, 3, 2, 4, 7),
            "직접 입력": (3, 1, 9, 2, 1, 7),
        }[ex]
        cc = st.columns(3)
        a = cc[0].number_input("a", value=float(defaults[0]), key="s_a")
        b = cc[1].number_input("b", value=float(defaults[1]), key="s_b")
        p = cc[2].number_input("p", value=float(defaults[2]), key="s_p")
        c_ = cc[0].number_input("c", value=float(defaults[3]), key="s_c")
        d_ = cc[1].number_input("d", value=float(defaults[4]), key="s_d")
        q = cc[2].number_input("q", value=float(defaults[5]), key="s_q")
        if ex == "오리와 양 (책자 24쪽)":
            st.caption("머리 15개, 다리 40개. 오리 x마리, 양 y마리는?")

    A = np.array([[a, b], [c_, d_]], float)
    B = np.array([[p], [q]], float)
    dA = det2(A)

    with c2:
        st.latex(rf"\det A = {dA:.4g}")
        Ainv = inv2(A)
        if Ainv is None:
            allzero = np.allclose([a, b, c_, d_], 0)
            same = (not allzero) and np.isclose(a * q, c_ * p) and np.isclose(b * q, d_ * p)
            if allzero and not np.allclose([p, q], 0):
                same = False
            if same:
                st.warning("det A = 0 이고 두 식이 같은 직선 → **해가 무수히 많습니다.**")
            else:
                st.error("det A = 0 이고 두 직선이 평행 → **해가 없습니다.**")
        else:
            X = Ainv @ B
            st.latex(rf"X=A^{{-1}}B=\begin{{pmatrix}}{X[0,0]:.4g}\\ {X[1,0]:.4g}\end{{pmatrix}}")
            st.success(f"x = {X[0,0]:.4g},  y = {X[1,0]:.4g}")

    # 두 직선 그래프
    fig, ax = plt.subplots(figsize=(5.4, 4.4), facecolor=FIG_BG)
    xs = np.linspace(-20, 20, 400)
    for (aa, bb, pp, col, lab) in [(a, b, p, "#2563eb", "line 1"), (c_, d_, q, "#dc2626", "line 2")]:
        if abs(bb) > 1e-9:
            ax.plot(xs, (pp - aa * xs) / bb, color=col, lw=2, label=lab)
        elif abs(aa) > 1e-9:
            ax.axvline(pp / aa, color=col, lw=2, label=lab)
    if Ainv is not None:
        X = Ainv @ B
        ax.plot(X[0, 0], X[1, 0], "o", ms=11, color="#16a34a", zorder=5, label="solution")
        lim = max(6, abs(X[0, 0]) * 1.8, abs(X[1, 0]) * 1.8)
    else:
        lim = 12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#334155", lw=0.8); ax.axvline(0, color="#334155", lw=0.8)
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    st.pyplot(fig)

    with st.expander("💡 생각해 보기"):
        st.markdown("""
- det A = 0 이라는 것은 그래프에서 두 직선이 어떤 상태라는 뜻인가요?
- 미지수가 3개, 4개, … 100개로 늘어나면? 손으로 역행렬을 구하는 건 사실상 불가능합니다.
  그래서 컴퓨터는 역행렬을 직접 구하지 않고 **가우스 소거법**으로 풉니다. (책자 5쪽 참고)
- 파이썬에서도 `np.linalg.inv(A) @ B` 보다 `np.linalg.solve(A, B)` 가 더 빠르고 정확합니다. 왜일까요?
        """)

# ================================================================ 4. 이미지 변환
with tabs[4]:
    st.header("이미지도 결국 행렬이다")
    st.caption("사진 한 장 = 픽셀 값을 담은 거대한 행렬. 이 좌표에 2×2 행렬을 곱하면?")

    up = st.file_uploader("이미지 업로드 (없으면 샘플 사용)", type=["png", "jpg", "jpeg"])
    N = 220
    if up is not None:
        from PIL import Image
        img = Image.open(up).convert("RGB").resize((N, N))
        arr = np.asarray(img)
    else:
        yy, xx = np.mgrid[0:N, 0:N]
        arr = np.stack([
            (xx * 255 // N).astype(np.uint8),
            (yy * 255 // N).astype(np.uint8),
            (((xx // 22 + yy // 22) % 2) * 200 + 30).astype(np.uint8),
        ], axis=-1)

    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.radio("변환", ["기울임(전단)", "회전", "확대/축소", "직접 입력"])
        if kind == "기울임(전단)":
            k = st.slider("k", -1.0, 1.0, 0.4, 0.05, key="im_k")
            M = np.array([[1, k], [0, 1]], float)
        elif kind == "회전":
            th = st.slider("θ(도)", -180, 180, 30, 5, key="im_t")
            r = np.radians(th)
            M = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        elif kind == "확대/축소":
            sx = st.slider("가로", 0.3, 2.5, 1.0, 0.1)
            sy = st.slider("세로", 0.3, 2.5, 1.5, 0.1)
            M = np.array([[sx, 0], [0, sy]], float)
        else:
            M = matrix_input("M", [[1, 0.5], [0.2, 1]], "t4")
        show_matrix(M, "M")
        st.latex(rf"\det M = {det2(M):.4g}")

    Minv = inv2(M)
    with c2:
        if Minv is None:
            st.error("det M = 0 → 역변환이 불가능해 이미지를 복원할 수 없습니다.")
        else:
            # 역방향 매핑: 결과의 각 픽셀이 원본 어디서 왔는지 M^-1로 역추적
            H = W = N
            oy, ox = np.mgrid[0:H, 0:W]
            cx, cy = W / 2, H / 2
            X = ox - cx
            Y = cy - oy                      # 화면 좌표 → 수학 좌표
            src = np.stack([X.ravel(), Y.ravel()])
            sxy = Minv @ src
            sx_ = np.round(sxy[0] + cx).astype(int)
            sy_ = np.round(cy - sxy[1]).astype(int)
            ok = (sx_ >= 0) & (sx_ < W) & (sy_ >= 0) & (sy_ < H)
            out = np.full((H * W, 3), 245, np.uint8)
            out[ok] = arr[sy_[ok], sx_[ok]]
            out = out.reshape(H, W, 3)
            i1, i2 = st.columns(2)
            i1.image(arr, caption="원본", use_container_width=True)
            i2.image(out, caption="변환 결과", use_container_width=True)

    with st.expander("💡 AI와 행렬"):
        st.markdown("""
- 이미지 분류 AI는 사진을 **행렬로 바꾼 뒤**, 여러 단계의 행렬 연산으로 특징을 뽑아냅니다. (책자 9쪽)
- 학습 데이터를 늘리는 **데이터 증강(augmentation)** 도 바로 이 회전·전단·확대 행렬을 이용합니다.
- 신경망의 한 층은 사실상 $y = Wx + b$, 즉 **행렬 곱 + 덧셈**입니다. 우리가 배운 일차변환이 층층이 쌓인 것이죠.
- 다만 신경망은 대부분 **정사각행렬이 아니고 det도 0에 가까워** 역행렬로 되돌릴 수 없습니다. 그래서 '학습'이 필요합니다.
        """)

# ================================================================ 7. 암호
with tabs[7]:
    st.header("행렬 암호(힐 암호) 맛보기")
    st.caption("역행렬이 있으면 암호를 풀 수 있고, 없으면 못 푼다 — 암호학의 출발점")

    c1, c2 = st.columns([1, 1])
    with c1:
        key_txt = st.text_input("열쇠 행렬 (a b c d)", "3 3 2 5")
        msg = st.text_input("영문 메시지", "MATRIX").upper()
    try:
        k = [int(v) for v in key_txt.split()]
        K = np.array([[k[0], k[1]], [k[2], k[3]]])
    except Exception:
        K = np.array([[3, 3], [2, 5]])

    def modinv(x, m=26):
        x %= m
        for i in range(1, m):
            if (x * i) % m == 1:
                return i
        return None

    letters = [c for c in msg if c.isalpha()]
    if len(letters) % 2:
        letters.append("X")
    nums = np.array([ord(c) - 65 for c in letters]).reshape(-1, 2).T

    dK = int(round(det2(K.astype(float)))) % 26
    inv_d = modinv(dK)

    with c2:
        show_matrix(K, "K")
        st.latex(rf"\det K \equiv {dK} \pmod{{26}}")
        if inv_d is None:
            st.error(f"det K = {dK} 는 26과 서로소가 아니어서 **mod 26에서 역행렬이 없습니다.** "
                     "→ 이 열쇠로는 암호를 만들어도 **복호화할 수 없습니다.**")
        else:
            enc = (K @ nums) % 26
            cipher = "".join(chr(v + 65) for v in enc.T.ravel())
            adj = np.array([[K[1, 1], -K[0, 1]], [-K[1, 0], K[0, 0]]])
            Kinv = (inv_d * adj) % 26
            dec = (Kinv @ enc) % 26
            plain = "".join(chr(v + 65) for v in dec.T.ravel())
            st.success(f"암호문: **{cipher}**")
            st.write("mod 26 역행렬 $K^{-1}$:")
            show_matrix(Kinv, "K^{-1}")
            st.info(f"복호문: **{plain}**")

    st.divider()
    st.subheader("그런데 왜 현대 암호는 이걸 안 쓸까?")
    st.markdown("""
| | 힐 암호 (행렬) | RSA · AES (현대 암호) |
|---|---|---|
| 열쇠 | 2×2면 경우의 수가 적음 | 2048비트 이상 |
| 약점 | 평문–암호문 쌍 몇 개만 알면 **연립방정식을 풀어** 열쇠가 그대로 노출 | 큰 수의 소인수분해가 현실적으로 불가능 |
| 열쇠 공유 | 암호화·복호화 열쇠가 같음(대칭) | RSA는 **공개키/비밀키가 다름** |

즉, **역행렬이 존재한다는 것 = 되돌릴 수 있다는 것**이고,
암호는 오히려 *"열쇠를 아는 사람만 되돌릴 수 있어야"* 합니다.
현대 암호는 '역연산이 수학적으로는 가능하지만 시간이 어마어마하게 오래 걸리는' 구조를 씁니다.

> 🔎 더 깊은 내용은 2학년 **정보보안 프로그램**에서 다룹니다.
    """)

# ================================================================ 5. 커널
with tabs[5]:
    st.header("커널 필터 — 컴퓨터가 윤곽선을 찾는 법")
    st.caption("작은 3×3 행렬을 이미지 위에서 미끄러뜨리며 곱해 더합니다. 이것이 합성곱(convolution)입니다.")

    KP = {
        "원본 유지":    [[0,0,0],[0,1,0],[0,0,0]],
        "블러(흐리게)": [[1/9]*3]*3,
        "샤프닝":       [[0,-1,0],[-1,5,-1],[0,-1,0]],
        "엣지 - 세로":  [[-1,0,1],[-2,0,2],[-1,0,1]],
        "엣지 - 가로":  [[-1,-2,-1],[0,0,0],[1,2,1]],
        "엣지 - 전체":  [[0,-1,0],[-1,4,-1],[0,-1,0]],
        "엠보싱":       [[-2,-1,0],[-1,1,1],[0,1,2]],
    }
    if "kn" not in st.session_state:
        st.session_state.kn = KP["엣지 - 세로"]

    c1, c2 = st.columns([1, 1.7])
    with c1:
        pick = st.selectbox("프리셋", list(KP.keys()), index=3)
        if st.button("프리셋 적용", use_container_width=True):
            st.session_state.kn = KP[pick]
        st.markdown("**커널 값 (직접 수정 가능)**")
        K = np.zeros((3, 3))
        for r in range(3):
            cols = st.columns(3)
            for cc in range(3):
                K[r, cc] = cols[cc].number_input(
                    f"k{r}{cc}", value=float(st.session_state.kn[r][cc]),
                    step=0.5, key=f"k_{r}_{cc}", label_visibility="collapsed")
        st.metric("커널의 합", f"{K.sum():.2f}")
        if abs(K.sum()) < 1e-9:
            st.info("합 = 0 → 평평한 곳은 검게, **변화가 있는 곳만** 살아남습니다 (엣지 검출)")
        elif abs(K.sum() - 1) < 1e-9:
            st.info("합 = 1 → 전체 밝기가 유지됩니다 (블러·샤프닝)")
        else:
            st.warning(f"합이 {K.sum():.2f} 이라 전체가 더 밝거나 어두워집니다")
        upk = st.file_uploader("이미지 업로드(선택)", type=["png","jpg","jpeg"], key="upk")

    N = 200
    if upk is not None:
        from PIL import Image
        base = np.asarray(Image.open(upk).convert("L").resize((N, N)), float)
    else:
        yy, xx = np.mgrid[0:N, 0:N]
        base = (((xx//25 + yy//25) % 2) * 150 + 40
                + 60*np.sin(xx/9) + 40*np.cos(yy/13))
        base = np.clip(base, 0, 255)

    pad = np.pad(base, 1, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3))
    out = np.einsum("ijkl,kl->ij", win, K)

    with c2:
        i1, i2 = st.columns(2)
        i1.image(base.astype(np.uint8), caption="원본", use_container_width=True, clamp=True)
        i2.image(np.clip(np.abs(out), 0, 255).astype(np.uint8),
                 caption="필터 적용 결과", use_container_width=True, clamp=True)

    with st.expander("💡 AI와 커널"):
        st.markdown("""
- 이미지 인식 AI(**CNN**)의 첫 단계가 바로 이 커널 연산입니다.
- 결정적인 차이: **진짜 AI는 커널의 9개 숫자를 사람이 정하지 않습니다.**
  수많은 사진을 보며 **스스로 찾아냅니다.** 그게 '학습'입니다.
- 우리가 위에서 손으로 넣은 숫자를, AI는 데이터로부터 알아냅니다.
        """)


# ================================================================ 6. CNN
with tabs[6]:
    st.header("CNN 한 층 체험 — 합성곱 → ReLU → 풀링")
    st.caption("이미지 인식 AI의 한 층은 딱 세 단계입니다. 층을 지날수록 작아지고 추상적이 됩니다.")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        kname = st.selectbox("사용할 커널", list(KP.keys()), index=3, key="cnn_k")
        nlayer = st.slider("층 수", 1, 4, 2)
        upc = st.file_uploader("이미지 업로드(선택)", type=["png","jpg","jpeg"], key="upc")

    Kc = np.array(KP[kname], float)
    M0 = 128
    if upc is not None:
        from PIL import Image
        x = np.asarray(Image.open(upc).convert("L").resize((M0, M0)), float)
    else:
        yy, xx = np.mgrid[0:M0, 0:M0]
        x = np.clip(((xx//16 + yy//16) % 2)*150 + 40 + 50*np.sin(xx/7), 0, 255)

    def conv(a, k):
        p = np.pad(a, 1, mode="edge")
        w = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
        return np.einsum("ijkl,kl->ij", w, k)

    def pool(a, s=2):
        H, W = a.shape; H, W = H - H % s, W - W % s
        return a[:H, :W].reshape(H//s, s, W//s, s).max(axis=(1, 3))

    stages = [("입력", x)]
    cur = x
    for L in range(nlayer):
        cv = conv(cur, Kc); rl = np.maximum(cv, 0); pl = pool(rl)
        stages += [(f"{L+1}층 ① 합성곱", cv), (f"{L+1}층 ② ReLU", rl), (f"{L+1}층 ③ 풀링", pl)]
        cur = pl

    with c2:
        for row in range(0, len(stages), 4):
            cols = st.columns(4)
            for col, (t, m) in zip(cols, stages[row:row+4]):
                v = np.clip(np.abs(m) * (255/max(np.abs(m).max(), 1e-9)), 0, 255)
                col.image(v.astype(np.uint8), caption=f"{t}\n{m.shape}",
                          use_container_width=True, clamp=True)

    st.info(f"크기 변화: {x.shape} → {cur.shape}  "
            f"({x.size:,}개 → {cur.size:,}개, {cur.size/x.size:.1%}로 압축)")

    with st.expander("💡 왜 작게 만들까?"):
        st.markdown("""
- **풀링**은 "이 근처에 특징이 있다"만 남기고 정확한 위치는 버립니다.
- 그래서 고양이가 사진의 왼쪽에 있든 오른쪽에 있든 **같은 고양이로 인식**합니다.
- 층이 깊어질수록: 선 → 모서리 → 눈·코 → 얼굴 → "고양이" 로 추상화됩니다.
- ⚠️ 풀링은 **정보를 버리는 연산**입니다. det=0 변환처럼 **되돌릴 수 없습니다.**
  AI가 학습을 해야만 하는 이유가 여기에도 있습니다.
        """)


st.divider()
st.caption("만든 이: 김하은 | 2026 수학정보 융합캠프 · 역행렬과 파이썬으로 암호 이해하기")
