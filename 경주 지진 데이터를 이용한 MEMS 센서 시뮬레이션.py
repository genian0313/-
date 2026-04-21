"""
================================================================
경주 지진(2016) MEMS 가속도계 시뮬레이션
================================================================
[실행 순서]
  1단계 : IRIS 서버에서 경주 지진 파형 수집
  2단계 : MEMS 물리 파라미터 계산 (질량·스프링·감쇠)
  3단계 : Python ODE 시뮬레이션 (질량체 변위 x(t) 계산)
  4단계 : 정전용량 변환 ΔC(t) → 전압 출력 V_out(t)
  5단계 : 감도 분석 + 모어 원(Mohr Circle) 안전성 진단
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ================================================================
# 1단계 : 지진 데이터 수집
# ================================================================
print("=" * 60)
print("  1단계: 경주 지진 데이터 수집")
print("=" * 60)

USE_SYNTHETIC = False   # IRIS 성공 시 False, 실패 시 True로 자동 전환

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    client  = Client("IRIS")
    eq_time = UTCDateTime("2016-09-12T11:32:54")   # KST 20:32:54 = UTC 11:32:54
    t_start = eq_time - 30
    t_end   = eq_time + 180

    st = client.get_waveforms(
        network        = "IU",
        station        = "INCN",      # 인천 관측소 (한반도 최근접 글로벌 스테이션)
        location       = "00",
        channel        = "BH?",       # BHZ(수직), BHN(남북), BHE(동서)
        starttime      = t_start,
        endtime        = t_end,
        attach_response= True,
    )

    # 전처리
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05)

    # 계측 응답 제거 → 가속도 [m/s²] 변환
    st.remove_response(
        output      = "ACC",
        pre_filt    = (0.005, 0.01, 45, 50),
        water_level = 60,
    )

    tr       = st.select(channel="BHZ")[0]
    time_raw = tr.times()          # 시간 축 [s]  (본진 30초 전 기준)
    a_ground = tr.data             # 가속도 [m/s²]

    print(f"  샘플링 레이트  : {tr.stats.sampling_rate} Hz")
    print(f"  PGA            : {np.max(np.abs(a_ground)):.4f} m/s²")
    print(f"  데이터 포인트  : {len(a_ground)}")
    print("  IRIS 연결 성공 — 실제 지진 데이터 사용")

except Exception as e:
    print(f"  IRIS 연결 실패 ({e})")
    print("  → 합성 신호(P파+S파)로 대체합니다.")
    USE_SYNTHETIC = True

if USE_SYNTHETIC:
    dt_syn   = 0.01
    time_raw = np.arange(0, 210, dt_syn)
    # P파: 고주파(5 Hz), 본진 35초 후 도착, 약한 진폭
    p_wave   = 0.3  * np.exp(-((time_raw - 35)**2) / 8)  * np.sin(2*np.pi*5   * time_raw)
    # S파: 저주파(1.2 Hz), 55초 후 도착, 강한 진폭
    s_wave   = 2.5  * np.exp(-((time_raw - 55)**2) / 60) * np.sin(2*np.pi*1.2 * time_raw)
    a_ground = p_wave + s_wave
    print(f"  합성 데이터 생성 완료 — {len(time_raw)} 포인트 ({time_raw[-1]:.0f} s)")

# 1단계 시각화
fig1, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(time_raw, a_ground * 100, color="steelblue", lw=0.8)
axes[0].axvline(x=30, color="red", ls="--", label="본진 발생 (t=30 s)")
axes[0].set_xlabel("시간 (s)")
axes[0].set_ylabel("가속도 (cm/s²)")
axes[0].set_title(f"2016 경주 지진 지반 가속도  [{'합성' if USE_SYNTHETIC else 'IU.INCN.BHZ 실측'}]")
axes[0].legend()
axes[0].grid(alpha=0.3)

N          = len(a_ground)
dt_data    = time_raw[1] - time_raw[0]
freqs_spec = np.fft.rfftfreq(N, d=dt_data)
spectrum   = np.abs(np.fft.rfft(a_ground))
axes[1].semilogy(freqs_spec, spectrum, color="darkorange", lw=0.8)
axes[1].set_xlabel("주파수 (Hz)")
axes[1].set_ylabel("진폭 스펙트럼")
axes[1].set_title("푸리에 스펙트럼 (MEMS 공진 주파수 설계 참고용)")
axes[1].set_xlim(0, 25)
axes[1].grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("step1_seismic_data.png", dpi=150)
plt.show()
print("  저장: step1_seismic_data.png\n")

# ================================================================
# 2단계 : MEMS 물리 모델 파라미터 계산
# ================================================================
print("=" * 60)
print("  2단계: MEMS 물리 파라미터 계산")
print("=" * 60)

# ── 실리콘 물성치 ──────────────────────────────────────────────
E_si    = 170e9      # 영률 (Young's modulus) — 단결정 <110> [Pa]
rho_si  = 2330       # 밀도 [kg/m³]
nu_si   = 0.28       # 포아송 비

# ── MEMS 구조 치수 ─────────────────────────────────────────────
pm_l    = 500e-6     # 질량체 길이 [m]
pm_w    = 500e-6     # 질량체 너비 [m]
pm_t    = 50e-6      # 질량체 두께 / 에칭 깊이 [m]

beam_L  = 300e-6     # 지지 빔 길이 [m]
beam_w  = 5e-6       # 지지 빔 너비 [m]  ← 감도·강성 결정 핵심 변수
beam_t  = pm_t       # 지지 빔 두께 [m]  (질량체와 동일 층)

d0      = 2e-6       # 전극 초기 간격 [m]
n_f     = 100        # 빗살 전극 쌍 수
h_f     = 50e-6      # 전극 높이 [m]
L_f     = 100e-6     # 전극 길이 [m]

# ── 질량 m ────────────────────────────────────────────────────
m_mass  = rho_si * pm_l * pm_w * pm_t
print(f"  질량 m              = {m_mass*1e9:.4f} ng  ({m_mass:.3e} kg)")

# ── 스프링 상수 k (양단 고정 빔 4개 병렬) ─────────────────────
#   k_single = 12·E·I / L³   (I = w³·t/12)
I_beam  = (beam_w**3 * beam_t) / 12
k_sp    = 4 * (12 * E_si * I_beam / beam_L**3)
print(f"  스프링 상수 k       = {k_sp:.4f} N/m")

# ── 고유 주파수 f₀ ────────────────────────────────────────────
omega_n = np.sqrt(k_sp / m_mass)
f_n     = omega_n / (2 * np.pi)
print(f"  고유 주파수 f₀      = {f_n:.2f} Hz")

# ── 감쇠 계수 c (목표 Q = 0.7: 임계 감쇠에 근접) ─────────────
Q_target = 0.7
c_damp   = (m_mass * omega_n) / Q_target
zeta     = c_damp / (2 * np.sqrt(k_sp * m_mass))
print(f"  감쇠비 ζ            = {zeta:.3f}  (1.0 = 임계 감쇠)")
print(f"  감쇠 계수 c         = {c_damp:.4e} N·s/m")

# ── 전극 정전용량 C₀ ──────────────────────────────────────────
eps0    = 8.854e-12
A_f     = h_f * L_f
C0      = eps0 * n_f * A_f / d0
S_C     = 2 * C0 / d0              # 정전용량 감도 [F/m]
V_ref   = 3.3
S_V     = S_C / C0 * V_ref         # 전압 감도 [V/m]
print(f"  초기 정전용량 C₀    = {C0*1e15:.4f} fF")
print(f"  정전용량 감도 S_C   = {S_C*1e9:.4f} nF/m\n")

# ================================================================
# 3단계 : Python ODE 시뮬레이션
# ================================================================
print("=" * 60)
print("  3단계: ODE 시뮬레이션 (질량체 운동 계산)")
print("=" * 60)

# 지진 가속도 → 연속 함수 보간 (RK45가 임의 t를 요청하므로 필수)
a_interp = interp1d(
    time_raw, a_ground,
    kind        = "linear",
    bounds_error= False,
    fill_value  = 0.0,
)

def mems_ode(t, y):
    """
    상태 벡터 y = [x, v]
      dy/dt = [v,  (-c·v - k·x)/m - a_ground(t)]
    """
    x, v   = y
    a_ext  = float(a_interp(t))
    dxdt   = v
    dvdt   = (-c_damp * v - k_sp * x) / m_mass - a_ext
    return [dxdt, dvdt]

t_span = (time_raw[0], time_raw[-1])
y0     = [0.0, 0.0]   # 초기 조건: 정지 상태

sol = solve_ivp(
    mems_ode,
    t_span,
    y0,
    method      = "RK45",
    t_eval      = time_raw,
    rtol        = 1e-8,
    atol        = 1e-12,
    dense_output= False,
)

if not sol.success:
    raise RuntimeError(f"ODE 풀이 실패: {sol.message}")

x_t = sol.y[0]    # 질량체 변위 [m]
v_t = sol.y[1]    # 질량체 속도 [m/s]
t   = sol.t       # 시간 축 [s]

x_max = np.max(np.abs(x_t))
print(f"  ODE 풀이 성공")
print(f"  최대 변위 |x|_max   = {x_max*1e6:.4f} μm")
print(f"  전극 간격 d₀        = {d0*1e6:.1f} μm")
print(f"  변위/간격 비율       = {x_max/d0*100:.1f} %")
if x_max < 0.3 * d0:
    print("  선형 동작 범위 내 — 정전용량 선형 근사 유효")
else:
    print("  주의: 대변위 — 비선형 정전용량 모델 필요 (d₀ 확대 검토)")

# 3단계 시각화
# 주파수 전달함수 계산 (5단계에서도 재사용)
freqs    = np.fft.rfftfreq(len(t), d=(t[1]-t[0]))
X_fft    = np.abs(np.fft.rfft(x_t))
A_fft    = np.abs(np.fft.rfft(a_ground))
H_full   = np.where(A_fft > 1e-20, X_fft / A_fft, 0)

fig3     = plt.figure(figsize=(13, 9))
gs3      = gridspec.GridSpec(3, 2, figure=fig3, hspace=0.45, wspace=0.35)

ax3_1 = fig3.add_subplot(gs3[0, :])
ax3_1.plot(t, a_ground * 100, color="#E24B4A", lw=0.8, alpha=0.9)
ax3_1.axhline(0, color="gray", lw=0.5)
ax3_1.set_ylabel("지반 가속도 (cm/s²)")
ax3_1.set_title("입력: 경주 지진 지반 가속도 a_ground(t)")
ax3_1.grid(alpha=0.25)

ax3_2 = fig3.add_subplot(gs3[1, :])
ax3_2.plot(t, x_t * 1e6, color="#378ADD", lw=1.0)
ax3_2.axhline(0, color="gray", lw=0.5)
ax3_2.axhline( d0*1e6*0.3, color="orange", lw=0.8, ls="--", label="선형 한계 (30% of d₀)")
ax3_2.axhline(-d0*1e6*0.3, color="orange", lw=0.8, ls="--")
ax3_2.fill_between(t, x_t*1e6, 0, where=(np.abs(x_t) > 0.3*d0),
                   color="red", alpha=0.2, label="비선형 구간")
ax3_2.set_ylabel("질량체 변위 (μm)")
ax3_2.set_title("출력: MEMS 질량체 상대 변위 x(t)")
ax3_2.legend(fontsize=9)
ax3_2.grid(alpha=0.25)

ax3_3 = fig3.add_subplot(gs3[2, 0])
ax3_3.plot(x_t*1e6, v_t*1e3, lw=0.6, color="#1D9E75", alpha=0.8)
ax3_3.set_xlabel("변위 (μm)")
ax3_3.set_ylabel("속도 (mm/s)")
ax3_3.set_title("위상 공간 (phase portrait)")
ax3_3.grid(alpha=0.25)

ax3_4 = fig3.add_subplot(gs3[2, 1])
ax3_4.semilogy(freqs, H_full, color="#7F77DD", lw=0.8)
ax3_4.axvline(f_n, color="red", ls="--", lw=1, label=f"f₀ = {f_n:.1f} Hz")
ax3_4.set_xlim(0, 30)
ax3_4.set_xlabel("주파수 (Hz)")
ax3_4.set_ylabel("|H(f)|  [m/(m/s²)]")
ax3_4.set_title("주파수 전달함수 |X/A|")
ax3_4.legend(fontsize=9)
ax3_4.grid(alpha=0.25, which="both")

fig3.suptitle("3단계: MEMS ODE 시뮬레이션 결과", fontsize=13, fontweight="bold")
fig3.savefig("step3_ode_simulation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  저장: step3_ode_simulation.png\n")

# ================================================================
# 4단계 : 정전용량 변환 → 전압 출력
# ================================================================
print("=" * 60)
print("  4단계: 기계 신호 → 전기 신호 변환")
print("=" * 60)

# pull-in 방지 클리핑 (|x| < d₀ 조건 강제)
x_safe = np.clip(x_t, -0.99*d0, 0.99*d0)

# 차동 정전용량
C1      = eps0 * n_f * A_f / (d0 + x_safe)
C2      = eps0 * n_f * A_f / (d0 - x_safe)
delta_C = C2 - C1

# 선형 근사 (비교용)
delta_C_lin = S_C * x_safe

# 비선형 오차
nl_err = (np.max(np.abs(delta_C - delta_C_lin))
          / np.max(np.abs(delta_C)) * 100)

# 전압 출력: V_out = (ΔC / C₀) · V_ref
V_out   = (delta_C / C0) * V_ref

# ADC 12비트 양자화
adc_lsb    = V_ref / 2**12
V_digital  = np.round(V_out / adc_lsb) * adc_lsb

# 가속도 역산 (준정적 근사: a_meas = k·x / m)
a_measured = (k_sp * x_safe) / m_mass

print(f"  초기 정전용량 C₀    = {C0*1e15:.4f} fF")
print(f"  최대 ΔC (실제)      = {np.max(np.abs(delta_C))*1e15:.4f} fF")
print(f"  최대 ΔC (선형 근사) = {np.max(np.abs(delta_C_lin))*1e15:.4f} fF")
print(f"  비선형 오차          = {nl_err:.3f} %  {'✓' if nl_err<1 else '✗ (d₀ 확대 검토)'}")
print(f"  전압 출력 최대값     = {np.max(np.abs(V_out))*1e3:.4f} mV")
print(f"  ADC LSB              = {adc_lsb*1e3:.4f} mV/count")
print(f"  SNR 추정             = {20*np.log10(np.max(np.abs(V_out))/adc_lsb):.1f} dB")

# 4단계 시각화
fig4 = plt.figure(figsize=(13, 11))
gs4  = gridspec.GridSpec(4, 2, figure=fig4, hspace=0.52, wspace=0.35)

ax4_1 = fig4.add_subplot(gs4[0, :])
ax4_1.plot(t, delta_C*1e15,     color="#378ADD", lw=0.9, label="실제 ΔC")
ax4_1.plot(t, delta_C_lin*1e15, color="#E24B4A", lw=0.7, ls="--", alpha=0.7, label="선형 근사")
ax4_1.set_ylabel("ΔC (fF)")
ax4_1.set_title("x(t) → 정전용량 변화 ΔC(t)")
ax4_1.legend(fontsize=9)
ax4_1.grid(alpha=0.25)

ax4_2 = fig4.add_subplot(gs4[1, :])
ax4_2.plot(t, V_out*1e3,     color="#1D9E75", lw=0.9, label="아날로그 출력")
ax4_2.plot(t, V_digital*1e3, color="#7F77DD", lw=0.6, ls="--", alpha=0.8, label="ADC 12bit 출력")
ax4_2.set_ylabel("출력 전압 (mV)")
ax4_2.set_title("정전용량 → 전압 출력 V_out(t)  +  ADC 디지털 변환")
ax4_2.legend(fontsize=9)
ax4_2.grid(alpha=0.25)

ax4_3 = fig4.add_subplot(gs4[2, :])
ax4_3.plot(t, a_ground*100,   color="#E24B4A", lw=0.8, alpha=0.8, label="실제 지반 가속도")
ax4_3.plot(t, a_measured*100, color="#378ADD", lw=0.9, ls="--",   label="MEMS 측정값 (역산)")
ax4_3.set_ylabel("가속도 (cm/s²)")
ax4_3.set_title("센서 정확도 검증: 실제 vs 측정 지반 가속도")
ax4_3.legend(fontsize=9)
ax4_3.grid(alpha=0.25)

error = (a_measured - a_ground) * 100
ax4_4 = fig4.add_subplot(gs4[3, 0])
ax4_4.plot(t, error, color="#D85A30", lw=0.7)
ax4_4.axhline(0, color="gray", lw=0.5)
ax4_4.set_xlabel("시간 (s)")
ax4_4.set_ylabel("오차 (cm/s²)")
ax4_4.set_title("측정 오차 (비선형 + 위상 지연)")
ax4_4.grid(alpha=0.25)

ax4_5 = fig4.add_subplot(gs4[3, 1])
ax4_5.hist(delta_C*1e15, bins=60, color="#7F77DD", edgecolor="white", lw=0.3)
ax4_5.set_xlabel("ΔC (fF)")
ax4_5.set_ylabel("빈도")
ax4_5.set_title("ΔC 분포 (동적 범위 확인)")
ax4_5.grid(alpha=0.25)

fig4.suptitle("4단계: 기계 신호 → 전기 신호 변환 결과", fontsize=13, fontweight="bold")
fig4.savefig("step4_signal_conversion.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n  최소 감지 가속도     = "
      f"{adc_lsb/V_ref * k_sp/m_mass * 100:.4f} cm/s²")
print("  저장: step4_signal_conversion.png\n")

# ================================================================
# 5단계 : 감도 분석 + 모어 원 안전성 진단
# ================================================================
print("=" * 60)
print("  5단계: 감도 분석 + 모어 원 안전성 진단")
print("=" * 60)

# ── [C] 감도 분석 ────────────────────────────────────────────
print("\n  [감도 분석]")
print("  " + "─"*46)

a_pwave_typical = 0.05    # P파 전형 진폭 [m/s²] ≈ 5 mg

x_per_a  = m_mass / k_sp           # 변위 감도 [m/(m/s²)]
dC_per_a = S_C * x_per_a           # 정전용량 감도 [F/(m/s²)]
dV_per_a = S_V * x_per_a           # 전압 감도 [V/(m/s²)]

a_min    = adc_lsb / dV_per_a      # 최소 감지 가속도 [m/s²]
snr_p    = a_pwave_typical / a_min  # P파 기준 SNR

# 주파수 응답 평탄 대역 (f₀/3 이하에서 오차 < 5%)
f_flat   = f_n / 3

print(f"  변위 감도     S_x  = {x_per_a*1e9:.4f} nm/(m/s²)")
print(f"  정전용량 감도 S_C  = {dC_per_a*1e18:.4f} aF/(m/s²)")
print(f"  전압 감도     S_V  = {dV_per_a*1e6:.4f} μV/(m/s²)")
print(f"  ADC LSB            = {adc_lsb*1e3:.4f} mV")
print(f"  최소 감지 가속도   = {a_min*1e3:.4f} mm/s²  ({a_min/9.8*1000:.3f} mg)")
print(f"  P파 기준 SNR       = {snr_p:.1f}  "
      f"{'✓ 감지 가능' if snr_p > 10 else '✗ 감지 불가 — 설계 수정 필요'}")
print(f"  고유 주파수 f₀     = {f_n:.2f} Hz")
print(f"  평탄 응답 대역     = 0 ~ {f_flat:.2f} Hz")
print(f"  S파(0.1~1 Hz) →  "
      f"{'평탄 대역 내 ✓' if 1.0 < f_flat else '대역 초과 ✗'}")

# ── [D] 모어 원 안전성 분석 ──────────────────────────────────
print("\n  [모어 원 안전성 분석]")
print("  " + "─"*46)

sigma_f   = 7e9     # 실리콘 파괴 강도 이론값 [Pa]
sigma_f_p = 1e9     # 실용 파괴 강도 (결함 고려) [Pa]

# 최대 변위 → 빔 1개 하중
F_max   = k_sp * x_max / 4
M_max   = F_max * beam_L / 2         # 최대 굽힘 모멘트 (양단 고정 빔)
c_dist  = beam_w / 2                  # 중립축 → 표면 거리
I_xx    = (beam_t * beam_w**3) / 12   # 굽힘 축 기준 2차 모멘트

sigma_x = M_max * c_dist / I_xx      # 굽힘 응력 [Pa]

Q_rect  = (beam_t * beam_w / 2) * (beam_w / 4)
tau_xy  = F_max * Q_rect / (I_xx * beam_t)   # 최대 전단응력 [Pa]

# 주응력 (모어 원 공식)
sigma_avg = sigma_x / 2
R_mohr    = np.sqrt((sigma_x/2)**2 + tau_xy**2)
sigma_1   = sigma_avg + R_mohr       # 최대 주응력
sigma_2   = sigma_avg - R_mohr       # 최소 주응력
tau_max   = R_mohr

SF_theory = sigma_f   / sigma_1
SF_prac   = sigma_f_p / sigma_1

print(f"  최대 변위   x_max  = {x_max*1e6:.4f} μm")
print(f"  빔 하중     F_max  = {F_max*1e9:.4f} nN")
print(f"  굽힘 응력   σ_x    = {sigma_x/1e6:.6f} MPa")
print(f"  전단 응력   τ_xy   = {tau_xy/1e6:.6f} MPa")
print(f"  원 중심     C      = {sigma_avg/1e6:.6f} MPa")
print(f"  반지름      R      = {R_mohr/1e6:.6f} MPa")
print(f"  최대 주응력 σ₁     = {sigma_1/1e6:.6f} MPa")
print(f"  최소 주응력 σ₂     = {sigma_2/1e6:.6f} MPa")
print(f"  안전계수 SF (이론) = {SF_theory:.1f}")
print(f"  안전계수 SF (실용) = {SF_prac:.1f}")
print(f"  안전 판정 (SF>3)   : "
      f"{'✓ 안전' if SF_prac > 3 else '✗ 위험 — 빔 보강 필요'}")

# ── 5단계 종합 시각화 ─────────────────────────────────────────
fig5 = plt.figure(figsize=(14, 12))
gs5  = gridspec.GridSpec(3, 2, figure=fig5, hspace=0.50, wspace=0.38)

# (1) 주파수 응답 + 감지 대역
ax5_1 = fig5.add_subplot(gs5[0, :])
freqs_p  = freqs[1:]
H_plot   = np.where(A_fft[1:] > 1e-20,
                    X_fft[1:] / A_fft[1:], 0) * k_sp / m_mass

ax5_1.semilogy(freqs_p, H_plot, color="#378ADD", lw=1.0, label="|H(f)|  전달함수")
ax5_1.axvline(f_n,    color="#E24B4A", ls="--", lw=1.2, label=f"f₀ = {f_n:.1f} Hz")
ax5_1.axvline(f_flat, color="#1D9E75", ls=":",  lw=1.2, label=f"평탄 대역 한계 {f_flat:.1f} Hz")
ax5_1.axvspan(0.1, 1.0, alpha=0.12, color="#D85A30", label="S파 주요 대역")
ax5_1.axvspan(1.0, 10., alpha=0.10, color="#7F77DD", label="P파 주요 대역")
ax5_1.set_xlim(0.05, 30)
ax5_1.set_xlabel("주파수 (Hz)")
ax5_1.set_ylabel("|H| (무차원)")
ax5_1.set_title("감도 분석: 주파수 전달함수와 지진파 감지 대역")
ax5_1.legend(fontsize=8, ncol=3)
ax5_1.grid(alpha=0.25, which="both")

# (2) 모어 원
ax5_2 = fig5.add_subplot(gs5[1, 0], aspect="equal")
theta    = np.linspace(0, 2*np.pi, 360)
circ_x   = sigma_avg/1e6 + R_mohr/1e6 * np.cos(theta)
circ_y   =                  R_mohr/1e6 * np.sin(theta)

ax5_2.plot(circ_x, circ_y, color="#378ADD", lw=1.5, label="모어 원")
ax5_2.fill(circ_x, circ_y, color="#378ADD", alpha=0.10)
ax5_2.axvline(sigma_f_p/1e6, color="#E24B4A", ls="--", lw=1.2, label="파괴 강도 (실용)")
ax5_2.axvline(sigma_f/1e6,   color="#E24B4A", ls=":",  lw=0.8, alpha=0.5, label="파괴 강도 (이론)")
ax5_2.scatter([sigma_1/1e6], [0], color="#1D9E75", s=60, zorder=5,
              label=f"σ₁ = {sigma_1/1e6:.4f} MPa")
ax5_2.scatter([sigma_2/1e6], [0], color="#1D9E75", s=60, zorder=5)
ax5_2.scatter([sigma_x/1e6], [tau_xy/1e6], color="#D85A30", s=60, zorder=5,
              label="응력 상태 P")
ax5_2.scatter([sigma_avg/1e6], [0], color="#7F77DD", s=30, zorder=5)
ax5_2.axhline(0, color="gray", lw=0.7)
ax5_2.axvline(0, color="gray", lw=0.7)
ax5_2.set_xlabel("수직응력 σ (MPa)")
ax5_2.set_ylabel("전단응력 τ (MPa)")
ax5_2.set_title(f"모어 원  (SF = {SF_prac:.0f}  →  {'안전 ✓' if SF_prac>3 else '위험 ✗'})")
ax5_2.legend(fontsize=8)
ax5_2.grid(alpha=0.25)

# (3) 설계 트레이드오프: 빔 너비 vs SF vs f₀
ax5_3  = fig5.add_subplot(gs5[1, 1])
ax5_3b = ax5_3.twinx()
bw_arr = np.linspace(2e-6, 15e-6, 200)
SF_arr, fn_arr = [], []
for bw in bw_arr:
    I_b    = (bw**3 * beam_t) / 12
    k_b    = 4 * (12 * E_si * I_b / beam_L**3)
    fn_b   = np.sqrt(k_b / m_mass) / (2*np.pi)
    x_b    = m_mass * np.max(np.abs(a_ground)) / k_b
    F_b    = k_b * x_b / 4
    M_b    = F_b * beam_L / 2
    I_xx_b = (beam_t * bw**3) / 12
    sig_b  = M_b * (bw/2) / I_xx_b if I_xx_b > 0 else 1e-30
    SF_arr.append(sigma_f_p / sig_b)
    fn_arr.append(fn_b)

SF_arr = np.array(SF_arr)
fn_arr = np.array(fn_arr)

ax5_3.plot(bw_arr*1e6, SF_arr, color="#1D9E75", lw=1.5, label="안전계수 SF")
ax5_3.axhline(3, color="#E24B4A", ls="--", lw=1, label="최소 SF = 3")
ax5_3.axvline(beam_w*1e6, color="#378ADD", ls=":", lw=1.2,
              label=f"현재 설계 ({beam_w*1e6:.0f} μm)")
ax5_3b.plot(bw_arr*1e6, fn_arr, color="#7F77DD", lw=1.2, ls="--", label="f₀ (Hz)")
ax5_3.set_xlabel("빔 너비 (μm)")
ax5_3.set_ylabel("안전계수 SF", color="#1D9E75")
ax5_3b.set_ylabel("고유 주파수 f₀ (Hz)", color="#7F77DD")
ax5_3.set_title("설계 트레이드오프: 빔 너비 vs SF vs f₀")
ax5_3.legend(fontsize=8, loc="upper left")
ax5_3b.legend(fontsize=8, loc="upper right")
ax5_3.grid(alpha=0.25)

# (4) 전체 신호 체인 검증
ax5_4   = fig5.add_subplot(gs5[2, :])
t_xmax  = t[np.argmax(np.abs(x_t))]
ax5_4.plot(t, a_ground*100,   color="#E24B4A", lw=0.8, alpha=0.7, label="실제 지반 가속도 (입력)")
ax5_4.plot(t, a_measured*100, color="#378ADD", lw=1.0, ls="--",   label="MEMS 측정 가속도 (출력)")
ax5_4.axvline(t_xmax, color="#1D9E75", ls=":", lw=1.2,
              label=f"최대 응력 발생  t = {t_xmax:.1f} s")
ax5_4.fill_between(t, a_ground*100, a_measured*100,
                   alpha=0.12, color="#7F77DD", label="측정 오차")
ax5_4.set_xlabel("시간 (s)")
ax5_4.set_ylabel("가속도 (cm/s²)")
ax5_4.set_title("전체 신호 체인 검증: 지반 가속도 → MEMS 측정값")
ax5_4.legend(fontsize=8, ncol=2)
ax5_4.grid(alpha=0.25)

fig5.suptitle("5단계: 감도 분석 + 모어 원 안전성 진단", fontsize=13, fontweight="bold")
fig5.savefig("step5_final_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("  저장: step5_final_analysis.png\n")

# ================================================================
# 최종 공학 판정 요약표
# ================================================================
print("=" * 60)
print("  최종 공학 판정 요약")
print("=" * 60)

criteria = [
    ("P파 감지 SNR > 10",          snr_p > 10,
     f"SNR = {snr_p:.1f}"),
    ("S파 성분 평탄 대역 내",       1.0 < f_flat,
     f"평탄 대역 0 ~ {f_flat:.2f} Hz"),
    ("비선형 오차 < 1 %",           nl_err < 1.0,
     f"오차 = {nl_err:.3f} %"),
    ("안전계수 SF > 3",             SF_prac > 3,
     f"SF = {SF_prac:.1f}"),
    ("최대 주응력 σ₁ < 파괴 강도",  sigma_1 < sigma_f_p,
     f"σ₁ = {sigma_1/1e6:.4f} MPa"),
]
for name, passed, note in criteria:
    mark = "✓" if passed else "✗"
    print(f"  {mark}  {name:<30s}  ({note})")

print("=" * 60)
print("  시뮬레이션 완료")
print(f"  데이터 소스: {'합성 신호 (P파+S파)' if USE_SYNTHETIC else 'IU.INCN.BHZ 실측 데이터'}")
print("=" * 60)