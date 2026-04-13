# xsec_alpha — Cross-Sectional Crypto Ranking System

gan_t(절대 수익률 예측)에서 실패한 교훈을 바탕으로,
**크로스섹션 상대 랭킹**으로 전환한 크립토 트레이딩 연구 프로젝트.

---

## 연구 일지

### 0. 외부 아이디어 검토

두 개의 YouTube 영상을 검토했다:
- **"Claude Just Changed the Stock Market Forever"** — Claude + Alpaca MCP 자동매매 튜토리얼
- **"giving the worlds most expensive AI $10,000 to trade crypto"** — AI 3개 크립토 대결 (Claude Opus가 ETH 50x 숏으로 65% 수익)

**판정:** 정확도/알파 관점에서 줄 수 있는 게 거의 없다.
- 영상 1은 자동 주문 실행 레이어 이야기지, 신호 품질과는 무관
- 영상 2는 36시간 레버리지 트레이딩 = 재현 불가능한 n=1 결과
- xsec_alpha는 이미 이 영상들보다 체계적인 검증 구조(IC 게이트, temporal holdout, 베타 중립)를 갖고 있음

유일하게 참고 가능한 건 자동 주문 연결 패턴이지만, IC < 0.05인 상태에서 자동화는 "손실을 자동화"하는 것뿐이므로 **신호 품질 확인이 먼저**라는 결론.

---

### 1. 출발점 — gan_t에서 배운 것

gan_t는 CVAE/GAN 기반 절대 수익률 예측 시스템이었다.
90일 백테스트 Sharpe 0.17 (랜덤과 구분 불가), 스크리너(55.8%) > 모델(53.1%).

**근본 문제:**
| 문제 | 증거 |
|------|------|
| 3h AC=0.025 (노이즈) | 예측 불가능한 timescale에서 학습 |
| KL=0 (posterior collapse) | CVAE 잠재공간 미사용 |
| fillna(0) | RSI=0, volatility=0 불가능 값 |
| train ⊃ test (100% overlap) | 모든 메트릭 무효 |
| confidence 상수 출력 | position sizing에 정보 없음 |
| 98% short + short 차단 | 실행 가능한 추천 거의 0 |

**핵심 교훈:** 절대 수익률 예측은 BTC 베타에 종속된다. 상대 랭킹으로 가야 베타 노이즈가 상쇄된다.

---

### 2. 설계 결정

| 항목 | gan_t | xsec_alpha |
|------|-------|------------|
| 예측 타겟 | 절대 수익률 | residual return (coin - beta × BTC) |
| 모델 | CVAE + GAN | XGBoost/LGBM/Ridge 앙상블 |
| 검증 | 22h 백테스트 루프 | IC 먼저, holdout, walk-forward 순서 |
| 정규화 | per-market MinMax | per-timestamp 크로스섹션 z-score |
| 결측치 | fillna(0) | ffill only, 첫 480행 drop |
| train/test | 100% overlap | temporal holdout 20% 분리 |

---

### 3. 유니버스 필터 — 현실적 신호 검증

IC를 전체 200+ 코인에서만 보면 극저유동성 코인에서만 나오는 환상적 신호를 잡을 수 있다.
그래서 **24h 거래대금 상위 100개**를 활성 유니버스로 고정하고, 학습/holdout/walk-forward/live 추천 5개 경로에 공통 적용했다.

- full universe → top100 → top50으로 줄여도 핵심 팩터 IC가 유지됨을 확인
- `config.Data.LIQUIDITY_TOP_N = 100`으로 한 곳에서 제어
- 공통 helper: `build_top_liquidity_universe_index()` in `data/features.py`

---

### 4. 팩터 설계 및 IC 측정

Upbit 200+ KRW 코인에서 8개 팩터를 계산하고, 활성 유니버스(top100)에서 IC를 측정했다.

**IC 결과 (120일, top100, 6h horizon):**
| 팩터 | Mean IC | t-stat | 해석 |
|------|---------|--------|------|
| kimchi_inv | +0.1663 | 높음 | 김프가 낮은 코인 선호 |
| binance_lead_1h | +0.1139 | 높음 | 글로벌 가격 선행 |
| volatility_inv_24h | +0.1045 | 높음 | 저변동성 우위 |
| reversal_1h | +0.0952 | 유의 | 1h 낙폭 반등 |
| reversal_4h | +0.0901 | 유의 | 4h 낙폭 반등 |
| order_flow_bear | +0.0615 | 유의 | 매도압 후 반등 |

- 전반/후반 half-split에서 전부 [STABLE]
- top50 유동성으로 줄여도 핵심 팩터 유지

**팩터 부호 정리:**
- 기존 `momentum_1h/4h` → `reversal_1h/4h` (부호 반전, 양의 IC 방향)
- 기존 `order_flow_bull` → `order_flow_bear` (부호 반전)
- 모델 wrapper에 legacy 호환 유지 (구 pickle도 동작)

---

### 5. Horizon 선택 — 1h vs 6h

`ic_decay.py`로 1h~48h IC를 측정한 뒤, 1h와 6h를 후보로 선정.

**1-bar lag falsification:**

| 지표 | 1h lag=0 | 1h lag=1 | 6h lag=0 | 6h lag=1 |
|------|----------|----------|----------|----------|
| Holdout IC | +0.307 | +0.081 | +0.197 | +0.109 |
| WF IC | +0.299 | +0.082 | +0.189 | +0.093 |
| WF Sharpe(net) | +98.3 | +14.3 | +21.2 | +10.8 |
| Holdout gate | PASS | **FAIL** | PASS | **PASS** |

**판정:** 1h는 lag 1바에서 IC가 74% 꺾임 → 마이크로스트럭처 의존으로 판단, **기각**.
6h는 lag=1에서도 holdout gate 통과 → **6h 유지**.

---

### 6. 포트폴리오 구조 실험

6h + lag=1 + buffer=10 + short cost 10bps 기준으로 구조별 비교:

**Holdout (net spread / 6h period):**
| 구조 | Net spread | Top-bottom gap z |
|------|-----------|-----------------|
| L/S 20/20 | +0.0034 | +2.19 |
| L/S 10/10 | +0.0052 | — |
| L/S 5/5 | +0.0081 | +4.16 |
| Short-only 5 | +0.0069 | — |

**Walk-forward (180일, short cost 10bps):**
| 구조 | Sharpe(net) | Total return | Win rate |
|------|-------------|-------------|----------|
| L/S 20/20, buffer 10 | +7.78 | +89.7% | — |
| L/S 5/5, buffer 10 | +9.95 | +789.6% | — |
| Short-only 5, buffer 10 | +10.07 | +910.4% | — |

> 주의: 이 숫자들은 액면 그대로 믿으면 안 된다. 특히 short-only 5의 +910%는
> 소수 극단 이벤트 의존 가능성이 있고, Sharpe 10은 현실에서 나오지 않는 수준이다.

---

### 7. Short 비용 민감도

| short cost | Sharpe(net) | Total return | 손실 윈도우 |
|-----------|-------------|-------------|------------|
| 0 bps | +11.18 | +156% | 0/6 |
| 10 bps | +7.25 | +83% | 2/6 |
| 20 bps | +3.33 | +31% | — |

- 10bps만 넣어도 6개 윈도우 중 2개가 음수로 전환
- 수익이 사실상 전부 short leg에서 발생 (avg long return = -0.04%)

---

### 8. 종목 집중도 감사

Short-only 5 walk-forward에서 종목별 기여도를 분석:

- Short 쪽 distinct name: **136개** (소수 독식 아님)
- Top1 net 기여 비중: 4.9%, Top5: 19.2%, Top10: 32.3%
- 20회 이상 반복 선택: 26개, 40회 이상: 3개
- 최악 단일 포지션: 한 캔들 +33.4% 역행 → -6.7% net contribution

**결론:** "두세 개 코인 독식"은 아니지만, tail-heavy short basket.

---

### 9. Regime 분석

BTC 7d return × 7d vol로 4구간 분류:

| Regime | n | IC | Gross | Net (10bps) | Hit |
|--------|---|-----|-------|-------------|-----|
| bear_highvol | 104 | +0.108 | +0.0045 | +0.0028 | 67.3% |
| bear_lowvol | 107 | +0.091 | +0.0037 | +0.0020 | 52.3% |
| bull_highvol | 68 | +0.085 | +0.0031 | +0.0014 | 60.3% |
| bull_lowvol | 57 | +0.077 | +0.0021 | +0.0004 | 57.9% |

- bull_lowvol이 가장 취약 (short cost 올리면 음수)
- 180일 데이터에 강한 추세장(|BTC 30d| >= 30%)이 거의 없음 (0.6%)

---

### 10. Long 실행 가능성 검증

같은 계약(6h, lag=1, buffer=10)에서 long-only 5를 테스트:

| 지표 | Long-only 5 |
|------|------------|
| Holdout IC | +0.1087 |
| Holdout net spread | +0.0013 |
| WF Total return | **-18.72%** |
| WF Sharpe(net) | **-0.78** |
| WF Max drawdown | -33.84% |
| 음수 윈도우 | 3/6 |

**판정:** IC는 있지만 long-only로 수익 전환 실패. 현재 모델은 **short 신호 전문**.

---

### 11. Tradability 검증

Bitget USDT perp 공개 API로 실제 숏 가능 여부 확인:

- Bitget 542개 USDT perp 중 WF short 후보 136개와 대조
- 이름 기준 커버리지: **93.4%** (127/136)
- Net contribution 기준 커버리지: **90.0%**
- 비상장: ARDR, TFUEL, QKC, STRAX 등 소수

---

### 12. 연구 루프를 닫는 결정

여기까지 오는 동안 반복된 패턴이 있었다:

1. IC 측정 → "좋다, 하지만 유니버스 확인하자"
2. 유니버스 확인 → "좋다, 하지만 horizon 비교하자"
3. Horizon 비교 → "좋다, 하지만 lag 테스트하자"
4. Lag 통과 → "좋다, 하지만 short cost 넣자"
5. Short cost 통과 → "좋다, 하지만 포지션 수 줄이자"
6. 포지션 줄임 → "좋다, 하지만 종목 집중도 보자"
7. 집중도 괜찮음 → "좋다, 하지만 tradability..."

**인식:** 이건 끝이 없다. 검증을 더 하면 할수록 새로운 "하지만"이 나오고, 실제로는 아무것도 결정하지 않고 있다. 백테스트로는 절대 답할 수 없는 것들(실제 슬리피지, 체결률, 펀딩비 변동)이 있고, 그건 라이브에서만 알 수 있다.

**결정:** 연구 루프를 닫고, tradability만 확인한 뒤 소규모 라이브 프로브로 검증 모드를 전환한다. 더 많은 백테스트보다 작은 라이브 프로브가 더 정보가 크다.

---

### 13. 현재 결론

**확인된 것:**
- 크립토 크로스섹션에서 떨어질 코인을 골라내는 신호가 존재한다 (6h, lag=1 IC +0.09)
- 핵심 팩터: kimchi_inv, binance_lead, volatility_inv, reversal 계열
- top100 유니버스, 전반/후반 split에서 안정적
- Bitget에서 실제 숏 가능

**아직 모르는 것:**
- 실전 슬리피지, 체결률, 펀딩비 변동에서 살아남는지
- 강한 상승장에서 short-only가 버티는지
- 실제 net PnL이 양수인지

**현재 판단:**
> "신호는 찾았다. 하지만 그걸로 수익을 내는 포트폴리오 구조는 아직 증명 안 됐다."

---

### 14. 라이브 프로브 계약 (2026-04-13 확정)

| 항목 | 값 |
|------|-----|
| 분석 데이터 | Upbit KRW (200+ 코인) |
| 활성 유니버스 | 24h 거래대금 top 100 |
| 실행 대상 | Bitget USDT perp tradable만 |
| Horizon | 6h |
| 모델 | Ensemble (Ridge + LightGBM) |
| SHORT | 5개 (실행) |
| LONG | 5개 (watch-only, 참고용) |
| Rebal buffer | 10 |
| 연구 기준 short cost | 10bps baseline / 20bps stress |
| 기간 | 2~4주 |
| 변경 금지 | 이 기간 중 팩터/호라이즌/포지션 수 조정 없음 |

**텔레그램 알림 구성:**
- 이전 추천 성적표 (SHORT 실현 손익)
- 현재 추천: SHORT 5 (실행) + WATCH LONG 5 (참고)
- 각 코인: 현재가, 목표가, 손절가, 신호 강도

**프로브 종료 후 판단 기준:**
- 실제 net PnL
- 체결 가능률
- 실제 비용 (슬리피지 + 펀딩비)
- 종목 의존도

---

### 15. 다음 단계 (프로브 병렬)

- [ ] 2~4주 라이브 프로브 실행 및 데이터 수집
- [ ] Long 전용 모델 연구 (별도 팩터/타깃/레짐 게이트)
- [ ] 360일+ 데이터로 추세장 포함 재검증
- [ ] 프로브 종료 후: 계속/조정/중단 결정

---

## 시스템 구조

```
Upbit API ──→ SQLite ──→ 8개 팩터 (크로스섹션 z-score)
                              │
Binance API ─────────────────→│
                              ↓
                    XGBoost/LGBM/Ridge 앙상블
                              │
                    Bitget tradable 필터
                              │
                 ┌─────────────┴─────────────┐
                 │                           │
          SHORT 5 (실행)           WATCH LONG 5 (참고)
                 │                           │
                 └─────────────┬─────────────┘
                               │
                    CSV + 텔레그램 + 대시보드
```

**스케줄:** systemd timer, 12h 간격 (00:00, 12:00 KST)

---

## 실행 방법

```bash
# 데이터 수집
python scripts/update_data.py --days 120

# IC 측정
python scripts/measure_ic.py --days 120 --top-liquidity 100

# 모델 학습
python scripts/train.py --days 120 --model-type ensemble

# Holdout 평가
python scripts/evaluate_holdout.py --days 120 --horizon 6 --execution-lag-bars 1

# Walk-forward 백테스트
python scripts/backtest_wf.py --days 180 --train-days 60 --test-days 14 --rebalance-hours 6 --execution-lag-bars 1

# 추천 생성 (dry-run)
python scripts/fetch_and_rank.py --dry-run --no-telegram

# 추천 생성 (실제 전송)
python scripts/fetch_and_rank.py
```

---

## 라이선스

개인 연구용. 투자 조언이 아님.
