# Options Flow Sentiment Indicator

A quantitative market regime classification system that identifies stress and calm environments using options market signals, volatility structure, and probabilistic state-transition modeling.

## Objective

Develop a robust market stress classifier capable of identifying changing risk conditions in equity markets.

The system focuses on **regime detection rather than return prediction**, providing a framework for risk management, volatility monitoring, and regime-aware portfolio decisions.

---

## Dataset

| Source              | Data                |
| ------------------- | ------------------- |
| Yahoo Finance       | SPX, VIX, VIX9D     |
| Alternative.me      | Fear & Greed Index  |
| Historical Coverage | 2011–2026           |
| Observations        | 4,100+ Trading Days |

---

## Methodology

### Composite Stress Score

Three independent signals are combined into a weighted market stress indicator:

| Signal                                | Weight |
| ------------------------------------- | ------ |
| Put/Call Sentiment Proxy              | 40%    |
| VIX Momentum                          | 35%    |
| Volatility Term Structure (VIX9D/VIX) | 25%    |

Output:

* Composite Score Range: -100 to +100
* Adaptive Rolling Thresholds
* Five Market Regimes

### Market Regimes

| Regime         |
| -------------- |
| Extreme Stress |
| Stress         |
| Neutral        |
| Calm           |
| Extreme Calm   |

---

## Validation Framework

The model was evaluated using multiple statistical techniques:

* Walk-Forward Out-of-Sample Testing
* Bootstrap Significance Testing (5,000 permutations)
* Markov Chain Transition Analysis
* Regime Persistence Measurement
* Crisis Episode Detection

---

## Key Results

### Regime Classification

* 15+ years of historical data analyzed
* 153 stress episodes identified
* Average stress duration: 2.8 trading days
* Average recovery period: 35 trading days

### Risk Management Performance

| Metric             | Strategy | Buy & Hold |
| ------------------ | -------- | ---------- |
| Sharpe Ratio       | 0.39     | 0.67       |
| Max Drawdown       | -12.3%   | -33.9%     |
| Drawdown Reduction | 64%      | —          |

### Statistical Findings

* Correlation with 3M forward returns: -0.005
* Bootstrap p-value: 0.827
* Regimes showed strong explanatory power for market conditions but no predictive power for future returns

Conclusion:

The model successfully classifies market stress environments but should not be interpreted as an alpha-generating return predictor.

---

## Regime Dynamics

Markov Chain analysis revealed:

* Extreme Stress persistence: 64.8%
* Extreme Calm persistence: 64.8%
* Average holding period of extreme states: 2.8 days
* Markets spend approximately 40% of time in stress-related regimes

These findings indicate significant regime clustering and short-term persistence in market behavior.

---

## Repository Structure

```bash
.
├── backtest.py
├── regime_transitions.py
├── options_sentiment_v3_output.png
├── regime_transitions_output.png
└── README.md
```

### backtest.py

Core research engine:

* Signal generation
* Composite scoring
* Regime classification
* Walk-forward validation
* Bootstrap testing
* Performance evaluation

### regime_transitions.py

State-transition research module:

* Transition matrix estimation
* Steady-state probabilities
* Holding period analysis
* Crisis episode detection
* Regime persistence studies

---

## Research Findings

1. Market regimes are observable and measurable through options market behavior.
2. Extreme regimes exhibit significantly higher persistence than neutral environments.
3. Risk reduction benefits are substantial despite limited return enhancement.
4. Stress classification provides stronger value for portfolio risk management than directional forecasting.

---

## Technical Skills Demonstrated

* Quantitative Research
* Statistical Testing
* Time Series Analysis
* Market Microstructure
* Markov Chains
* Walk-Forward Validation
* Bootstrap Methods
* Python Data Engineering
* Financial Data Analysis
* Risk Management Systems

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* yfinance

---

## Disclaimer

This project is a research system designed for market regime analysis and risk management studies. It is not intended as an investment recommendation or trading signal generator.
