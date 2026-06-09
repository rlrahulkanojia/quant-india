"""Backtest engines (India equity fork).

Engines:
  - BaseEngine: ABC for bar-by-bar execution with market rules
  - GlobalEquityEngine: US / HK / India equities
  - CompositeEngine: Cross-market engine with shared capital pool
  - _market_hooks: Shared market-rule utilities (symbol detection, etc.)

Inheritance:
  BaseEngine
  ├── GlobalEquityEngine
  └── CompositeEngine (delegates to sub-engines as rule providers)
"""
