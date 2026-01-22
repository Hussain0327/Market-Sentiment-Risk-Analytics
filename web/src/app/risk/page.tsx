'use client';

import { useState, useEffect } from 'react';
import KPICard from '@/components/KPICard';
import RiskMetrics from '@/components/RiskMetrics';
import PriceChart from '@/components/PriceChart';
import SymbolSelector from '@/components/SymbolSelector';
import type { SymbolsData, RiskData, PriceData } from '@/lib/data';

export default function RiskPage() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>('AAPL');
  const [risk, setRisk] = useState<RiskData | null>(null);
  const [price, setPrice] = useState<PriceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/symbols.json')
      .then((res) => res.json())
      .then((data: SymbolsData) => {
        setSymbols(data.symbols);
        const params = new URLSearchParams(window.location.search);
        const urlSymbol = params.get('symbol');
        if (urlSymbol && data.symbols.includes(urlSymbol)) {
          setSelected(urlSymbol);
        }
      });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);

    Promise.all([
      fetch(`/data/risk/${selected}.json`).then((res) => res.json()),
      fetch(`/data/prices/${selected}.json`).then((res) => res.json()),
    ]).then(([riskData, priceData]) => {
      setRisk(riskData);
      setPrice(priceData);
      setLoading(false);
    });
  }, [selected]);

  if (loading || !risk || !price) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading...</div>
      </div>
    );
  }

  const { metrics } = risk;

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">Risk Analytics</h1>

      <div className="mb-6">
        <SymbolSelector
          symbols={symbols}
          selected={selected}
          onChange={setSelected}
        />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard
          title="Risk Score"
          value={Math.round(metrics.risk_score)}
          subtitle={
            metrics.risk_score < 40
              ? 'Low Risk'
              : metrics.risk_score < 60
              ? 'Moderate'
              : metrics.risk_score < 80
              ? 'High Risk'
              : 'Very High'
          }
        />
        <KPICard
          title="VaR (95%)"
          value={`${metrics.var_95.toFixed(2)}%`}
          subtitle="Daily value at risk"
        />
        <KPICard
          title="Volatility (21d)"
          value={`${metrics.volatility_21d.toFixed(1)}%`}
          subtitle="Annualized"
        />
        <KPICard
          title="Max Drawdown"
          value={`${metrics.max_drawdown.toFixed(2)}%`}
          subtitle="Historical maximum"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Price Chart */}
        <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold text-white mb-4">
            {selected} Price History
          </h2>
          <PriceChart data={price.history} height={350} showGrid />
        </div>

        {/* Risk Metrics */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold text-white mb-4">
            Risk Metrics
          </h2>
          <RiskMetrics data={metrics} />
        </div>
      </div>

      {/* Risk Comparison Table */}
      <div className="mt-8 bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-white mb-4">
          Risk Metric Definitions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h3 className="font-medium text-white mb-2">Value at Risk (VaR)</h3>
            <p className="text-zinc-400">
              The maximum expected loss over a given time period at a specified
              confidence level. VaR 95% means there's a 5% chance losses could
              exceed this amount.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-white mb-2">Expected Shortfall (ES)</h3>
            <p className="text-zinc-400">
              Also known as CVaR, this measures the expected loss when VaR is
              exceeded. It captures tail risk better than VaR alone.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-white mb-2">Volatility</h3>
            <p className="text-zinc-400">
              Annualized standard deviation of returns. Higher volatility means
              larger price swings and greater uncertainty.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-white mb-2">Sharpe Ratio</h3>
            <p className="text-zinc-400">
              Risk-adjusted return measure. Higher values indicate better
              returns per unit of risk taken.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
