'use client';

import { clsx } from 'clsx';
import type { RiskData } from '@/lib/data';

interface RiskMetricsProps {
  data: RiskData['metrics'];
  compact?: boolean;
}

function MetricRow({
  label,
  value,
  suffix = '',
  danger,
}: {
  label: string;
  value: number;
  suffix?: string;
  danger?: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
      <span className="text-sm text-zinc-400">{label}</span>
      <span
        className={clsx(
          'text-sm font-medium',
          danger ? 'text-red-400' : 'text-white'
        )}
      >
        {value.toFixed(2)}{suffix}
      </span>
    </div>
  );
}

function RiskGauge({ score }: { score: number }) {
  const getColor = (score: number) => {
    if (score < 40) return 'bg-green-500';
    if (score < 60) return 'bg-yellow-500';
    if (score < 80) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getLabel = (score: number) => {
    if (score < 40) return 'Low Risk';
    if (score < 60) return 'Moderate';
    if (score < 80) return 'High Risk';
    return 'Very High';
  };

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-zinc-400">Risk Score</span>
        <span className="text-lg font-semibold text-white">{Math.round(score)}</span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all', getColor(score))}
          style={{ width: `${Math.min(100, score)}%` }}
        />
      </div>
      <p className={clsx('mt-1 text-xs', getColor(score).replace('bg-', 'text-'))}>
        {getLabel(score)}
      </p>
    </div>
  );
}

export default function RiskMetrics({ data, compact = false }: RiskMetricsProps) {
  if (compact) {
    return (
      <div className="space-y-2">
        <RiskGauge score={data.risk_score} />
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-zinc-400">VaR (95%)</span>
            <p className="text-red-400 font-medium">{data.var_95.toFixed(2)}%</p>
          </div>
          <div>
            <span className="text-zinc-400">Volatility</span>
            <p className="text-white font-medium">{data.volatility_21d.toFixed(1)}%</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <RiskGauge score={data.risk_score} />

      <div className="mt-4 space-y-1">
        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
          Value at Risk
        </h4>
        <MetricRow label="VaR (95%)" value={data.var_95} suffix="%" danger />
        <MetricRow label="VaR (99%)" value={data.var_99} suffix="%" danger />
        <MetricRow label="ES (95%)" value={data.es_95} suffix="%" danger />
        <MetricRow label="ES (99%)" value={data.es_99} suffix="%" danger />
      </div>

      <div className="mt-4 space-y-1">
        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
          Volatility
        </h4>
        <MetricRow label="21-Day" value={data.volatility_21d} suffix="%" />
        <MetricRow label="63-Day" value={data.volatility_63d} suffix="%" />
      </div>

      <div className="mt-4 space-y-1">
        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
          Drawdown
        </h4>
        <MetricRow label="Current" value={data.current_drawdown} suffix="%" danger={data.current_drawdown < -5} />
        <MetricRow label="Maximum" value={data.max_drawdown} suffix="%" danger />
      </div>

      <div className="mt-4 space-y-1">
        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
          Returns
        </h4>
        <MetricRow label="Sharpe Ratio" value={data.sharpe_ratio} />
      </div>
    </div>
  );
}
