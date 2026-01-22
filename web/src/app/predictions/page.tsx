'use client';

import { useState, useEffect } from 'react';
import KPICard from '@/components/KPICard';
import SymbolSelector from '@/components/SymbolSelector';
import type { SymbolsData, PredictionData } from '@/lib/data';

function MetricBar({
  label,
  value,
  max = 1,
  format = 'percent',
}: {
  label: string;
  value: number;
  max?: number;
  format?: 'percent' | 'decimal';
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const displayValue =
    format === 'percent' ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-zinc-400">{label}</span>
        <span className="text-sm font-medium text-white">{displayValue}</span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function PredictionsPage() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>('AAPL');
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [allPredictions, setAllPredictions] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/symbols.json')
      .then((res) => res.json())
      .then(async (data: SymbolsData) => {
        setSymbols(data.symbols);

        // Load all predictions for comparison
        const all = await Promise.all(
          data.symbols.map((s) =>
            fetch(`/data/predictions/${s}.json`).then((res) => res.json())
          )
        );
        setAllPredictions(all);
      });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);

    fetch(`/data/predictions/${selected}.json`)
      .then((res) => res.json())
      .then((data) => {
        setPrediction(data);
        setLoading(false);
      });
  }, [selected]);

  if (loading || !prediction) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading...</div>
      </div>
    );
  }

  const { model, classifier, regressor, features } = prediction;

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">ML Predictions</h1>

      <div className="mb-6">
        <SymbolSelector
          symbols={symbols}
          selected={selected}
          onChange={setSelected}
        />
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard
          title="Training Samples"
          value={model.n_samples}
          subtitle="Walk-forward validation"
        />
        <KPICard
          title="Features Used"
          value={model.n_features}
          subtitle="Technical + sentiment"
        />
        <KPICard
          title="Direction Accuracy"
          value={`${(classifier.accuracy * 100).toFixed(1)}%`}
          subtitle="Classification model"
        />
        <KPICard
          title="Direction (Regressor)"
          value={`${(regressor.direction_accuracy * 100).toFixed(1)}%`}
          subtitle="Return prediction"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Classifier Metrics */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold text-white mb-4">
            Classifier Metrics (Direction)
          </h2>
          <p className="text-sm text-zinc-400 mb-4">
            XGBoost classifier predicting next-day price direction (up/down)
          </p>
          <MetricBar label="AUC Score" value={classifier.auc} />
          <MetricBar label="Accuracy" value={classifier.accuracy} />
          <MetricBar label="Precision" value={classifier.precision} />
          <MetricBar label="Recall" value={classifier.recall} />
          <MetricBar label="F1 Score" value={classifier.f1} />
        </div>

        {/* Regressor Metrics */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold text-white mb-4">
            Regressor Metrics (Returns)
          </h2>
          <p className="text-sm text-zinc-400 mb-4">
            XGBoost regressor predicting next-day returns
          </p>
          <MetricBar
            label="Direction Accuracy"
            value={regressor.direction_accuracy}
          />
          <MetricBar label="RMSE" value={regressor.rmse} max={0.05} format="decimal" />
          <MetricBar label="MAE" value={regressor.mae} max={0.05} format="decimal" />
          <div className="mt-4 pt-4 border-t border-zinc-800">
            <div className="flex items-center justify-between text-sm">
              <span className="text-zinc-400">R² Score</span>
              <span className={`font-medium ${regressor.r2 < 0 ? 'text-red-400' : 'text-white'}`}>
                {regressor.r2.toFixed(4)}
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-500">
              {regressor.r2 < 0
                ? 'Negative R² indicates high prediction difficulty (expected for stock returns)'
                : 'Positive R² indicates some predictive power'}
            </p>
          </div>
        </div>
      </div>

      {/* Top Features */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4">
          Top Model Features
        </h2>
        <div className="flex flex-wrap gap-2">
          {features.map((feature) => (
            <span
              key={feature}
              className="px-3 py-1.5 bg-zinc-800 rounded-lg text-sm text-zinc-300"
            >
              {feature}
            </span>
          ))}
        </div>
      </div>

      {/* Model Comparison */}
      {allPredictions.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold text-white mb-4">
            Cross-Symbol Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className="text-left py-3 px-2 text-zinc-400 font-medium">
                    Symbol
                  </th>
                  <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                    AUC
                  </th>
                  <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                    Accuracy
                  </th>
                  <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                    F1
                  </th>
                  <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                    Dir. Acc
                  </th>
                  <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                    Samples
                  </th>
                </tr>
              </thead>
              <tbody>
                {allPredictions.map((p) => (
                  <tr
                    key={p.symbol}
                    className={`border-b border-zinc-800/50 ${
                      p.symbol === selected ? 'bg-blue-500/10' : 'hover:bg-zinc-800/30'
                    }`}
                  >
                    <td className="py-3 px-2 font-medium text-white">
                      {p.symbol}
                    </td>
                    <td className="py-3 px-2 text-right text-zinc-300">
                      {(p.classifier.auc * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right text-zinc-300">
                      {(p.classifier.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right text-zinc-300">
                      {(p.classifier.f1 * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right text-zinc-300">
                      {(p.regressor.direction_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right text-zinc-300">
                      {p.model.n_samples}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Trained timestamp */}
      <p className="mt-6 text-xs text-zinc-600">
        Model trained: {new Date(model.trained_at).toLocaleString()}
      </p>
    </div>
  );
}
