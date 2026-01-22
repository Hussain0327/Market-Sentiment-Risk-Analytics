'use client';

import { useState, useEffect } from 'react';
import KPICard from '@/components/KPICard';
import SentimentChart from '@/components/SentimentChart';
import SymbolSelector from '@/components/SymbolSelector';
import type { SymbolsData, SentimentData, PriceData } from '@/lib/data';

export default function SentimentPage() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>('AAPL');
  const [sentiment, setSentiment] = useState<SentimentData | null>(null);
  const [price, setPrice] = useState<PriceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/symbols.json')
      .then((res) => res.json())
      .then((data: SymbolsData) => {
        setSymbols(data.symbols);
        // Check URL for symbol param
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
      fetch(`/data/sentiment/${selected}.json`).then((res) => res.json()),
      fetch(`/data/prices/${selected}.json`).then((res) => res.json()),
    ]).then(([sentimentData, priceData]) => {
      setSentiment(sentimentData);
      setPrice(priceData);
      setLoading(false);
    });
  }, [selected]);

  if (loading || !sentiment || !price) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading...</div>
      </div>
    );
  }

  const { latest, history } = sentiment;

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">Sentiment Analysis</h1>

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
          title="Sentiment Score"
          value={latest.score.toFixed(3)}
          subtitle={`Confidence: ${(latest.confidence * 100).toFixed(1)}%`}
        />
        <KPICard
          title="Signal"
          value={latest.signal.charAt(0).toUpperCase() + latest.signal.slice(1)}
          subtitle={`Strength: ${(latest.signal_strength * 100).toFixed(1)}%`}
        />
        <KPICard
          title="Articles Analyzed"
          value={latest.article_count}
          subtitle={`As of ${latest.date}`}
        />
        <KPICard
          title="Bullish/Bearish Ratio"
          value={`${(latest.bullish_ratio * 100).toFixed(0)}% / ${(latest.bearish_ratio * 100).toFixed(0)}%`}
          subtitle="Article sentiment split"
        />
      </div>

      {/* Chart */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4">
          {selected} Sentiment Trend
        </h2>
        <SentimentChart data={history} height={350} />
        <div className="mt-4 flex items-center justify-center gap-6 text-xs text-zinc-500">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span>Bullish threshold (0.2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-red-500"></div>
            <span>Bearish threshold (-0.2)</span>
          </div>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-white mb-4">
          Recent Sentiment History
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="text-left py-3 px-2 text-zinc-400 font-medium">
                  Date
                </th>
                <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                  Score
                </th>
                <th className="text-center py-3 px-2 text-zinc-400 font-medium">
                  Signal
                </th>
                <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                  Confidence
                </th>
                <th className="text-right py-3 px-2 text-zinc-400 font-medium">
                  Articles
                </th>
              </tr>
            </thead>
            <tbody>
              {history.slice(0, 10).map((row) => (
                <tr
                  key={row.date}
                  className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                >
                  <td className="py-3 px-2 text-white">{row.date}</td>
                  <td
                    className={`py-3 px-2 text-right font-medium ${
                      row.score > 0.15
                        ? 'text-green-400'
                        : row.score < -0.15
                        ? 'text-red-400'
                        : 'text-zinc-300'
                    }`}
                  >
                    {row.score.toFixed(3)}
                  </td>
                  <td className="py-3 px-2 text-center">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        row.signal === 'bullish'
                          ? 'bg-green-500/20 text-green-400'
                          : row.signal === 'bearish'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-zinc-700 text-zinc-300'
                      }`}
                    >
                      {row.signal}
                    </span>
                  </td>
                  <td className="py-3 px-2 text-right text-zinc-300">
                    {(row.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-2 text-right text-zinc-300">
                    {row.article_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
