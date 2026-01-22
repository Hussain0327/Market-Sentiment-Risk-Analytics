import {
  getSymbolsServer,
  getPriceDataServer,
  getSentimentDataServer,
  getRiskDataServer,
  formatNumber,
  formatPercent,
  formatVolume,
} from '@/lib/data';
import KPICard from '@/components/KPICard';
import Link from 'next/link';

async function getOverviewData() {
  const { symbols } = await getSymbolsServer();

  const data = await Promise.all(
    symbols.map(async (symbol) => {
      const [price, sentiment, risk] = await Promise.all([
        getPriceDataServer(symbol),
        getSentimentDataServer(symbol),
        getRiskDataServer(symbol),
      ]);
      return { symbol, price, sentiment, risk };
    })
  );

  return data;
}

export default async function OverviewPage() {
  const data = await getOverviewData();

  // Calculate portfolio summary
  const avgSentiment =
    data.reduce((sum, d) => sum + d.sentiment.latest.score, 0) / data.length;
  const avgRisk =
    data.reduce((sum, d) => sum + d.risk.metrics.risk_score, 0) / data.length;
  const totalVolume = data.reduce((sum, d) => sum + d.price.latest.volume, 0);
  const bullishCount = data.filter(
    (d) => d.sentiment.latest.signal === 'bullish'
  ).length;

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">Portfolio Overview</h1>

      {/* Summary KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard
          title="Tracked Symbols"
          value={data.length}
          subtitle="Tech sector focus"
        />
        <KPICard
          title="Avg Sentiment"
          value={avgSentiment.toFixed(3)}
          subtitle={avgSentiment > 0.1 ? 'Bullish' : avgSentiment < -0.1 ? 'Bearish' : 'Neutral'}
        />
        <KPICard
          title="Bullish Signals"
          value={`${bullishCount}/${data.length}`}
          subtitle="Stocks with positive sentiment"
        />
        <KPICard
          title="Avg Risk Score"
          value={Math.round(avgRisk)}
          subtitle={avgRisk < 50 ? 'Low-Moderate' : 'Elevated'}
        />
      </div>

      {/* Symbol Cards */}
      <h2 className="text-lg font-semibold text-white mb-4">All Symbols</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {data.map(({ symbol, price, sentiment, risk }) => (
          <div
            key={symbol}
            className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors"
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-white">{symbol}</h3>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  sentiment.latest.signal === 'bullish'
                    ? 'bg-green-500/20 text-green-400'
                    : sentiment.latest.signal === 'bearish'
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-zinc-700 text-zinc-300'
                }`}
              >
                {sentiment.latest.signal}
              </span>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold text-white">
                  ${formatNumber(price.latest.close)}
                </p>
                <p
                  className={`text-sm ${
                    price.stats.total_return >= 0
                      ? 'text-green-400'
                      : 'text-red-400'
                  }`}
                >
                  {formatPercent(price.stats.total_return)} total
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <p className="text-zinc-500">Sentiment</p>
                  <p className="text-white font-medium">
                    {sentiment.latest.score.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500">Risk Score</p>
                  <p
                    className={`font-medium ${
                      risk.metrics.risk_score < 50
                        ? 'text-green-400'
                        : risk.metrics.risk_score < 70
                        ? 'text-yellow-400'
                        : 'text-red-400'
                    }`}
                  >
                    {Math.round(risk.metrics.risk_score)}
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500">VaR (95%)</p>
                  <p className="text-red-400 font-medium">
                    {risk.metrics.var_95.toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500">Volume</p>
                  <p className="text-white font-medium">
                    {formatVolume(price.latest.volume)}
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-zinc-800 flex gap-2">
              <Link
                href={`/sentiment?symbol=${symbol}`}
                className="flex-1 text-center py-1.5 text-xs font-medium text-zinc-400 hover:text-white bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
              >
                Sentiment
              </Link>
              <Link
                href={`/risk?symbol=${symbol}`}
                className="flex-1 text-center py-1.5 text-xs font-medium text-zinc-400 hover:text-white bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
              >
                Risk
              </Link>
            </div>
          </div>
        ))}
      </div>

      {/* Data freshness */}
      <p className="mt-8 text-xs text-zinc-600">
        Last updated: {data[0]?.price.latest.date}
      </p>
    </div>
  );
}
