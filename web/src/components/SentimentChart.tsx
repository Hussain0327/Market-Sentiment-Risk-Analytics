'use client';

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface SentimentChartProps {
  data: Array<{
    date: string;
    score: number;
    signal?: string;
  }>;
  height?: number;
}

export default function SentimentChart({ data, height = 250 }: SentimentChartProps) {
  // Reverse for chronological order
  const chartData = [...data].reverse();

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
              <stop offset="50%" stopColor="#3b82f6" stopOpacity={0.1} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.3} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fill: '#71717a', fontSize: 11 }}
            tickLine={{ stroke: '#27272a' }}
            axisLine={{ stroke: '#27272a' }}
            tickFormatter={(value) => {
              const date = new Date(value);
              return `${date.getMonth() + 1}/${date.getDate()}`;
            }}
            interval="preserveStartEnd"
            minTickGap={50}
          />
          <YAxis
            domain={[-0.5, 0.5]}
            tick={{ fill: '#71717a', fontSize: 11 }}
            tickLine={{ stroke: '#27272a' }}
            axisLine={{ stroke: '#27272a' }}
            tickFormatter={(value) => value.toFixed(1)}
            width={40}
          />
          <ReferenceLine y={0} stroke="#52525b" strokeDasharray="3 3" />
          <ReferenceLine y={0.2} stroke="#22c55e" strokeDasharray="2 2" opacity={0.5} />
          <ReferenceLine y={-0.2} stroke="#ef4444" strokeDasharray="2 2" opacity={0.5} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #27272a',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#a1a1aa' }}
            formatter={(value: number, name: string) => {
              const signal = value > 0.15 ? 'Bullish' : value < -0.15 ? 'Bearish' : 'Neutral';
              return [value.toFixed(3), `Score (${signal})`];
            }}
            labelFormatter={(label) => new Date(label).toLocaleDateString()}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#sentimentGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
