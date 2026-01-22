'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

interface PriceChartProps {
  data: Array<{
    date: string;
    close: number;
    high?: number;
    low?: number;
  }>;
  height?: number;
  showGrid?: boolean;
}

export default function PriceChart({ data, height = 300, showGrid = false }: PriceChartProps) {
  // Reverse data for chronological order
  const chartData = [...data].reverse();

  const minPrice = Math.min(...chartData.map((d) => d.low ?? d.close));
  const maxPrice = Math.max(...chartData.map((d) => d.high ?? d.close));
  const padding = (maxPrice - minPrice) * 0.1;

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          )}
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
            domain={[minPrice - padding, maxPrice + padding]}
            tick={{ fill: '#71717a', fontSize: 11 }}
            tickLine={{ stroke: '#27272a' }}
            axisLine={{ stroke: '#27272a' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
            width={55}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#18181b',
              border: '1px solid #27272a',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#a1a1aa' }}
            formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
            labelFormatter={(label) => new Date(label).toLocaleDateString()}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#3b82f6' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
