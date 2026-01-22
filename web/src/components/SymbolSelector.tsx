'use client';

import { clsx } from 'clsx';

interface SymbolSelectorProps {
  symbols: string[];
  selected: string;
  onChange: (symbol: string) => void;
}

export default function SymbolSelector({
  symbols,
  selected,
  onChange,
}: SymbolSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {symbols.map((symbol) => (
        <button
          key={symbol}
          onClick={() => onChange(symbol)}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            selected === symbol
              ? 'bg-blue-600 text-white'
              : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
          )}
        >
          {symbol}
        </button>
      ))}
    </div>
  );
}
