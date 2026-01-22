import { clsx } from 'clsx';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: number;
  changeLabel?: string;
  icon?: React.ReactNode;
  className?: string;
}

export default function KPICard({
  title,
  value,
  subtitle,
  change,
  changeLabel,
  icon,
  className,
}: KPICardProps) {
  const isPositive = change !== undefined && change >= 0;

  return (
    <div
      className={clsx(
        'bg-zinc-900 border border-zinc-800 rounded-xl p-5',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-zinc-400">{title}</p>
          <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
          {subtitle && (
            <p className="mt-1 text-xs text-zinc-500">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className="p-2 bg-zinc-800 rounded-lg">
            {icon}
          </div>
        )}
      </div>

      {change !== undefined && (
        <div className="mt-3 flex items-center gap-1">
          <span
            className={clsx(
              'text-sm font-medium',
              isPositive ? 'text-green-500' : 'text-red-500'
            )}
          >
            {isPositive ? '+' : ''}{change.toFixed(2)}%
          </span>
          {changeLabel && (
            <span className="text-xs text-zinc-500">{changeLabel}</span>
          )}
        </div>
      )}
    </div>
  );
}
