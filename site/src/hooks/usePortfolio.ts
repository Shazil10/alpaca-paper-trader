import { useEffect, useState } from 'react';
import type { PortfolioData } from '../types';

export function usePortfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/data/portfolio.json', { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to load portfolio.json');
        const json = (await res.json()) as PortfolioData;
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Load error');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, error, loading };
}

export const money = (n: number | null | undefined) =>
  n == null
    ? '—'
    : new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0,
      }).format(n);
