import { useEffect, useMemo, useRef } from 'react';
import type { PortfolioData } from '../types';
import { money } from '../hooks/usePortfolio';
import ScrollReveal from './ScrollReveal';
import './BookSnapshot.css';

export default function BookSnapshot({ data }: { data: PortfolioData }) {
  const barsRef = useRef<HTMLDivElement>(null);

  const budgetRows = useMemo(() => {
    const deployed: Record<string, number> = {};
    for (const p of data.positions || []) {
      if (p.strategy_id === 'discretionary') continue;
      deployed[p.strategy] = (deployed[p.strategy] || 0) + (p.notional || 0);
    }
    const map: Array<{ name: string; cap: number; used: number }> = [
      {
        name: 'Clenow Trend',
        cap: data.budgets['strategies.momentum.clenow_trend'],
        used: deployed['Clenow Trend'] || 0,
      },
      {
        name: 'Ranked Asset Alloc',
        cap: data.budgets['strategies.ranks.ranked_asset_alloc'],
        used: deployed['Ranked Asset Alloc'] || 0,
      },
      {
        name: 'High Pullback Reversion',
        cap: data.budgets['strategies.mean_reversion.high_pullback_reversion'],
        used: deployed['High Pullback Reversion'] || 0,
      },
    ];
    return map.filter((r) => r.cap);
  }, [data]);

  useEffect(() => {
    const root = barsRef.current;
    if (!root) return;
    const fills = root.querySelectorAll<HTMLElement>('.budget-fill');
    requestAnimationFrame(() => {
      fills.forEach((el) => {
        el.style.width = `${el.dataset.width}%`;
      });
    });
  }, [budgetRows]);

  return (
    <section className="section-shell book-section" id="book">
      <ScrollReveal>
        <div className="section-label">03. Book snapshot</div>
        <h2 className="section-title">Positions by strategy</h2>
        <p className="as-of mono">
          As of {data.as_of} (EOD snapshot · illustrative)
        </p>
        <p className="disclaimer">{data.disclaimer}</p>
      </ScrollReveal>

      <div className="table-wrap glass">
        <table className="book">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Symbol</th>
              <th>Qty</th>
              <th>Notional</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {data.positions.map((p, i) => (
              <tr
                key={`${p.symbol}-${i}`}
                className={p.strategy_id === 'discretionary' ? 'disc' : ''}
              >
                <td>{p.strategy}</td>
                <td className="sym">{p.symbol}</td>
                <td>{p.qty}</td>
                <td>{money(p.notional)}</td>
                <td>{p.note || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="budget-bars" ref={barsRef} aria-label="Lifetime budget caps">
        {budgetRows.map((r) => {
          const used = Math.min(r.used, r.cap);
          const pct = Math.round((used / r.cap) * 100);
          return (
            <div className="budget-row" key={r.name}>
              <span>{r.name}</span>
              <div className="budget-track">
                <div className="budget-fill" data-width={pct} />
              </div>
              <span>
                {pct}% / {money(r.cap)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
