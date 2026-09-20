import { useEffect, useRef } from 'react';
import type { SleevePnl } from '../types';
import { money } from '../hooks/usePortfolio';
import ScrollReveal from './ScrollReveal';
import './SleeveContribution.css';

export default function SleeveContribution({ rows }: { rows: SleevePnl[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const max = Math.max(...rows.map((r) => Math.abs(r.pnl)), 1);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    requestAnimationFrame(() => {
      root.querySelectorAll<HTMLElement>('.sleeve-fill').forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      });
    });
  }, [rows]);

  return (
    <section className="section-shell" id="contribution">
      <ScrollReveal>
        <div className="section-label">04. Sleeve contribution</div>
        <h2 className="section-title">Where live PnL came from</h2>
        <p className="lede">
          Muted view of realized contribution by algorithmic sleeve. Short paper
          sample — process evidence, not a performance pitch.
        </p>
      </ScrollReveal>

      <div className="sleeve-chart glass">
        <div className="sleeve-bars" ref={ref}>
          {rows.map((r) => {
            const pos = r.pnl >= 0;
            const pct = Math.round((Math.abs(r.pnl) / max) * 100);
            return (
              <div className="sleeve-item" key={r.id}>
                <span className="label">{r.label}</span>
                <div className="sleeve-track">
                  <div
                    className={`sleeve-fill ${pos ? 'pos' : 'neg'}`}
                    data-width={pct}
                  />
                </div>
                <span className={`val ${pos ? 'pos' : 'neg'}`}>
                  {pos ? '+' : '−'}
                  {money(Math.abs(r.pnl))}
                </span>
              </div>
            );
          })}
        </div>
        <p className="chart-caption mono">
          Paper account · discretionary excluded · approximate
        </p>
      </div>
    </section>
  );
}
