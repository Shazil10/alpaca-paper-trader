import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { Strategy, StrategyStatus } from '../types';
import { money } from '../hooks/usePortfolio';
import ScrollReveal from './ScrollReveal';
import './Strategies.css';

const FILTERS: Array<{ id: 'all' | StrategyStatus; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'running', label: 'Running' },
  { id: 'research', label: 'Research' },
  { id: 'retired', label: 'Retired' },
];

const STATUS_LABEL: Record<StrategyStatus, string> = {
  running: 'Running',
  research: 'Research',
  retired: 'Retired',
};

export default function Strategies({ strategies }: { strategies: Strategy[] }) {
  const [filter, setFilter] = useState<'all' | StrategyStatus>('all');
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const visible = useMemo(
    () =>
      filter === 'all' ? strategies : strategies.filter((s) => s.status === filter),
    [filter, strategies],
  );

  return (
    <section className="section-shell" id="strategies">
      <ScrollReveal>
        <div className="section-label">02. Strategies</div>
        <h2 className="section-title">Running, research, retired</h2>
        <p className="lede">
          Live sleeves are capital-constrained and monitored. Research sleeves stay
          in notebooks until they clear ops and validation constraints.
        </p>
      </ScrollReveal>

      <div className="status-tabs" role="tablist">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            className={`status-tab ${filter === f.id ? 'active' : ''}`}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="strategy-grid">
        <AnimatePresence mode="popLayout">
          {visible.map((s) => {
            const open = expanded[s.id];
            const rules = open ? s.rules : s.rules.slice(0, 3);
            return (
              <motion.article
                key={s.id}
                className="strategy-card glass"
                layout
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.3 }}
              >
                <div className="card-top">
                  <h3>{s.name}</h3>
                  <span className={`pill ${s.status}`}>{STATUS_LABEL[s.status]}</span>
                </div>
                <p className="thesis">{s.thesis}</p>
                <ul className="rules">
                  {rules.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
                {s.rules.length > 3 && (
                  <button
                    type="button"
                    className="expand-btn mono"
                    onClick={() =>
                      setExpanded((e) => ({ ...e, [s.id]: !e[s.id] }))
                    }
                  >
                    {open ? 'Show less' : `+${s.rules.length - 3} more`}
                  </button>
                )}
                <div className="card-meta mono">
                  {s.budget != null ? (
                    <span>Budget {money(s.budget)}</span>
                  ) : (
                    <span>No live budget</span>
                  )}
                </div>
                {s.note && <p className="card-note">{s.note}</p>}
                <div className="card-links mono">
                  {s.code_url && (
                    <a href={s.code_url} target="_blank" rel="noopener noreferrer">
                      Code
                    </a>
                  )}
                  {s.notebook_url && (
                    <a
                      href={s.notebook_url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Notebook
                    </a>
                  )}
                </div>
              </motion.article>
            );
          })}
        </AnimatePresence>
      </div>
    </section>
  );
}
