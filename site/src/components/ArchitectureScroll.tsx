import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import ScrollReveal from './ScrollReveal';
import './ArchitectureScroll.css';

gsap.registerPlugin(ScrollTrigger);

const STEPS = [
  {
    id: 'research',
    badge: '01 / RESEARCH',
    title: 'Ideas earn a budget first.',
    body: 'Hypotheses are tested in notebooks. Only rules that survive research get a live adapter and dollars — live metrics stay separate from backtest Sharpes.',
    chip: 'Notebooks first',
    sub: 'parameters · walk-forward',
    bullets: [
      'Parameter sets locked from studies (e.g. R2 v2)',
      'Research sleeves stay capital-free until ready',
      'No confusing paper PnL with research Sharpe',
    ],
  },
  {
    id: 'strategies',
    badge: '02 / STRATEGIES',
    title: 'One contract, three engines.',
    body: 'Every live sleeve exposes generate_signals(budget, strategy_id, held_symbols). Inside: momentum, regime rotation, or pullback mean reversion.',
    chip: 'Shared interface',
    sub: 'momentum · rotation · MR',
    bullets: [
      'Clenow: cross-sectional trend + inverse-vol sizing',
      'Ranked: monthly V4/V8 blend + optional DAF leverage',
      'Pullback MR: deep 52-week drawdowns with quality gates',
    ],
  },
  {
    id: 'orchestrator',
    badge: '03 / ORCHESTRATOR',
    title: 'Capital and ownership, not just signals.',
    body: 'trade.py turns intents into safe broker actions — lifetime caps, attribution, sell-before-buy, and a cash reserve.',
    chip: 'trade.py',
    sub: 'budgets · attribution',
    bullets: [
      'Lifetime budget caps; sells recycle capacity',
      'Fills tagged strategies.*:uuid',
      'Per-strategy holdings from history ∩ positions',
      'SELL first, refresh, then BUY · 10% cash reserve',
    ],
  },
  {
    id: 'ops',
    badge: '04 / OPS',
    title: 'Unattended runs with an audit trail.',
    body: 'Weekday GitHub Actions: universe → trade → report. DST-aware gates, daily artifacts, and a manual liquidation path when orders stick.',
    chip: 'GitHub Actions',
    sub: 'cron · reports · liquidate',
    bullets: [
      'Dual cron + ET time gate',
      'CSV / MD / HTML order tape with FIFO PnL',
      'Emergency close workflow for paper positions',
    ],
  },
];

export default function ArchitectureScroll() {
  const sectionRef = useRef<HTMLElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef(0);
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);
  const reduce = useReducedMotion();

  const step = useMemo(() => STEPS[active], [active]);

  useEffect(() => {
    if (reduce) return;
    const section = sectionRef.current;
    const pin = pinRef.current;
    if (!section || !pin) return;

    const st = ScrollTrigger.create({
      trigger: section,
      start: 'top top',
      end: () => `+=${Math.max(window.innerHeight * 2.8, 2200)}`,
      pin: pin,
      scrub: 0.65,
      anticipatePin: 1,
      onUpdate: (self) => {
        progressRef.current = self.progress;
        setProgress(self.progress);
        const idx = Math.min(
          STEPS.length - 1,
          Math.floor(self.progress * STEPS.length + 1e-6),
        );
        setActive(idx);
      },
    });

    const onResize = () => ScrollTrigger.refresh();
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      st.kill();
    };
  }, [reduce]);

  const onSelect = (i: number) => {
    if (!reduce) return;
    setActive(i);
    setProgress((i + 0.5) / STEPS.length);
  };

  const spineFill = Math.min(1, Math.max(0, progress));

  return (
    <section className="arch section-shell" id="architecture" ref={sectionRef}>
      <ScrollReveal>
        <div className="section-label">01. Architecture</div>
        <h2 className="section-title">Research to a scheduled book</h2>
        <p className="lede">
          Scroll through the loop. A sticky story card tracks the active stage while
          the spine lights up — research, strategies, orchestration, ops.
        </p>
      </ScrollReveal>

      <div className="arch-pin" ref={pinRef}>
        <div className="arch-stage">
          <div className="arch-card-col">
            <AnimatePresence mode="wait">
              <motion.article
                key={step.id}
                className="arch-card glass"
                initial={reduce ? false : { opacity: 0, y: 18, filter: 'blur(4px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={reduce ? undefined : { opacity: 0, y: -12, filter: 'blur(4px)' }}
                transition={{ duration: 0.35, ease: [0.645, 0.045, 0.355, 1] }}
              >
                <div className="arch-card-top">
                  <span className="arch-badge mono">{step.badge}</span>
                  <span className="arch-accent-line" aria-hidden="true" />
                </div>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
                <ul>
                  {step.bullets.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              </motion.article>
            </AnimatePresence>
          </div>

          <div className="arch-timeline" aria-label="Architecture stages">
            <div className="arch-spine">
              <div className="arch-spine-track" />
              <div
                className="arch-spine-fill"
                style={{ height: `${spineFill * 100}%` }}
              />
              <div
                className="arch-packet"
                style={{ top: `calc(${spineFill * 100}% - 6px)` }}
              />
            </div>

            <ol className="arch-nodes">
              {STEPS.map((s, i) => {
                const on = i === active;
                const done = i < active;
                return (
                  <li
                    key={s.id}
                    className={`arch-node ${on ? 'active' : ''} ${done ? 'done' : ''}`}
                  >
                    <button
                      type="button"
                      className="arch-node-btn"
                      onClick={() => onSelect(i)}
                      aria-current={on ? 'step' : undefined}
                    >
                      <span className={`arch-dot ${on ? 'glow' : ''}`} />
                      <span className="arch-node-card glass">
                        <span className="arch-node-title">{s.chip}</span>
                        <span className="arch-node-sub mono">{s.sub}</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>
    </section>
  );
}
