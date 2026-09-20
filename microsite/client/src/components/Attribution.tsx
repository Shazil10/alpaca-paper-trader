import { useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import SectionHeader from "./SectionHeader";
import { compactMoney, money } from "@/lib/sleeves";
import type { Position, SleevePnl } from "@/types/portfolio";

interface AttributionProps {
  positions: Position[];
  budgets: Record<string, number>;
  pnl: SleevePnl[];
  asOf: string;
  disclaimer: string;
}

const percent = (n: number) => `${Math.abs(n) < 1 ? n.toFixed(1) : Math.round(n)}%`;

export default function Attribution({
  positions,
  budgets,
  pnl,
  asOf,
  disclaimer,
}: AttributionProps) {
  const reduceMotion = useReducedMotion();
  const panelsRef = useRef<HTMLDivElement>(null);
  const inView = useInView(panelsRef, { once: true, amount: 0.2 });
  // Bars carry their real width at rest and the animation only replays them
  // from zero, so a missing in-view signal costs the animation, not the data.
  const draw = !reduceMotion && inView ? "is-drawing" : "";

  const deployed = Object.fromEntries(
    Object.keys(budgets).map((name) => [
      name,
      positions
        .filter((p) => p.strategy === name && !p.discretionary)
        .reduce((sum, p) => sum + p.notional, 0),
    ]),
  );
  const maxAbs = Math.max(...pnl.map((r) => Math.abs(r.pnl)), 1);

  return (
    <section id="portfolio" className="section attribution-section">
      <div className="section-inner">
        <SectionHeader
          label="03. Book"
          title="Where the money actually is"
          lede="A snapshot at the end of the trading day: how much of each sleeve's allowance is currently in use, and what it has made or lost on trades it has already closed."
        />

        <div className="book-meta">
          <p className="micro-label">Automated sleeves only</p>
          <span className="metric">AS OF {asOf}</span>
        </div>

        <div ref={panelsRef} className={`alloc-panel ${draw}`}>
          <div className="panel-caption">
            <h3>How much of each allowance is in use</h3>
            <code>SELLING FREES ROOM BACK UP</code>
          </div>
          <div className="alloc-list">
            {Object.entries(budgets).map(([name, cap]) => {
              const used = deployed[name] ?? 0;
              const pct = Math.round((used / cap) * 100);
              return (
                <div key={name}>
                  <div className="alloc-head">
                    <span>{name}</span>
                    <strong className="metric">
                      {money(used)} <small>of {money(cap)} · {pct}%</small>
                    </strong>
                  </div>
                  <div className="budget-track">
                    <i
                      role="meter"
                      aria-label={`${name} capital in use`}
                      aria-valuenow={pct}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className={`pnl-panel ${draw}`}>
          <div className="panel-caption">
            <h3>Made and lost on closed trades</h3>
            <code>PAPER MONEY · NOT A TRACK RECORD</code>
          </div>
          <ul className="pnl-rows">
            {pnl.map((row) => {
              const neg = row.pnl < 0;
              const cap = budgets[row.label];
              const width = Math.max((Math.abs(row.pnl) / maxAbs) * 88, 3);
              return (
                <li key={row.label}>
                  <span className="pnl-label">{row.label}</span>
                  <div className="pnl-plot" aria-hidden="true">
                    <i
                      className={neg ? "is-negative" : "is-positive"}
                      style={
                        neg
                          ? { right: "92%", width: `${width / 8}%` }
                          : { width: `${width}%` }
                      }
                    />
                  </div>
                  <div className="pnl-figures">
                    <strong className={neg ? "is-negative" : "is-positive"}>
                      {neg ? "−" : "+"}
                      {money(Math.abs(row.pnl))}
                    </strong>
                    {cap ? (
                      <small>
                        {neg ? "−" : "+"}
                        {percent(Math.abs((row.pnl / cap) * 100))} of {compactMoney(cap)}
                      </small>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
          <div className="attribution-footnote">
            <span><i className="mint-key" />MADE</span>
            <span><i className="red-key" />LOST</span>
            <span className="metric" style={{ marginLeft: "auto" }}>CLOSED TRADES ONLY</span>
          </div>
        </div>

        <p className="disclaimer">{disclaimer}</p>
      </div>
    </section>
  );
}
