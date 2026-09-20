import { Activity, Cpu, ShieldCheck, Timer } from "lucide-react";
import { liveSleeves, money, researchSleeves, totalCap } from "@/lib/sleeves";
import type { Strategy } from "@/types/portfolio";

interface HeroProps {
  strategies: Strategy[];
}

/** Every sleeve routes through one gate, so the markers ring the centre and
 *  their links converge on it. Positions are generated from however many
 *  sleeves exist rather than from a fixed list of slots. */
const GATE = { x: 50, y: 46 };
const RADIUS = { x: 27, y: 26 };
const START_ANGLE = -68;

function ringPosition(index: number, count: number) {
  const angle = ((START_ANGLE + (360 / count) * index) * Math.PI) / 180;
  return {
    x: GATE.x + Math.cos(angle) * RADIUS.x,
    y: GATE.y + Math.sin(angle) * RADIUS.y,
  };
}

export default function Hero({ strategies }: HeroProps) {
  const live = liveSleeves(strategies);
  const research = researchSleeves(strategies);
  const capital = totalCap(strategies);

  const stats = [
    { label: "Capital capped", value: money(capital), icon: ShieldCheck },
    { label: "Infrastructure", value: "Scheduled CI", icon: Cpu },
    { label: "Cadence", value: "Every weekday", icon: Timer },
    { label: "Sleeves", value: `${live.length} live / ${research.length} queued`, icon: Activity },
  ];

  const nodes = live.map((sleeve, index) => ({
    label: sleeve.short,
    cap: sleeve.budget ? `$${Math.round(sleeve.budget / 1000)}k cap` : null,
    ...ringPosition(index, live.length),
  }));

  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero-noise" aria-hidden="true" />
      <div className="hero-inner">
        <div className="hero-copy">
          <p className="eyebrow"><i />Trading live on a paper account · 5 months unattended</p>
          <h1 id="hero-title">
            Automated multi-strategy trading engine
          </h1>
          <p className="hero-body">
            A trading system that decides and places its own orders every
            weekday, with no one at the keyboard. Independent strategy sleeves
            propose trades; one control layer decides how much money each may
            actually use.
          </p>

          {/* PERSONAL NOTE: replace the placeholder below with one honest sentence
              on why you built this — the real reason, not a recruiting reason. */}
          <p className="personal-note">
            <span className="note-tag">Your words</span>
            [One sentence: why did you actually build this? What were you trying
            to find out?]
          </p>

          <ul className="stat-grid">
            {stats.map(({ label, value, icon: Icon }) => (
              <li key={label}>
                <Icon size={15} strokeWidth={1.6} aria-hidden="true" />
                <span>{label}</span>
                <strong>{value}</strong>
              </li>
            ))}
          </ul>
        </div>

        <aside className="hero-visual" aria-hidden="true">
          <span className="visual-corner--top" />
          <div className="system-route">
            UNIVERSE <i /> SLEEVES <i /> CAPITAL CONTROL <i /> REPORT
          </div>
          <div className="hero-lattice">
            <i /><i /><i />
          </div>

          <svg className="hero-links" viewBox="0 0 100 100" preserveAspectRatio="none">
            {nodes.map(({ label, x, y }) => (
              <line key={label} x1={x} y1={y} x2={GATE.x} y2={GATE.y} vectorEffect="non-scaling-stroke" />
            ))}
          </svg>

          <ul className="hero-nodes">
            {nodes.map(({ label, cap, x, y }) => (
              <li
                key={label}
                className={x < GATE.x ? "is-left" : ""}
                style={{ left: `${x}%`, top: `${y}%` }}
              >
                <i />
                <span>
                  {label}
                  {cap && <small>{cap}</small>}
                </span>
              </li>
            ))}
            <li className="hero-gate" style={{ left: `${GATE.x}%`, top: `${GATE.y}%` }}>
              <i />
              <span>Capital control</span>
            </li>
          </ul>

          <div className="hero-instrument">
            <div>
              <span className="instrument-kicker">Capital under system control</span>
              <strong>{money(capital)}</strong>
            </div>
            <span className="status-light"><i />NO MANUAL STEP</span>
          </div>
        </aside>
      </div>

      <p className="scroll-cue">
        <i />Scroll to follow one order
      </p>
    </section>
  );
}
