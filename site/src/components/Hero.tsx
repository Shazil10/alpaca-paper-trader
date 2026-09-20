import { motion, useReducedMotion } from 'framer-motion';
import './Hero.css';

const CHIPS = ['Python', 'Alpaca paper', 'GitHub Actions', 'Lifetime budgets', 'Order attribution'];

export default function Hero() {
  const reduce = useReducedMotion();

  return (
    <section className="hero" id="top">
      <motion.p
        className="hero-eyebrow mono"
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.05 }}
      >
        Alpaca paper · research → production
      </motion.p>
      <motion.h1
        initial={reduce ? false : { opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.12 }}
      >
        Multi-strategy
        <span>paper trading system</span>
      </motion.h1>
      <motion.p
        className="hero-body"
        initial={reduce ? false : { opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, delay: 0.22 }}
      >
        Strategies own selection and sizing. A single orchestrator enforces capital
        caps, attribution, and execution safety. The whole loop runs unattended on
        GitHub Actions — built to show process, not a flashy equity curve.
      </motion.p>
      <motion.div
        className="hero-chips"
        initial={reduce ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.32 }}
      >
        {CHIPS.map((c) => (
          <span className="chip" key={c}>
            {c}
          </span>
        ))}
      </motion.div>
      <motion.div
        className="hero-actions"
        initial={reduce ? false : { opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <a className="btn" href="#architecture">
          See the system
        </a>
        <a
          className="btn btn-ghost"
          href="https://github.com/Shazil10/alpaca-paper-trader"
          target="_blank"
          rel="noopener noreferrer"
        >
          View code
        </a>
      </motion.div>
    </section>
  );
}
