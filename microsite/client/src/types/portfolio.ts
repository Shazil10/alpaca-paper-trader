export type StrategyStatus = "Running" | "Research" | "Retired";

export interface StrategyLink {
  label: string;
  url: string;
}

/** Which schematic explains how this sleeve decides. Omit for none. */
export type MechanismKind = "inverse-vol" | "sector-tilt" | "pullback-gauge";

export interface Strategy {
  id: string;
  name: string;
  status: StrategyStatus;
  budget: number | null;
  /** Short label used for capital attribution; matches position and PnL rows. */
  short: string;
  /** One jargon-free clause: what this sleeve looks for, in ordinary words. */
  plain: string;
  chips?: string[];
  thesis: string;
  sizing?: string;
  holdings?: string[];
  note?: string;
  /** Drives the asymmetric board: the featured sleeve takes the tall column. */
  featured?: boolean;
  /** Shown only on the featured card, where there is room for detail. */
  rules?: string[];
  /** A single citation — the paper, book, or write-up behind the approach. */
  evidence?: StrategyLink;
  /** Illustrative diagram of the decision rule, never of results. */
  mechanism?: MechanismKind;
}

export interface Position {
  strategy: string;
  symbol: string;
  qty: number;
  notional: number;
  notional_display: string;
  note: string;
  discretionary?: boolean;
}

export interface SleevePnl {
  label: string;
  pnl: number;
}

/** Caps are summed from the strategy list, never stored alongside it, so a
 *  promoted or retired sleeve cannot leave a stale total behind. */
export interface PortfolioData {
  as_of: string;
  disclaimer: string;
  repo: string;
  portfolio_home: string;
  strategies: Strategy[];
  positions: Position[];
  sleeve_realized_pnl: SleevePnl[];
}
