export type StrategyStatus = 'running' | 'research' | 'retired';

export interface Strategy {
  id: string;
  name: string;
  status: StrategyStatus;
  module: string | null;
  budget: number | null;
  code_url: string | null;
  notebook_url: string | null;
  thesis: string;
  rules: string[];
  note: string;
}

export interface Position {
  strategy: string;
  strategy_id: string;
  symbol: string;
  qty: number;
  notional: number;
  note: string;
}

export interface SleevePnl {
  id: string;
  label: string;
  pnl: number;
}

export interface PortfolioData {
  as_of: string;
  disclaimer: string;
  repo: string;
  portfolio_home: string;
  budgets: Record<string, number>;
  sleeve_realized_pnl: SleevePnl[];
  strategies: Strategy[];
  positions: Position[];
}
