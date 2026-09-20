/** Every count, cap, and total on the page is derived here from the strategy
 *  list. Nothing about "how many sleeves exist" is written into a component,
 *  so promoting or retiring a sleeve is a data edit and nothing else. */
import type { Strategy } from "@/types/portfolio";

export const liveSleeves = (strategies: Strategy[]) =>
  strategies.filter((s) => s.status === "Running");

export const researchSleeves = (strategies: Strategy[]) =>
  strategies.filter((s) => s.status === "Research");

/** Lifetime cap per sleeve, keyed by the short label used for attribution. */
export const capsByShort = (strategies: Strategy[]): Record<string, number> =>
  Object.fromEntries(
    liveSleeves(strategies)
      .filter((s) => s.budget !== null)
      .map((s) => [s.short, s.budget as number]),
  );

export const totalCap = (strategies: Strategy[]) =>
  liveSleeves(strategies).reduce((sum, s) => sum + (s.budget ?? 0), 0);

export const money = (n: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);

export const compactMoney = (n: number) =>
  n >= 1000 ? `$${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k` : money(n);
