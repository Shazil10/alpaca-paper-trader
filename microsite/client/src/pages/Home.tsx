import { useEffect, useState } from "react";
import Attribution from "@/components/Attribution";
import Footer from "@/components/Footer";
import Hero from "@/components/Hero";
import ArchitectureScroll from "@/components/ArchitectureScroll";
import Nav from "@/components/Nav";
import Strategies from "@/components/Strategies";
import { capsByShort } from "@/lib/sleeves";
import type { PortfolioData } from "@/types/portfolio";

export default function Home() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/data/portfolio.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Portfolio data failed to load");
        return response.json() as Promise<PortfolioData>;
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(true);
      });
    return () => controller.abort();
  }, []);

  if (error) {
    return (
      <main className="data-state">
        <h1>Portfolio snapshot could not be loaded.</h1>
        <p>Refresh, or confirm the data feed is available.</p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="data-state" aria-live="polite">
        <p>Loading paper book…</p>
      </main>
    );
  }

  const budgets = capsByShort(data.strategies);

  return (
    <div id="top" className="site-shell">
      <Nav repo={data.repo} portfolioHome={data.portfolio_home} />
      <main>
        <Hero strategies={data.strategies} />
        <ArchitectureScroll strategies={data.strategies} />
        <Strategies strategies={data.strategies} />
        <Attribution
          positions={data.positions}
          budgets={budgets}
          pnl={data.sleeve_realized_pnl}
          asOf={data.as_of}
          disclaimer={data.disclaimer}
        />
      </main>
      <Footer repo={data.repo} portfolioHome={data.portfolio_home} />
    </div>
  );
}
