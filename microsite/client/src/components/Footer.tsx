import { ExternalLink } from "lucide-react";

interface FooterProps {
  repo: string;
  portfolioHome: string;
}

export default function Footer({ repo, portfolioHome }: FooterProps) {
  return (
    <footer className="site-footer">
      <p className="footer-route">
        UNIVERSE <i /> SLEEVES <i /> CAPITAL CONTROL <i /> BROKER
      </p>
      <div className="footer-inner">
        <div className="footer-signature">
          <h2>
            System architecture
            <span>&amp; engineering by Shazil Farukh</span>
          </h2>

          {/* PERSONAL NOTE: replace with something specific — what you would change
              next, what surprised you, or what you learned running it this long. */}
          <p className="personal-note personal-note--footer">
            <span className="note-tag">Your words</span>
            [One specific closing line: what surprised you, or what you would
            build differently next time?]
          </p>
        </div>

        <div className="footer-actions">
          <a href={portfolioHome}>shazilfarukh.com</a>
          <a className="github-cta" href={repo} target="_blank" rel="noreferrer">
            View Source Code on GitHub
            <ExternalLink size={14} aria-hidden="true" />
          </a>
        </div>
      </div>
      <div className="footer-bottom">
        <span>ALPACA PAPER · NOT INVESTMENT ADVICE</span>
        <a href="#top">BACK TO TOP</a>
      </div>
    </footer>
  );
}
