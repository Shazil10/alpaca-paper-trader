import { useEffect, useState } from "react";
import { ExternalLink, Menu, X } from "lucide-react";

const links = [
  { label: "Architecture", href: "#architecture" },
  { label: "Sleeves", href: "#strategies" },
  { label: "Book", href: "#portfolio" },
];

interface NavProps {
  repo: string;
  portfolioHome: string;
}

export default function Nav({ repo, portfolioHome }: NavProps) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <header className={`site-nav ${scrolled ? "is-scrolled" : ""}`}>
      <a className="brand-lockup" href={portfolioHome}>
        <span className="live-pill">
          <i /> Live Paper
        </span>
        <span className="brand-name">Alpaca Trading Infrastructure</span>
      </a>

      <nav className="desktop-nav" aria-label="Primary">
        {links.map((link) => (
          <a key={link.href} href={link.href}>
            {link.label}
          </a>
        ))}
        <a className="github-cta" href={repo} target="_blank" rel="noreferrer">
          View Source on GitHub
          <ExternalLink size={14} aria-hidden="true" />
        </a>
      </nav>

      <button
        className="menu-trigger"
        type="button"
        aria-label={open ? "Close navigation" : "Open navigation"}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>

      <div className={`mobile-nav ${open ? "is-open" : ""}`}>
        {links.map((link) => (
          <a key={link.href} href={link.href} onClick={() => setOpen(false)}>
            {link.label}
          </a>
        ))}
        <a className="github-cta" href={repo} target="_blank" rel="noreferrer">
          View Source on GitHub
          <ExternalLink size={14} />
        </a>
      </div>
    </header>
  );
}
