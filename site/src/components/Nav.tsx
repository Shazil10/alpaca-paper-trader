import { useEffect, useState } from 'react';
import './Nav.css';

const LINKS = [
  { href: '#architecture', label: 'Architecture', num: '01' },
  { href: '#strategies', label: 'Strategies', num: '02' },
  { href: '#book', label: 'Book', num: '03' },
  { href: '#contribution', label: 'Contribution', num: '04' },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header className={`nav ${scrolled ? 'scrolled' : ''}`}>
      <a className="nav-logo" href="https://shazilfarukh.com">
        SF
      </a>
      <ul className="nav-links">
        {LINKS.map((l) => (
          <li key={l.href}>
            <a href={l.href}>
              <span className="num">{l.num}.</span>
              <span className="hide-sm">{l.label}</span>
            </a>
          </li>
        ))}
        <li>
          <a
            className="btn nav-cta"
            href="https://github.com/Shazil10/alpaca-paper-trader"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </li>
      </ul>
    </header>
  );
}
