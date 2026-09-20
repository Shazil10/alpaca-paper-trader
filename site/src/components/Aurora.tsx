import { useEffect, useRef } from 'react';
import './Aurora.css';

/** Soft mint aurora wash — React Bits–style background, recolored to brand green. */
export default function Aurora() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      el.style.opacity = '0.35';
      return;
    }

    let frame = 0;
    let raf = 0;
    const tick = () => {
      frame += 0.004;
      const x = 50 + Math.sin(frame) * 12;
      const y = 30 + Math.cos(frame * 0.7) * 10;
      el.style.setProperty('--ax', `${x}%`);
      el.style.setProperty('--ay', `${y}%`);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <div className="aurora" ref={ref} aria-hidden="true" />;
}
