import { motion, useReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';

export default function ScrollReveal({
  children,
  className = '',
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const reduce = useReducedMotion();

  if (reduce) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.55, ease: [0.645, 0.045, 0.355, 1], delay }}
    >
      {children}
    </motion.div>
  );
}
