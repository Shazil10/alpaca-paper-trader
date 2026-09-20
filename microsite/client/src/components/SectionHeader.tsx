interface SectionHeaderProps {
  label: string;
  title: string;
  lede: string;
  className?: string;
}

export default function SectionHeader({ label, title, lede, className = "" }: SectionHeaderProps) {
  const sectionIndex = label.slice(0, 2);

  return (
    <header className={`section-header ${className}`}>
      <span className="section-index" aria-hidden="true">{sectionIndex}</span>
      <p className="section-label">{label}</p>
      <h2>{title}</h2>
      <p className="section-lede">{lede}</p>
    </header>
  );
}
