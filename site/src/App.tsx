import Aurora from './components/Aurora';
import Nav from './components/Nav';
import Hero from './components/Hero';
import ArchitectureScroll from './components/ArchitectureScroll';
import Strategies from './components/Strategies';
import BookSnapshot from './components/BookSnapshot';
import SleeveContribution from './components/SleeveContribution';
import Footer from './components/Footer';
import { usePortfolio } from './hooks/usePortfolio';

export default function App() {
  const { data, error, loading } = usePortfolio();

  return (
    <>
      <Aurora />
      <Nav />
      <div className="page">
        <Hero />
        <ArchitectureScroll />
        {loading && (
          <p className="mono" style={{ color: 'var(--slate)', padding: '40px 0' }}>
            Loading portfolio…
          </p>
        )}
        {error && (
          <p className="mono" style={{ color: 'var(--loss)', padding: '40px 0' }}>
            Could not load portfolio data.
          </p>
        )}
        {data && (
          <>
            <Strategies strategies={data.strategies} />
            <BookSnapshot data={data} />
            <SleeveContribution rows={data.sleeve_realized_pnl} />
          </>
        )}
        <Footer />
      </div>
    </>
  );
}
