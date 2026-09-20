/** Small schematics of how a sleeve decides — never of how it performed.
 *  Every figure here is a fixed illustration, labelled as such, so nothing can
 *  be misread as a result or a backtest. */
import type { MechanismKind } from "@/types/portfolio";

interface SleeveMechanismProps {
  kind: MechanismKind;
}

/** Illustrative jumpiness levels and the share of money each would receive
 *  under inverse-volatility weighting. Fixed values, not measurements. */
const VOL_SPLIT = [
  { label: "calm", share: 38 },
  { label: "steady", share: 28 },
  { label: "choppy", share: 20 },
  { label: "jumpy", share: 14 },
];

const SECTOR_TILT = [
  { label: "Energy", weight: 100, lead: true },
  { label: "Banks", weight: 78, lead: true },
  { label: "Health", weight: 40, lead: false },
  { label: "Utilities", weight: 22, lead: false },
  { label: "Tech", weight: 12, lead: false },
];

export default function SleeveMechanism({ kind }: SleeveMechanismProps) {
  if (kind === "inverse-vol") {
    return (
      <figure className="mechanism">
        <figcaption>
          How the money is split <span className="mechanism-tag">Illustrative</span>
        </figcaption>
        <div className="mech-bars">
          {VOL_SPLIT.map(({ label, share }) => (
            <div key={label} className="mech-bar">
              <span className="mech-track">
                <i style={{ height: `${share * 2.4}%` }} />
                <em>{share}%</em>
              </span>
              <span className="mech-bar-label">{label}</span>
            </div>
          ))}
        </div>
        <p className="mechanism-note">
          The calmer a share has been, the larger its slice — so one wild holding
          cannot swing the sleeve on its own.
        </p>
      </figure>
    );
  }

  if (kind === "sector-tilt") {
    return (
      <figure className="mechanism">
        <figcaption>
          Where a month's ranking tilts <span className="mechanism-tag">Illustrative</span>
        </figcaption>
        <ul className="mech-tilt">
          {SECTOR_TILT.map(({ label, weight, lead }) => (
            <li key={label} className={lead ? "is-lead" : ""}>
              <span>{label}</span>
              <i style={{ width: `${weight}%` }} />
            </li>
          ))}
        </ul>
        <p className="mechanism-note">
          Leaders are held, laggards are skipped, and the whole ranking is redone
          from scratch next month.
        </p>
      </figure>
    );
  }

  return (
    <figure className="mechanism">
      <figcaption>
        When a fall becomes a candidate <span className="mechanism-tag">Illustrative</span>
      </figcaption>
      <div className="mech-gauge">
        <span className="gauge-zone" />
        <span className="gauge-marker" />
        <span className="gauge-end gauge-end--low">year&apos;s low</span>
        <span className="gauge-end gauge-end--high">year&apos;s high</span>
      </div>
      <p className="mechanism-note">
        Only shares deep in the shaded band are considered, and only once the
        fall has stopped and the business still passes its checks.
      </p>
    </figure>
  );
}
