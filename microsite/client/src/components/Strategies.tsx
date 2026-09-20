import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import SectionHeader from "./SectionHeader";
import SleeveMechanism from "./SleeveMechanism";
import { liveSleeves, money, researchSleeves, totalCap } from "@/lib/sleeves";
import type { Strategy } from "@/types/portfolio";

interface StrategiesProps {
  strategies: Strategy[];
}

/** A citation, not a code link: it names the source of the idea. */
function Evidence({ sleeve }: { sleeve: Strategy }) {
  if (!sleeve.evidence) return null;
  return (
    <a className="evidence-link" href={sleeve.evidence.url} target="_blank" rel="noreferrer">
      {sleeve.evidence.label}
      <ExternalLink size={11} aria-hidden="true" />
    </a>
  );
}

/* PERSONAL NOTE: replace the placeholder wording with the builder's own reason
   for choosing this approach — what convinced them, or what they rejected. */
function VoiceNote() {
  return (
    <p className="personal-note personal-note--card">
      <span className="note-tag">Your words</span>
      [Why this one? What convinced you it was worth real capital?]
    </p>
  );
}

function SleeveHead({ sleeve }: { sleeve: Strategy }) {
  return (
    <div className="sleeve-top">
      <span className="status-tag"><i />Active</span>
      <span className="sleeve-allocation">
        {sleeve.budget !== null ? money(sleeve.budget) : "—"}
        <small>Most it may ever use</small>
      </span>
    </div>
  );
}

function SleeveFacts({ sleeve }: { sleeve: Strategy }) {
  return (
    <div className="sleeve-facts">
      {sleeve.sizing && (
        <div>
          <span>How it sizes</span>
          <strong>{sleeve.sizing}</strong>
        </div>
      )}
      {sleeve.holdings && sleeve.holdings.length > 0 && (
        <div>
          <span>Holding now</span>
          <strong>{sleeve.holdings.join(" · ")}</strong>
        </div>
      )}
    </div>
  );
}

function TermChips({ sleeve }: { sleeve: Strategy }) {
  if (!sleeve.chips || sleeve.chips.length === 0) return null;
  return (
    <ul className="term-chips">
      {sleeve.chips.map((chip) => (
        <li key={chip}>{chip}</li>
      ))}
    </ul>
  );
}

/** The featured sleeve runs the full width and splits internally, so the board
 *  is a wide banner over compact cards rather than an even grid. */
function FeaturedSleeve({ sleeve }: { sleeve: Strategy }) {
  return (
    <article className="sleeve-card sleeve-card--featured">
      <SleeveHead sleeve={sleeve} />
      <h3>{sleeve.name}</h3>

      <div className="featured-body">
        <div className="featured-main">
          <p className="sleeve-thesis">{sleeve.thesis}</p>
          {sleeve.rules && sleeve.rules.length > 0 && (
            <ul className="architecture-bullets">
              {sleeve.rules.map((rule) => (
                <li key={rule}>{rule}</li>
              ))}
            </ul>
          )}
          <TermChips sleeve={sleeve} />
        </div>
        <div className="featured-side">
          {sleeve.mechanism && <SleeveMechanism kind={sleeve.mechanism} />}
          <VoiceNote />
        </div>
      </div>

      <SleeveFacts sleeve={sleeve} />
      <Evidence sleeve={sleeve} />
    </article>
  );
}

function SleeveCard({ sleeve }: { sleeve: Strategy }) {
  return (
    <article className="sleeve-card">
      <SleeveHead sleeve={sleeve} />
      <h3>{sleeve.name}</h3>
      <p className="sleeve-thesis">{sleeve.thesis}</p>
      {sleeve.mechanism && <SleeveMechanism kind={sleeve.mechanism} />}
      <VoiceNote />
      <TermChips sleeve={sleeve} />
      <SleeveFacts sleeve={sleeve} />
      <Evidence sleeve={sleeve} />
    </article>
  );
}

export default function Strategies({ strategies }: StrategiesProps) {
  const live = liveSleeves(strategies);
  const research = researchSleeves(strategies);
  // Open by default: the queue is the evidence that nothing ships unreviewed.
  const [researchOpen, setResearchOpen] = useState(true);
  const featured = live.find((s) => s.featured) ?? live[0];
  const rest = live.filter((s) => s.id !== featured?.id);

  return (
    <section id="strategies" className="section strategies-section">
      <div className="section-inner">
        <SectionHeader
          label="02. Sleeves"
          title="One capital contract, several live engines"
          lede={`Each sleeve is a separate set of rules with its own spending limit — ${money(totalCap(strategies))} across all of them — and none of them can touch money the others have committed.`}
        />

        <div className="sleeve-board">
          {featured && <FeaturedSleeve sleeve={featured} />}
          {rest.length > 0 && (
            <div className="sleeve-rest">
              {rest.map((sleeve) => (
                <SleeveCard key={sleeve.id} sleeve={sleeve} />
              ))}
            </div>
          )}
        </div>

        {research.length > 0 && (
          <div className={`research-block ${researchOpen ? "is-open" : ""}`}>
            <button
              type="button"
              className="research-toggle"
              onClick={() => setResearchOpen((open) => !open)}
              aria-expanded={researchOpen}
              aria-controls="research-queue"
            >
              Still being tested — no money attached
              <span className="research-count metric">{String(research.length).padStart(2, "0")}</span>
              <ChevronDown size={15} className={researchOpen ? "is-open" : ""} aria-hidden="true" />
            </button>
            {researchOpen && (
              <ul className="research-list" id="research-queue">
                {research.map((item) => (
                  <li key={item.id}>
                    <strong>{item.name}</strong>
                    <p>{item.thesis}</p>
                    <Evidence sleeve={item} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
