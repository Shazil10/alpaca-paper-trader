/** The one signature interaction: scrolling the spine walks a labelled payload
 *  through the system, so the reader learns the order of operations by moving.
 *  Mobile and reduced-motion get the same content as directly selectable steps. */
import { useCallback, useEffect, useRef, useState } from "react";
import { motion, useMotionValueEvent, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { Boxes, GitBranch, Radar, Workflow } from "lucide-react";
import SectionHeader from "./SectionHeader";
import { liveSleeves } from "@/lib/sleeves";
import type { Strategy } from "@/types/portfolio";

interface Beat {
  badge: string;
  kicker: string;
  headline: string;
  body: string;
  /** What leaves this stage — the packet carries this label as it travels. */
  payload: string;
  chipTitle: string;
  chipSub: string;
  bullets: string[];
  Icon: typeof Boxes;
}

/** Stage two describes whatever sleeves currently exist, so the narrative
 *  cannot fall out of step with the strategy list. */
function buildBeats(strategies: Strategy[]): Beat[] {
  const live = liveSleeves(strategies);

  return [
    {
      badge: "01 / RESEARCH",
      kicker: "Choosing what to look at",
      headline: "An idea has to earn its budget before it gets any money.",
      body: "Every weekday morning, before a single rule runs, the system rebuilds the list of shares it is allowed to trade — large enough to sell easily, and above a minimum price. New ideas are tested against past data first and hold no money while they are being tested.",
      payload: "candidate shares",
      chipTitle: "Morning screen",
      chipSub: "easy to sell · above a price floor",
      bullets: [
        "The tradable list is rebuilt each weekday morning, ahead of any decision",
        "Untested ideas sit in a queue with no money attached to them",
      ],
      Icon: Radar,
    },
    {
      badge: "02 / SLEEVES",
      kicker: "Deciding what to buy",
      headline: "Separate engines, one shared way of asking.",
      body: "Each sleeve reads that same list and answers the same question in its own way: what would you buy, and how much of it. No sleeve can see or change what another one is doing, so a bad idea in one cannot spread.",
      payload: "requested trades",
      chipTitle: "Same question, different answers",
      chipSub: live.map((s) => s.short).join(" · "),
      bullets: live.map((s) => `${s.short} — ${s.plain}`),
      Icon: Boxes,
    },
    {
      badge: "03 / CAPITAL CONTROL",
      kicker: "Deciding what gets funded",
      headline: "Requests are checked against real limits before anything is bought.",
      body: "This is the part that says no. Every sleeve has a total it may never spend past, and selling frees that room up again. Sells are settled before buys so the cash is actually there, and a slice of the account is deliberately never spent.",
      payload: "approved orders",
      chipTitle: "Spending limits",
      chipSub: "limits · records · cash held back",
      bullets: [
        "Each sleeve has a lifetime limit; selling returns room to it",
        "Every purchase is recorded against the sleeve that asked for it",
        "Sells go first so buys are funded, and a cash reserve is left alone",
      ],
      Icon: GitBranch,
    },
    {
      badge: "04 / EXECUTION",
      kicker: "Placing and recording orders",
      headline: "Orders go out on a schedule, and everything is written down.",
      body: "A scheduled job sends the approved orders to the broker each weekday, after checking the market is genuinely open. What happened is logged at the end of the day, and there is a manual path to force a position closed if one ever gets stuck.",
      payload: "filled orders",
      chipTitle: "Runs on a timer",
      chipSub: "checked · logged · reversible",
      bullets: [
        "A weekday schedule with time-zone checks, and no button to press",
        "An end-of-day record of every order is kept for review",
        "A manual override exists for when a position has to go",
      ],
      Icon: Workflow,
    },
  ];
}

interface ArchitectureScrollProps {
  strategies: Strategy[];
}

export default function ArchitectureScroll({ strategies }: ArchitectureScrollProps) {
  const [active, setActive] = useState(0);
  const [isDesktop, setIsDesktop] = useState(false);
  const reduceMotion = useReducedMotion();
  const timelineRef = useRef<HTMLDivElement>(null);
  const stepRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const { scrollYProgress } = useScroll({
    target: timelineRef,
    offset: ["start center", "end center"],
  });
  const packetTop = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  const beats = buildBeats(strategies);
  const stageCount = beats.length;
  const beat = beats[active] ?? beats[0];
  const Icon = beat.Icon;
  const animate = isDesktop && !reduceMotion;
  // The card changes sides as the payload advances, so the section has a
  // left-right rhythm instead of one fixed column of text.
  const flipped = active % 2 === 1;

  useEffect(() => {
    const media = window.matchMedia("(min-width: 901px)");
    const update = () => setIsDesktop(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  // The active stage is derived from the same progress value that moves the
  // packet, so the card, the spine fill, and the marker can never disagree.
  const syncStage = useCallback(
    (progress: number, force = false) => {
      const raw = progress * stageCount;
      const next = Math.min(stageCount - 1, Math.max(0, Math.floor(raw)));
      setActive((prev) => {
        if (next === prev) return prev;
        // The deadband stops the stage flickering on a boundary, but it only
        // guards adjacent steps: a bigger jump means the reader arrived
        // mid-section rather than scrolling through it.
        if (force || Math.abs(next - prev) > 1) return next;
        const distanceIntoStage = raw - next;
        return (next > prev ? distanceIntoStage > 0.12 : distanceIntoStage < 0.88)
          ? next
          : prev;
      });
    },
    [stageCount],
  );

  useMotionValueEvent(scrollYProgress, "change", (progress) => {
    if (animate) syncStage(progress);
  });

  // Landing part-way down the section produces no scroll event, so read the
  // measured progress once the layout has settled.
  useEffect(() => {
    if (!animate) return;
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => syncStage(scrollYProgress.get(), true));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [animate, syncStage, scrollYProgress]);

  const selectStep = (index: number) => {
    setActive(index);
    if (animate) {
      stepRefs.current[index]?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <section id="architecture" className="section architecture-section">
      <div className="architecture-orbit" aria-hidden="true" />
      <div className="section-inner">
        <SectionHeader
          label="01. Architecture"
          title="How one order actually happens"
          lede={
            animate
              ? "Scroll this section to follow a single trade from idea to filled order. The marker on the line carries whatever is being handed to the next stage."
              : "Every trade passes through the same stages in the same order. Pick any one to see what it hands to the next."
          }
        />

        <div className={`architecture-grid ${flipped ? "is-flipped" : ""}`}>
          <div className="architecture-sticky">
            <div className="architecture-card">
              <div className="card-grid-texture" aria-hidden="true" />
              <div className="architecture-card-topline">
                <span>{beat.badge}</span>
                <span className="live-state"><i /> CARRYING: {beat.payload.toUpperCase()}</span>
              </div>

              {/* CSS keyed animation rather than a JS entrance: the resting state
                  is visible even if the animation never runs. */}
              <div className="architecture-card-content" key={beat.badge}>
                <div className="architecture-icon"><Icon size={22} strokeWidth={1.6} /></div>
                <p className="micro-label">{beat.kicker}</p>
                <h3>{beat.headline}</h3>
                <p className="architecture-body">{beat.body}</p>

                <div className="architecture-chip">
                  <span>{beat.chipTitle}</span>
                  <code>{beat.chipSub}</code>
                </div>

                <ul className="architecture-bullets">
                  {beat.bullets.map((bullet) => <li key={bullet}>{bullet}</li>)}
                </ul>
              </div>

              <div className="architecture-card-footer">
                <span>PAPER EXECUTION</span>
                <span>{String(active + 1).padStart(2, "0")} — {String(beats.length).padStart(2, "0")}</span>
              </div>
            </div>
          </div>

          <div ref={timelineRef} className="architecture-timeline">
            <div className="timeline-track" aria-hidden="true">
              <motion.span
                className="timeline-fill"
                style={animate ? { scaleY: scrollYProgress } : { scaleY: active / (beats.length - 1) }}
              />
              {animate && (
                <motion.span className="timeline-packet" style={{ top: packetTop }}>
                  <i />
                  <span className="packet-label">{beat.payload}</span>
                </motion.span>
              )}
            </div>
            {beats.map((item, index) => {
              const TimelineIcon = item.Icon;
              return (
                <button
                  ref={(node) => { stepRefs.current[index] = node; }}
                  className={`timeline-step ${index === active ? "is-active" : ""} ${index < active ? "is-complete" : ""}`}
                  key={item.badge}
                  type="button"
                  onClick={() => selectStep(index)}
                  aria-current={index === active ? "step" : undefined}
                >
                  <span className="timeline-node"><TimelineIcon size={17} strokeWidth={1.7} /></span>
                  <span className="timeline-copy">
                    <small>{item.badge}</small>
                    <strong>{item.headline}</strong>
                    <em>hands over: {item.payload}</em>
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="system-truth" aria-label="System summary">
          <span>UNIVERSE<small>Rebuilt every weekday</small></span>
          <i aria-hidden="true" />
          <span>SLEEVES<small>{liveSleeves(strategies).map((s) => s.short).join(" / ")}</small></span>
          <i aria-hidden="true" />
          <span>CAPITAL CONTROL<small>Limits, records, reserve</small></span>
          <i aria-hidden="true" />
          <span>EXECUTION<small>Scheduled, logged, reversible</small></span>
        </div>
      </div>
    </section>
  );
}
