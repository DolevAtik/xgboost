import { useState } from "react";
import { motion } from "framer-motion";
import { StorySection } from "../components/StorySection";
import { useRise } from "../lib/motion";
import { TEAM, type Member } from "../lib/team";
import { useLang } from "../lib/i18n";

const initials = (name: string) =>
  name
    .split(/\s+/)
    .map((p) => p[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

/** The member's photo from public/team/, or their initials until it is added. */
function Portrait({ m }: { m: Member }) {
  const [failed, setFailed] = useState(false);
  const { t } = useLang();
  return (
    <div className="relative aspect-square w-full overflow-hidden rounded-xl border hairline bg-gradient-to-br from-deep to-night">
      {!failed ? (
        <img
          src={`/team/${m.photo}`}
          alt={t(`Portrait of ${m.name}`, `תמונה של ${m.name}`)}
          loading="lazy"
          onError={() => setFailed(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center" role="img" aria-label={m.name}>
          <span className="text-5xl font-semibold tracking-tight text-signal/70">{initials(m.name)}</span>
        </div>
      )}
      <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-night/70 to-transparent" />
    </div>
  );
}

/** Meet the team. */
export default function Team() {
  const rise = useRise(0, 16);
  const { t, he, lang } = useLang();
  return (
    <StorySection
      id="team"
      kicker={t("Meet the team", "הכירו את הצוות")}
      title={
        he ? (
          <>
            האנשים שמאחורי <bdi className="text-signal">Drive Foresight</bdi>.
          </>
        ) : (
          <>
            The people behind <span className="text-signal">Drive Foresight</span>.
          </>
        )
      }
      lede={t(
        "Three engineers covering the full path, from raw fleet telemetry, through the models, to the product you can try below.",
        "שלושה מהנדסים שמכסים את כל הדרך: מהנתונים הגולמיים של הצי, דרך המודלים, ועד המוצר שאתם עומדים לנסות.",
      )}
    >
      <ul className="grid gap-4 md:grid-cols-3">
        {TEAM.map((m) => (
          <motion.li key={m.name} {...rise} className="panel flex flex-col">
            <Portrait m={m} />
            <div className="mt-5">
              <h3 className="text-lg font-semibold">
                <bdi>{m.name}</bdi>
              </h3>
              <p className="mt-0.5 text-sm text-signal">{m.role[lang]}</p>
              <p className="mt-3 text-sm leading-relaxed text-soft">{m.summary[lang]}</p>
            </div>
            <ul className="mt-5 flex flex-wrap gap-2 border-t hairline pt-4" aria-label={t(`${m.name}'s focus areas`, `תחומי האחריות של ${m.name}`)}>
              {m.focus.map((f) => (
                <li key={f.en} className="chip">
                  {f[lang]}
                </li>
              ))}
            </ul>
          </motion.li>
        ))}
      </ul>
    </StorySection>
  );
}
