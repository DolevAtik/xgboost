import { useEffect, useState } from "react";
import { MotionConfig } from "framer-motion";
import { DataProvider } from "./lib/data";
import { Lab } from "./lab/Lab";
import { Story } from "./Story";

/** "#/lab" and "#/lab/<id>" open the artifact lab; every other hash is an in-page anchor. */
function useLabRoute(): { lab: boolean; id?: string } {
  const read = () => {
    const m = window.location.hash.match(/^#\/lab(?:\/([a-z]))?/);
    return m ? { lab: true, id: m[1] } : { lab: false };
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => {
      const next = read();
      setRoute((prev) => {
        if (prev.lab !== next.lab || prev.id !== next.id) window.scrollTo(0, 0);
        return next;
      });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export default function App() {
  const route = useLabRoute();
  return (
    <MotionConfig reducedMotion="user">
      <DataProvider>{route.lab ? <Lab id={route.id} /> : <Story />}</DataProvider>
    </MotionConfig>
  );
}
