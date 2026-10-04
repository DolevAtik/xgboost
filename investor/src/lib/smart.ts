/** Plain-language names for the fifteen SMART attributes the model reads. */
export const SMART: Record<string, { name: string; plain: string; smartctl: string }> = {
  smart_1_raw: { name: "Read errors", plain: "how often reading data needed a retry", smartctl: "Raw_Read_Error_Rate" },
  smart_3_raw: { name: "Spin-up time", plain: "how long the platters take to reach speed", smartctl: "Spin_Up_Time" },
  smart_4_raw: { name: "Start/stop count", plain: "how many times the motor has started", smartctl: "Start_Stop_Count" },
  smart_5_raw: { name: "Reallocated sectors", plain: "damaged spots the drive has already retired", smartctl: "Reallocated_Sector_Ct" },
  smart_7_raw: { name: "Seek errors", plain: "misses while moving the read head", smartctl: "Seek_Error_Rate" },
  smart_9_raw: { name: "Power-on hours", plain: "the drive's age in running hours", smartctl: "Power_On_Hours" },
  smart_10_raw: { name: "Spin retries", plain: "failed attempts to spin up", smartctl: "Spin_Retry_Count" },
  smart_12_raw: { name: "Power cycles", plain: "how many times it was switched on", smartctl: "Power_Cycle_Count" },
  smart_192_raw: { name: "Emergency retracts", plain: "head parked because power was cut", smartctl: "Power-Off_Retract_Count" },
  smart_193_raw: { name: "Load cycles", plain: "times the head was parked and unparked", smartctl: "Load_Cycle_Count" },
  smart_194_raw: { name: "Temperature", plain: "degrees Celsius", smartctl: "Temperature_Celsius" },
  smart_197_raw: { name: "Pending sectors", plain: "unstable spots waiting to be retired", smartctl: "Current_Pending_Sector" },
  smart_198_raw: { name: "Uncorrectable sectors", plain: "spots that could not be read at all", smartctl: "Offline_Uncorrectable" },
  smart_199_raw: { name: "Cable/interface errors", plain: "data corrupted on the way to the host", smartctl: "UDMA_CRC_Error_Count" },
  smart_240_raw: { name: "Head flying hours", plain: "time the read head spent in motion", smartctl: "Head_Flying_Hours" },
};

export const smartName = (attr: string) => SMART[attr]?.name ?? attr;

const AGG: Record<string, string> = { last: "latest", mean: "30-day average", delta: "change over the window" };

/**
 * "smart_197_raw_d7__last" -> "Pending sectors — 7-day growth, latest".
 * Channels are the raw attribute (log-scaled), its 1-day difference (_d1), its 7-day
 * difference (_d7), and `observed`, the share of days the drive actually reported.
 */
export function featureLabel(feature: string): { title: string; detail: string } {
  const [channel, agg = ""] = feature.split("__");
  const aggText = AGG[agg] ?? agg;
  if (channel === "observed") return { title: "Reporting completeness", detail: `share of days reported, ${aggText}` };
  const m = channel.match(/^(smart_\d+_raw)(?:_d(\d+))?$/);
  if (!m) return { title: channel, detail: aggText };
  const base = smartName(m[1]);
  const diff = m[2] ? `${m[2]}-day growth, ` : "";
  return { title: base, detail: `${diff}${aggText}` };
}

/** The attributes the Signal chapter leads with, in story order. */
export const SIGNAL_ATTRS = ["smart_197_raw", "smart_198_raw", "smart_5_raw", "smart_194_raw"] as const;
