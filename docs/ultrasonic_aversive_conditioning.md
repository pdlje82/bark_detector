# Ultrasonic Aversive Conditioning — Research Summary & Design Implications

This document summarises what peer-reviewed science says about using ultrasonic
stimuli to reduce unwanted barking, and translates each finding into a concrete
design decision for the bark detector project.

> **Context:** The bark detector's Phase 1 goal is logging.  Any aversive
> response capability would be a Phase 2 addition.  Read the ethical and legal
> section at the bottom before implementing it.

---

## 1. Do Ultrasonic Aversive Stimuli Actually Work?

**Short answer: yes initially, but habituation is the critical failure mode.**

One controlled app-based trial with a reactive German Shepherd mix achieved
roughly **60 % bark reduction** in initial sessions.  However, after two weeks
of regular use at a fixed frequency, the response rate dropped to only **25 %**
— a clear sign the dog had habituated to the stimulus and learned to ignore it.

Key risks identified in the literature:

- **Habituation** — dogs stop responding to a predictable, repetitive stimulus.
  A fixed-frequency device will likely become ineffective within weeks.
- **Anxiety escalation** — the aversive stimulus can increase generalised fear
  in some individuals, leading to secondary behavioural problems unrelated to
  barking (e.g. destructive behaviour, aggression, separation anxiety).

**Design implication:** A single fixed-frequency emitter is almost guaranteed
to stop working.  Frequency randomisation across the 22–28 kHz range is
essential from day one.

---

## 2. Optimal Frequency Range

Research identifies **~25 kHz** as particularly effective for dogs — it sits
above human hearing while still being salient enough to interrupt behaviour
without requiring extreme sound pressure levels.

Practitioner recommendations to counter habituation:

| Technique | Rationale |
|---|---|
| Dual or variable frequency (22–28 kHz) | Prevents the dog from tuning out a single predictable tone |
| Randomise frequency per bark event | No two events sound identical |
| Auto-off after 8–10 seconds | Limits total exposure per event; protects hearing |
| Escalating intensity | Start at minimum effective level; increase only if no response |

**Design implication:** The bark detector should select a random frequency
within 22–28 kHz on each trigger, with a hard 8–10 second burst cap enforced
in code regardless of how long the barking continues.

---

## 3. Physiological Stress Responses in Dogs

A 2018 paper in *PLOS ONE* (Franzini de Souza et al.) measured physiological
responses to sound stimuli in sound-sensitive dogs:

- Marked **autonomic imbalance toward sympathetic predominance** (fight-or-flight
  activation) during and after the stimulus
- Elevated **cortisol release** — a reliable biomarker for psychological stress
- Behavioural indicators persisted into the **recovery period** after the sound
  stopped, not just during exposure

A related study on kennel noise found that dogs exposed to sounds up to 95 dB
displayed:

- Paw lifting and lowered body postures
- Body shaking and snout licks (displacement behaviours indicating negative
  emotional state)
- Heart rate increases of **up to 54 % from baseline** in some individuals
- Dose-response relationship: more pronounced reactions at higher decibel levels

**Design implication:** Start at the **lowest effective sound pressure level**
and escalate only if the dog shows no response.  Log the stimulus level
alongside bark events in InfluxDB so you can audit the dose over time.

---

## 4. Classical Counterconditioning vs. Aversive Stimuli

A 2022 pilot study (PMC) on kenneled dogs found that **positive
counterconditioning outperformed purely aversive approaches** for long-term
bark reduction.

Key mechanism identified: in some dogs, *barking itself acts as a stimulus for
further barking* — a positive feedback loop driven by arousal.  Classical
counterconditioning (pairing the bark trigger with something pleasant)
specifically targets this loop.  Pure aversive stimuli, by contrast, risk:

- **Learned helplessness** — the dog stops offering any behaviour, including
  calm ones, because all responses have been punished
- **Aggression without warning** — suppression of the warning bark (growl →
  bark → bite sequence) without resolving the underlying emotional state

**Design implication:** Aversive conditioning is most defensible as a *last
resort* after logging confirms a genuine problem, not as a first response.
The data collected in Phase 1 may itself be sufficient — if the logs show
barking is within tolerable limits, no aversive response is needed at all.

---

## 5. The Broader Aversive Training Literature

Ziv (2017), *Journal of Veterinary Behavior* — a review of 17 studies on
aversive training methods:

> "Using aversive methods can jeopardize both the physical and mental health of
> dogs.  While positive punishment can be effective, there is no evidence that
> it is more effective than positive reinforcement-based training."

A separate multivariate analysis found modest but statistically significant
**positive associations** between owner use of aversive methods and:

- Persistent barking (the behaviour being targeted — aversive methods can make
  it worse, not better)
- Stranger-directed aggression
- Separation-related problems

**Design implication:** The literature does not support aversive conditioning
as a reliable or safe first-line intervention.  It should be treated as a
tool of last resort, used at minimum effective dose, and monitored closely
for secondary behavioural changes.

---

## Summary Table — Findings vs. Design Decisions

| Research Finding | Implication for Implementation |
|---|---|
| Habituation within ~2 weeks at fixed frequency | Randomise frequency 22–28 kHz on every bark event |
| ~25 kHz most effective range | Use as centre frequency for randomisation |
| 8–10 s max exposure recommended | Hard-cap burst duration in code; never extend regardless of continued barking |
| Stress response scales with dB level | Start at minimum level; escalate only with no response |
| Positive counterconditioning outperforms aversives long-term | Treat aversive output as last resort; explore owner notification first |
| Aversive methods associated with increased barking in some cases | Monitor bark frequency in InfluxDB after activation — if it rises, stop |
| Neighbor's dog, no owner consent | Ethical and legal grey area — see section below |

---

## Ethical & Legal Considerations

Using an aversive device on a **neighbour's dog without the owner's knowledge**
is legally ambiguous across most EU jurisdictions.  Relevant considerations for
Spain:

- **Animal welfare law (Ley 7/2023)** — Spain's 2023 animal welfare act
  prohibits actions that cause unnecessary suffering to animals.  Whether a
  calibrated ultrasonic stimulus constitutes "unnecessary suffering" depends on
  intensity and duration, but the law creates legal exposure.
- **Neighbour law** — the affected party is the dog's owner; they have not
  consented to an intervention on their animal.
- **Evidence value** — your Phase 1 InfluxDB logs *are* evidence of a
  noise nuisance problem.  A formal complaint to the building community or
  local authority is legally cleaner than unilateral aversive conditioning.

**Recommended sequence:**

1. **Log (Phase 1, current)** — establish a baseline with timestamps,
   frequency, and duration.  This data is already being collected.
2. **Analyse** — if the data shows barking exceeds tolerable limits (e.g.
   multiple episodes per day, late night events), you have objective evidence.
3. **Talk to the neighbour** — present the data.  Most people respond to
   objective evidence better than complaints.
4. **Formal complaint** — if conversation fails, the logs are documentation
   for a community or municipal complaint.
5. **Aversive conditioning (Phase 2, optional)** — only if all other avenues
   are exhausted, at minimum effective dose, with continued monitoring.

The logging phase is not just a technical precursor — it is likely the most
effective intervention available.
