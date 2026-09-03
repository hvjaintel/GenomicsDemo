# Presenter guide — for the non-genomics presenter

**Audience for this guide:** an ODM/OEM sales or solutions presenter who will run this
booth demo and field questions, and who is *not* a genomics specialist.

You do not need to understand genomics to present this well. You need to understand
**what the machine is doing, why it matters commercially, and where your knowledge
stops.** This guide covers all three, in that order.

---

## The one rule

**Never invent an answer.** The audience at a genomics booth will include people who do
this for a living. A confident wrong answer is worse than no answer, and it is the fastest
way to lose a serious prospect — they will assume everything else you said was also
approximate.

The demo itself was built on this principle: it refuses to display a speedup it cannot
stand behind. You should hold the same line verbally. Section 9 gives you the exact words
for handing a question off, and they are words that *build* credibility rather than
costing it.

> "That's past where I can give you a precise answer, and I'd rather not guess on
> something you'd be making a decision on. Let me take your card and get you the
> engineer who ran these numbers."

Say this early and often. It works. It is also the honest position: you *are* the
hardware person, and nobody expects the hardware person to be a bioinformatician.

---

## 1. What this demo actually does, in plain English

### The 30-second version

> This server is reading a real human genome and finding the places where that person's
> DNA differs from the reference human genome. That's the first computational step in
> nearly all clinical and research genomics. It's normally a heavy compute job. We're
> doing it here, on a bench server, on standard Xeon CPUs — no GPU, no accelerator card.

### The background you need (and no more)

**A genome is 3.1 billion letters long.** A DNA sequencing machine cannot read it end to
end. It reads it in tens of millions of short, overlapping fragments, and some of those
reads contain errors.

**"Variant calling" is the job of working out the truth from that pile of noisy
fragments.** For every position in the genome, you ask: does this person genuinely differ
from the reference here, or is this a sequencing error? A typical person has roughly 4–5
million real differences. Getting this wrong in either direction matters — a missed
variant can be a missed diagnosis, and a false one can send a clinician down the wrong
path.

**DeepVariant is Google's tool for doing that with deep learning**, and the way it works
is the single best hook you have as a non-specialist:

> DeepVariant turns the problem into **image recognition**. It literally renders the
> stacked-up DNA reads at each position as an image, then runs a convolutional neural
> network over those images to classify what it's looking at. It's the same class of
> maths as image classification — which is exactly why it's compute-hungry, and exactly
> why the vector units in these CPUs matter so much.

That is a genuinely accurate description, it is memorable, and it hands you the bridge
from biology to silicon in one sentence. Use it.

**What the demo proves, then, is three things:**

1. This deep-learning workload runs **fast on plain Xeon CPUs**, using AVX-512 — the wide
   vector units built into every core. No add-in card.
2. It **scales across cores**: more cores, proportionally less waiting, *identical
   results*.
3. It runs on a **quiet, air-cooled, bench-deployable box** — so it can live in the lab
   where the science happens, not in a data hall.

---

## 2. Why a customer cares (the commercial translation)

You are not selling variant calling. You are selling **turnaround time, control, and
predictable cost**. Translate accordingly:

| What's on screen | What it means to the buyer |
|---|---|
| Whole genome in **~26 minutes** | A sample sequenced today is analysed today, not queued overnight. In a clinical setting, turnaround time is the product. |
| Runs on **CPU only** | No GPU supply constraints, no CUDA-specific stack, no separate accelerator budget line. It runs on the servers they already know how to buy, deploy and support. |
| **2.66× from 6× the cores**, same answers | Capacity is a purchasing decision, not a re-engineering project. Buy more cores, get more throughput, change no code. |
| **Air-cooled, bench-deployable** | Deploy next to the sequencer. No data-hall dependency, no facilities project, no liquid cooling. |
| **Runs fully offline** | The genome never leaves the box. For patient data this is often a hard regulatory requirement, not a preference. |
| **Efficiency tab** | Lets them sanity-check power and cost against their own sample volume. |

**The line that lands with buyers:**

> The throughput is already in the box. You're not buying an accelerator — you're using
> the silicon you already paid for.

---

## 3. Minimum viable glossary

You only need these. If a term comes up that isn't here, that's an escalation (§9).

| Term | What to say if asked |
|---|---|
| **Genome** | The full set of a person's DNA — about 3.1 billion letters. |
| **Reference genome (GRCh38)** | The standard "map" of a human genome that everyone compares against. GRCh38 is the current mainstream version. |
| **Variant** | A place where this person's DNA differs from that reference. |
| **Variant calling** | Deciding which apparent differences are real versus sequencing noise. What this demo runs. |
| **SNP / indel** | The two common variant types — a single changed letter, or a short insertion/deletion. You'll see both counted on the Results tab. |
| **DeepVariant** | Google's deep-learning variant caller. Open source. Industry standard. Version 1.10 here. |
| **BAM file** | The file of sequencing reads already lined up against the reference. The demo's input. The whole-genome one here is 46 GB. |
| **VCF file** | The output — the list of variants found. |
| **HG002** | A public, extensively characterised reference human sample used industry-wide for benchmarking. Not a patient. Say this plainly if anyone asks about consent or privacy. |
| **35x depth / coverage** | Each position was read about 35 times over. Standard for a clinical-grade whole genome. |
| **chr20** | Chromosome 20 — one chromosome, ~2% of the genome. We race on this because it finishes in booth time. |
| **AVX-512** | The wide vector maths units built into every Xeon core. What's doing the heavy lifting here. |
| **oneDNN** | Intel's deep-learning maths library. It's what actually chooses to use AVX-512, and the app reads its logs to prove it. |
| **SMT / hyperthreading** | Two work streams sharing one physical core. Deliberately *excluded* from the race — see §7. |

---

## 4. The floor script

### Opening (use when someone slows down near the bench)

> This is a real human genome being analysed, right now, on this server. No GPU, no
> accelerator card — just Xeon CPUs. Want to see it race itself?

Short, concrete, and offers a thing to watch. Don't lead with the ODM story or the spec
sheet; lead with the running workload.

### The race (your main set piece, ~8 minutes)

Set it up **before** you press the button, because the setup is what makes the result mean
something:

> I'm going to run the exact same job twice. Same genome, same software, same container,
> same settings. The *only* thing I change is how many of this machine's cores the job is
> allowed to touch — 16 cores, then 96. Watch the variant counts at the end: they'll be
> identical. That's the whole point — this isn't a speed-versus-accuracy trade. It's the
> same work, finished sooner.

Press **RACE: 16 cores vs 96 cores**. Then, while it runs, you have ~8 minutes of dwell
time. Use it for the qualifying conversation — what do they sequence, what volume, where
does analysis run today, what's their turnaround-time pain.

When the speedup card lands:

> 2.66×. And the two runs found exactly the same 210,390 variants. More cores bought
> time, and cost nothing in accuracy.

### The close

> Everything you just watched is on standard Xeon CPUs in an air-cooled box that can sit
> in your lab. If you want more throughput, you buy more cores — you don't rewrite
> anything, and you don't go hunting for GPUs.

---

## 5. Screen by screen

| Tab | What it is | Your line |
|---|---|---|
| **Status** | Machine and pre-flight health. Leave this up between visitors. | "96 cores, a terabyte of RAM, and the accelerators are already in the CPU." |
| **Dataset** | Pick the sample. Use *HG002 chr20* for the live race. | "A real human chromosome, at clinical-grade depth." |
| **Run console** | Live output, stage progress bars, and a DNA helix animation. | The helix is an **activity** indicator, not progress. It turns while the container produces output and freezes when output stops — so a stuck run *looks* stuck. Progress is the stage bars. |
| **Scaling race** | The set piece. 16 cores vs 96 cores. | See §4. |
| **Results** | Variant counts, SNPs, indels. | "Identical counts from both legs. Same answer, less waiting." |
| **Efficiency** | Power and cost model, with a samples-per-week slider. | **Say the word "illustrative" out loud.** These are calculated estimates from stated assumptions, not measurements — and the panel labels itself that way. Offer to plug in *their* volume; that's what makes it useful. |

**On the Efficiency tab specifically:** the assumptions are visible and editable in
config (400 W idle, 1100 W under load, $0.16/kWh, 20 usable hours/day). If someone
challenges a number, the correct response is enthusiasm, not defence: *"Those are the
stated assumptions — tell me yours and we'll put them in."*

---

## 6. The numbers, exactly

Quote these precisely or not at all. **Always attach the hardware** — a wall-clock time
without its machine is not a claim anyone can check.

**The bench server:** 2× Intel Xeon 6740P, 48 cores per socket, **96 cores / 192 threads**
total, 1 TB RAM, U.2 NVMe data disks.

**Whole genome (HG002, 35×, GRCh38, DeepVariant 1.10):**

| Configuration | Time | Variants |
|---|---|---|
| All 192 threads | **25m 47s** | 7,709,239 |
| 96 cores | 32m 59s | 7,709,239 |
| 16 cores | 2h 04m 51s | 7,709,239 |

Core-to-core that's **3.79×**. For scale: Google's own published figure for this version
is **69 minutes on a 96-vCPU cloud instance**.

**chr20 — what the booth race runs:**

| Leg | Time | Variants |
|---|---|---|
| 96 cores | **127.4 s** | 210,390 |
| 16 cores | **339.4 s** | 210,390 |
| Ratio | **2.66×** | identical |

**The detail worth knowing:** the whole-genome runs at 16 and 96 cores produced
**byte-identical output** — same 7,709,239 variants, 6,443,505 SNPs, 1,265,734 indels,
down to identical quality scores on the first variant. Core count changed the wall clock
and nothing else. That is a stronger statement than the ratio, and it's the one a
scientist will care about most.

---

## 7. Question bank

### Easy / expected

**"Is this real, or a recording?"**
> Real, running now. If we ever fall back to a recorded run, the app puts a large banner
> on screen saying so — we don't quietly replay.

**"Whose genome is that?"**
> HG002 — a public reference sample used across the industry for benchmarking. Not a
> patient, no privacy issue. It's the standard yardstick.

**"Why only a chromosome?"**
> So it finishes while you're standing here. The whole genome takes about 26 minutes; we
> have those numbers measured and I can show you them. chr20 gives you the same
> comparison in booth time.

**"How long for a whole genome?"**
> About 26 minutes using the whole machine. Google's published number for the same
> software on a 96-vCPU cloud instance is 69 minutes.

**"Does it need a GPU?"**
> No. This is CPU-only. The maths runs on AVX-512 — the vector units in every Xeon core.

### Commercial

**"What does this cost to run?"**
> Let's put your volume into the Efficiency tab. Those are illustrative figures from
> stated assumptions, and I'd rather run yours than quote mine.

**"How does this compare to a GPU server?"**
> Honestly? I don't have a head-to-head GPU benchmark on this bench, and I'm not going to
> invent one. What I can tell you is what this removes: no GPU procurement, no separate
> accelerator budget, no CUDA-specific stack to support, and it runs on servers you
> already know how to deploy. If the GPU comparison is the decision point for you, that's
> exactly the conversation I'd like to set up with our engineering team.

**Do not attempt to win the GPU argument.** You will be talking to people who have
benchmarked both. The honest framing above is genuinely strong — deployability and
supply, not raw peak throughput.

**"Can we scale beyond one box?"**
> Cluster-scale deployment is past what this bench demonstrates. What this proves is the
> per-node story. Let me connect you with someone who can talk about the cluster
> architecture properly.

**"Could we run our own data on it?"**
> Yes — a GRCh38-aligned BAM with its index. And it runs with the container fully network
> isolated, so your data has no route off this machine.

### Technical, and answerable

**"Why 2.66× and not 6×, if you gave it 6× the cores?"**

This question is a *good sign* — it's an engaged, competent listener. Answer it straight,
because the honest answer is more credible than the round number:

> Because it's not all parallel maths. The first stage builds those pileup images, and
> it's I/O-bound and Python-bound, so it doesn't scale linearly. We report 2.66× rather
> than rounding it up — the neural-network stage scales much better than the prep stage.

**"Is hyperthreading included?"**

> Deliberately not. Both legs of the race get whole physical cores, and neither gets a
> hyperthread sibling — otherwise we'd be crediting cores for a win that was partly SMT.
> We do measure SMT separately: it's worth about 1.28× on a whole genome and only 1.07×
> on chr20, because a whole genome has enough independent work to keep the sibling
> threads busy.

That answer will impress the right person considerably. It shows the benchmark was
designed by someone trying not to fool themselves.

**"How do you know AVX-512 is actually being used, and not just available?"**

> We ask oneDNN to report which instruction set it selected, then count how many compute
> kernels actually dispatched to AVX-512, and put both on screen. Available and actually
> used are two different facts, and we show them separately. If what it reports
> contradicts what we asked for, the app marks the run failed instead of showing a number.

**"What about AMX?"** — see the dedicated section below. This one has a trap in it.

**"Does it run offline?"**

> Fully. Every pipeline container runs with networking switched off entirely — not "we
> don't upload anything", but no network interface at all. It's enforced in the code and
> verified by test.

**"Is the comparison rigged? Different settings on each leg?"**

> The app won't let it be. Both legs are built from one specification, and it compares a
> fingerprint of every shared parameter before it will display a speedup. If anything
> other than the core count differs, you get no number.

---

## 8. The AMX question — read this before the show

Someone who follows Intel closely **will** ask why this doesn't use AMX, the matrix
engine in these Xeons. There is a real, documented answer, and there is a trap.

**The trap:** AMX sounds like the more advanced feature, so it is tempting to imply the
demo uses it, or would be faster with it. Both would be wrong, and this is a
well-informed question coming from a well-informed person.

**The honest answer, which is a good one:**

> AMX accelerates lower-precision integer and bfloat16 maths. The public DeepVariant model
> is fp32, and there's no fp32 path through AMX — so on this workload AMX contributes
> nothing. We checked properly rather than assuming: zero of the 95 compute primitives
> dispatched to AMX, and the measured difference was 0.98× — noise. AVX-512 is what runs
> fp32 convolutions, so AVX-512 is what's doing the work here.

If they push further — and some will — you have one more real fact:

> There is an older DeepVariant, 1.5, with a bfloat16 path that does light AMX up
> properly. We tested it. It's *slower* end to end: 314 seconds versus 123 for current
> fp32. So using AMX here would mean going backwards about 2.5×. We have the full
> engineering write-up if your team wants it.

Then stop and escalate. That write-up is `docs/AMX-FINDINGS.md`, it is internal, and
offering to have an engineer walk them through it is a genuinely valuable follow-up — it
demonstrates rigour rather than marketing.

---

## 9. Escalating gracefully

Handing off is a strength. The phrasing that works:

> "I want to give you a number you can actually build a plan on, and that one's outside
> what I've measured. Can I take your card and put you with the engineer who ran these?"

**Escalate on anything involving:**
- Clinical accuracy, sensitivity/specificity, F1 scores, or benchmarking against truth sets
- Regulatory, clinical validation, or accreditation
- Which pipeline suits their specific assay or panel
- Head-to-head GPU or competitor benchmarks
- Cluster or multi-node architecture
- Anything about a *patient* workflow

**Never:**
- Quote a runtime without stating the hardware it was measured on
- Present the Efficiency figures as measured — they are illustrative and labelled so
- Claim clinical validation, regulatory approval, or diagnostic suitability
- Round 2.66× up to "nearly 3" or "about 6× the cores so 6× the speed"
- Guess at a genomics term you don't know

---

## 10. If something breaks, mid-audience

Stay relaxed — a failure handled calmly costs you nothing, and this app is built to fail
visibly rather than silently.

| What you see | What to do | What to say |
|---|---|---|
| A run fails | Press START again | "Let's just run that again — one click." |
| Helix stops turning | Normal during long compute batches; the caption says how long output has been quiet | "It's mid-batch — the animation tracks output, not progress. The stage bars are the real progress." |
| UI unresponsive | Ctrl-C, then `./run_demo.sh --skip-checks` — about 2 seconds | "Give me two seconds to restart the front end." |
| Docker died | `sudo systemctl restart docker`, re-run | Same. |
| Hardware unavailable entirely | Switch `demo_mode.policy: always` in config for a recorded run | **Say it out loud:** "This one's a recording of a real run — the banner will tell you." The app displays it anyway. Never let the audience discover it themselves. |

Between visitors there is nothing to reset. Press the button again; every run writes to
its own directory. To clear the screen, switch to **Status** and back.

---

## 11. Cheat card

Print this bit.

**Pitch:** Real human genome analysis, deep learning, on plain Xeon CPUs. No GPU, no
accelerator card. Quiet enough for the lab.

**Numbers (bench: 2× Xeon 6740P, 96C/192T, 1 TB):**
- Whole genome: **25m 47s** (192 threads) — Google's published cloud figure is 69 min
- Race, chr20: **339.4 s** on 16 cores → **127.4 s** on 96 cores = **2.66×**
- Both legs: **210,390 variants — identical**
- Whole genome core-to-core: 2h 04m 51s → 32m 59s = **3.79×**, byte-identical output

**Three-beat close:** Throughput is already in the box → buy cores, not accelerators →
deploy it where the science happens.

**Two things to always say:** "illustrative" on the Efficiency tab; the hardware spec
whenever you quote a time.

**The escape hatch:** *"That's outside what I've measured — let me get you the engineer
who ran these numbers."*
