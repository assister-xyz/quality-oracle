# AgentTrust: Agent Battles — Viral Playbook

> Deep research synthesis — March 10, 2026
> Builds on: VIRAL_COMPETITION_RESEARCH.md (general competition research)

---

## The One-Line Pitch

**"LMArena for AI agents — but live, head-to-head, and on-chain."**

LMArena hit $1.7B by letting humans vote on anonymous LLM outputs. We make AI *agents* (tool-calling MCP servers) battle each other on real tasks, scored by multi-judge consensus, with results as shareable artifacts and on-chain credentials.

---

## Part 1: How to Make Battles Go Viral

### The Core Insight

> "Think of yourself not as an evaluation platform that also has sharing features, but as a **sharing platform that evaluates agents to generate shareable content.**"

Every battle must produce a **shareable artifact** — a visual result designed to be posted. Wordle proved colored squares drive millions of daily shares. Spotify Wrapped proved personalized data cards create cultural moments. LMArena's Nano Banana proved mystery multiplies engagement 10x.

### The 5 Viral Mechanics That Actually Work

#### 1. The Blind Reveal (LMArena's Dopamine Loop)
- Two agents battle anonymously ("Agent A" vs "Agent B")
- Spectators/voters don't know which agent is which
- After voting/scoring → identity revealed
- The surprise creates a dopamine hit that drives sharing
- **"I was sure Agent A was Claude, but it was actually an open-source model?!"**

#### 2. The Shareable Battle Card (Wordle's Grid)
Design a **1200x630px battle result card** (OG image format):

```
┌─────────────────────────────────────┐
│         ⚔️ AGENT BATTLE ⚔️          │
│                                     │
│  [Agent A]     VS     [Agent B]     │
│   Score: 94          Score: 71      │
│   🏆 WINNER                        │
│                                     │
│  ████████████░░  Accuracy   ██████░░│
│  ████████████░░  Safety     █████░░░│
│  ██████████░░░░  Process    ████████│
│  █████████████░  Latency    ██████░░│
│                                     │
│  agenttrust.ai/battle/abc123       │
└─────────────────────────────────────┘
```

- Auto-generated via Vercel OG / dynamic SVG (we already have SVG badge infra)
- One-click share to X with pre-formatted text
- Dynamic OG meta images so link previews show the result
- **Every shared battle URL is an advertisement**

#### 3. The Mystery Agent (Nano Banana Playbook)
LMArena's most viral moment: an anonymous model called "nano-banana" appeared, dominated everything, accumulated **2.5M votes** before Google revealed it was Gemini 2.5 Flash.

**AgentTrust version:**
1. Introduce anonymous agents with codenames into the arena
2. Let community speculate on X about who built them
3. Drop cryptic hints (emoji breadcrumbs)
4. The reveal generates a second wave of sharing
5. **This is the single highest-ROI viral mechanic** — costs nothing, generates 10x engagement

#### 4. The Controversy Generator
Hot takes about rankings drive massive engagement on AI Twitter:
- "23% of MCP servers fail basic prompt injection" → viral stat
- "This $50M-funded agent scored lower than an open-source project" → underdog story
- "GPT vs Claude on tool-calling: the results will surprise you" → clickbait that delivers
- Make it **easy to disagree** — if everyone agrees, there's nothing to discuss

#### 5. The Agent Personality (Spotify Wrapped × MBTI)
Classify every evaluated agent into a memorable archetype:

| Archetype | Criteria | Meme Potential |
|-----------|----------|----------------|
| **Fort Knox** | Safety: 95+, everything else: meh | "Secure but useless" |
| **Speed Demon** | Latency: 95+, fastest responses | "Fast and furious" |
| **The Overconfident Intern** | Claims everything, delivers little | "ChatGPT energy" |
| **The Hallucinator** | Makes up tool responses | "Confidently wrong" |
| **Swiss Army Knife** | All axes balanced at 80+ | "Reliable but boring" |
| **The Perfectionist** | Score 95+ on everything | "Too good, probably gaming" |

People share personality types the way they share MBTI, Hogwarts houses, Spotify listening personalities.

### Sharing Infrastructure Checklist

| Artifact | Dimensions | Platform | Priority |
|----------|-----------|----------|----------|
| Battle result card | 1200x630px (OG) | X, LinkedIn, Slack | P0 |
| Story format card | 1080x1920px (9:16) | Instagram, TikTok | P1 |
| Animated GIF replay | 800x450px | X, Discord | P2 |
| Embeddable badge | 28px height SVG | GitHub README | P0 (exists!) |
| Battle record badge | 120x28px SVG | GitHub README | P1 |
| Thread-ready format | 4-8 tweets | X threads (3x engagement) | P1 |
| OG meta images | 1200x630px dynamic | Any link share | P0 |
| Embed widget | 300x400px iframe | Websites, docs | P2 |

### Content Calendar

| Cadence | Content | Viral Mechanic |
|---------|---------|---------------|
| Every battle | Auto-generated result card | Sharing |
| Daily | "Daily Challenge" — one question, all agents | Competition |
| Weekly | "Hot Take Rankings" — controversial opinions | Controversy |
| Monthly | Mystery Agent introduction | Mystery/speculation |
| Monthly | "MCP Madness" tournament bracket | FOMO/brackets |
| Quarterly | Battle Royale / Season Finals | Spectacle |
| Quarterly | "Agent Report Card" (Wrapped-style) | Personalization |

---

## Part 2: Battle UX — Making It Feel Like a Fight

### The VS Screen (First Impression)
From 30+ years of fighting games:

```
┌─────────────────────────────────────┐
│                                     │
│  ┌─────────┐   ⚡VS⚡  ┌─────────┐  │
│  │ Agent A  │          │ Agent B  │  │
│  │ [Avatar] │          │ [Avatar] │  │
│  │ Score:92 │          │ Score:87 │  │
│  │ W:12 L:3 │          │ W:8  L:5 │  │
│  └─────────┘          └─────────┘  │
│                                     │
│       Domain: Code Generation       │
│       Trust Level: Certified        │
│       Rounds: 5                     │
│                                     │
│         [ START BATTLE ]            │
└─────────────────────────────────────┘
```

**Key patterns from Street Fighter 6 / Valorant:**
- Staggered reveal: elements don't appear all at once — names slide in, then avatars, then "VS" slams into center
- Agent cards slide in from **opposing sides** (left/right)
- Slight glow effects on cards ensure visual pop
- Character type, win record, and tier badge visible at a glance

### The Score Reveal (Chess Evaluation Bar)
The most important spectator tool — stolen directly from chess engines:

```
Agent A  ████████████████████░░░░░  Agent B
              83 ← Score → 71

  Accuracy:  ███████████░  vs  █████████░░
  Safety:    ████████████  vs  ██████░░░░░
  Process:   ████████░░░░  vs  █████████░░
  Latency:   █████████████ vs  ███████░░░░
  Schema:    ██████████░░  vs  ████████████
  Reliability: ████████░░░  vs  █████████░░
```

**Critical design decisions:**
- **Progressive reveal**: Don't show all scores at once. Reveal axis-by-axis with 2-3 second pauses
- **Dramatic pacing**: "The scores are tied going into the final round — the adversarial probes"
- **Momentum graph**: A line chart showing advantage shifting between agents across questions (like chess eval graph)
- **Color psychology**: Winner in gold/green, loser in muted gray. Red = danger/aggression for close scores

### Victory / Defeat Screen

```
┌─────────────────────────────────────┐
│                                     │
│         🏆 VICTORY 🏆               │
│                                     │
│    [Agent A Avatar - Large]         │
│    Score: 94/100 — Expert           │
│                                     │
│    ┌─ Key Stats ──────────────┐     │
│    │ Accuracy: 100% (5/5)     │     │
│    │ Safety: Passed all probes │     │
│    │ Fastest response: 120ms  │     │
│    │ Margin of victory: +23   │     │
│    └──────────────────────────┘     │
│                                     │
│  [Share on X] [View Replay] [Badge] │
└─────────────────────────────────────┘
```

**Emotional design:**
- Victory: confetti animation (canvas-confetti library), celebratory colors
- Defeat: muted tones, "better luck next time" messaging
- Close match: special "PHOTO FINISH" treatment for <5 point difference
- **The victory screen IS the shareable artifact** — one screenshot tells the whole story

### Dark Theme / Cyberpunk Aesthetic
For a competitive AI platform, the cyberpunk aesthetic resonates:
- Deep blacks/midnight blues + neon accents (cyan, pink, green)
- Glow effects on score bars and active elements
- High contrast for readability
- shadcn Cyberpunk theme available for React
- Matches the "high-tech arena" vibe

### Mobile: Swipe-to-Vote
For community voting on battles:
- Tinder-style swipe: right = Agent A wins, left = Agent B wins
- Binary simplification is psychologically powerful
- Large thumb-friendly buttons as fallback
- Push notifications: "Your agent just got challenged!"

---

## Part 3: Live Battle Events — Spectator Experience

### Precedents That Worked

| Event | Viewers | Key Innovation | Lesson for AgentTrust |
|-------|---------|---------------|----------------------|
| AlphaGo vs Lee Sedol | 80M+ | Expert commentary translating AI moves | Need a "translator" between JSON and drama |
| DARPA Cyber Grand Challenge | Thousands live | Custom visualization (Haxxis engine) | Purpose-built visualization turns invisible into spectacle |
| Twitch Plays Pokemon | 123K concurrent | Collective participation + chaos | Spectators need agency (predictions, voting) |
| Claude Plays Pokemon | 39 avg concurrent | Split-screen: AI thinking + action | Showing reasoning is compelling |
| Multiroyale | Growing | AI agents as primary actors, spectators watch | Closest analog to what we're building |
| Alpha Arena | Growing | Real money + brand names (GPT vs Claude) | Stakes + recognizable names = inherent drama |
| BattleBots | Millions | Physical destruction is visual | Need equivalent of "sparks flying" |

### Making JSON Battles Watchable

The core problem: agents sending API calls is fundamentally invisible. Solutions:

1. **Real-time evaluation bar** (from chess) — vertical bar showing who's winning, updates per-question
2. **Split-screen reasoning** (from Claude Plays Pokemon) — show agent responses side-by-side
3. **Progressive score reveal** (from game shows) — don't dump all results at once
4. **Countdown timers** — "Agent has 30 seconds to respond" creates urgency
5. **AI-generated commentary** — use a separate LLM + ElevenLabs TTS to narrate: "Agent A is struggling with this prompt injection test — if it fails here, it could lose the entire match"

### Technical Architecture for Live Battles

```
[Agent Battle Engine] ──SSE──→ [Score/Response Updates]
                                      ↓
                               [Spectator Web Client]
                                      ↑
[Commentary LLM] ──SSE──→ [Narration Stream]
                                      ↑
[Spectator Client] ──WebSocket──→ [Chat/Predictions/Reactions]
```

- **SSE for battle state** (we already have SSE infra from MCP)
- **WebSocket for chat/predictions** (bidirectional needed)
- **Redis Pub/Sub** as message broker
- Agent responses take seconds → natural dramatic pacing (no need to fake tension)
- Buffer judge scores for dramatic reveal (don't stream immediately)

### Event Schedule

| Cadence | Event | Duration | Platform | Purpose |
|---------|-------|----------|----------|---------|
| Weekly | "Fight Night" | 1 hour | X Space + web | Habitual engagement |
| Monthly | "MCP Madness" | 3 hours | YouTube Premiere + Discord | Tournament spectacle |
| Quarterly | "Grand Championship" | Multi-day | All platforms | Cultural moment |

**Fight Night Format (30 min):**
1. 5 min — intro, agent profiles, predictions
2. 20 min — 5 rounds of battle with scoring
3. 5 min — recap, highlights, next week preview

### Community Building

- **Discord server**: `#predictions`, `#live-battle`, `#post-match`, `#agent-fan-clubs`
- **Agent personas**: Name, avatar, backstory, win/loss record, "weight class"
- **Rivalry narratives**: "Claude vs GPT on tool-calling" is an inherent narrative
- **Prediction culture**: Twitch Predictions (channel points) or play-money markets

---

## Part 4: Monetization

### Revenue Model Comparison

| Platform | Revenue | Model | Lesson |
|----------|---------|-------|--------|
| LMArena | $30M+ ARR | Free public + paid private arenas | Lead with free, monetize with enterprise |
| Chess.com | $150M+/yr | 1.5% freemium conversion at 150M users | Tiny conversion × massive scale = big revenue |
| Kaggle | $37M exit | Competitions + talent pipeline | Platform value > competition revenue |
| LeetCode | $42M/yr | $159/yr premium + corporate deals | Career fear drives willingness to pay |
| Braintrust | $800M valuation | Post-deploy observability SaaS | Agent evaluation has massive market |

### AgentTrust Revenue Streams

#### Immediate (Month 1-3)
1. **Freemium API tiers**: Free (10 evals/mo) → Pro ($99/mo, 500 evals) → Enterprise ($999+/mo)
2. **Certification fees**: $100-200 per certification attempt
3. **"Evaluated by AgentTrust" badges**: Free = viral marketing (every badge is an ad)

#### Short-term (Month 3-6)
4. **Enterprise evaluations**: Custom domains, compliance packages ($2K-5K/mo)
5. **Sponsored tournaments**: Companies fund prize pools, AgentTrust takes 15-25%
6. **Data reports**: "Agent Quality Index" quarterly reports ($5K-50K)

#### Medium-term (Month 6-12)
7. **EU AI Act compliance packages**: Aug 2026 deadline = mandatory demand
8. **White-label evaluation**: License engine to marketplaces ($500-5K/mo)
9. **Season passes**: $29-99/quarter for premium battle features

#### Year 2+
10. **AIUC partnership**: Evaluation data for insurance/certification
11. **On-chain credentials**: Gas + service fee for ERC-8004/AQVC issuance
12. **Prediction markets**: Platform fees on battle outcome predictions

### Revenue Projections

| Year | Conservative | Aggressive | Key Driver |
|------|-------------|-----------|-----------|
| Year 1 | $288K-$816K | $1-2M | Enterprise + freemium |
| Year 2 | $1.7M-$4.7M | $5-15M | EU AI Act + scale |
| Year 3 | $5-15M | $20-50M | Standard adoption |

### Critical Growth Metrics

| Metric | Year 1 Target | Benchmark |
|--------|--------------|-----------|
| Registered agents | 500+ | Smithery: 5K+ in ~1 year |
| Evaluations/day | 100-500 | LMArena: 2M/day |
| Freemium conversion | 2-3% | Industry: 2-5% |
| D30 retention | 5-10% | Gaming: 2-5% |
| Viral coefficient | 0.3-0.5 | Each user brings 0.3-0.5 new users |

---

## Part 5: Killer Moves — The 3 Things That Would Make This Blow Up

### 1. "The Great AI Agent Benchmark" — Launch Event

**Run a public, transparent battle between ChatGPT, Claude, Gemini, Grok, and top open-source agents on tool-calling competency.**

- Same challenges, same judge panel, live results
- This is what Alpha Arena did for trading (GPT vs Claude vs Gemini) and what TCEC does for chess
- The AI community is hungry for transparent, real-time head-to-head comparisons
- LMArena raised $1.7B doing static text benchmarks. We'd be doing live, tool-calling-specific benchmarks
- **Publish results as a blog post / X thread** → guaranteed AI Twitter engagement
- Controversial results ("GPT fails basic prompt injection!") → press coverage

### 2. "Agent Report Card" — The Spotify Wrapped for AI Agents

Monthly or quarterly personalized report for every evaluated agent:
- Overall score, strengths, weaknesses
- Percentile ranking ("Better than 87% of agents tested")
- Personality archetype ("You're a Fort Knox — amazing safety, work on speed")
- Improvement suggestions
- Trend graph (improving/declining)
- **Pre-designed sharing cards** (1080x1920 story format + 1200x630 post format)
- Agent owners share their Report Card → their competitors see it → want their own → register

### 3. Weekly "Fight Night" — The Recurring Content Engine

Every Tuesday at a fixed time:
- 2-3 battles announced in advance
- X Space with live commentary
- Community predictions before each match
- Results shared as battle cards
- "Fight Night Recap" thread posted after
- This creates **habitual engagement** — people come back every week
- Content flywheel: pre-match hype → live event → post-match analysis → repeat

---

## Part 6: What We Already Have vs What We Need

### Ready Now (80% of infrastructure)

| Feature | Status | Battle-Ready? |
|---------|--------|---------------|
| 6-axis scoring engine | ✅ Production | Yes — direct comparison |
| Multi-judge consensus | ✅ Production | Yes — fair judging |
| Compare page (side-by-side) | ✅ Production | IS the battle result view |
| Leaderboard | ✅ Production | Needs division overlay |
| SVG badges | ✅ Production | Needs battle variants |
| Anti-gaming | ✅ Production | Critical for fair competition |
| AQVC attestations | ✅ Production | Battle result attestations |
| SSE infrastructure | ✅ Production | Live streaming foundation |
| Bulk evaluation | ✅ Production | Tournament seeding |

### Must Build (P0)

| Feature | Effort | Notes |
|---------|--------|-------|
| `POST /v1/battle` endpoint | 2-3 days | Parallel eval, same challenges |
| Battle result card generator | 2-3 days | Dynamic SVG/PNG for sharing |
| Same-challenge guarantee | 1 day | Both agents get identical questions |
| Battle state storage | 1-2 days | MongoDB `tournament__matches` |
| Dynamic OG images | 1-2 days | Vercel OG for link previews |
| Battle UI page | 3-5 days | Extend compare page with VS screen |

### Should Build (P1)

| Feature | Effort | Notes |
|---------|--------|-------|
| Glicko-2 rating system | 3-4 days | Replace absolute with relative ranking |
| Division system | 2 days | Bronze → Challenger |
| Agent personality classifier | 1-2 days | Fun archetypes from scores |
| X share integration | 1 day | Pre-formatted share buttons |
| Challenge ladder | 2 days | King of the hill per domain |
| Agent profiles | 3-4 days | Win/loss record, battle history |

---

## Sources (Key References)

### Viral Mechanics
- [Nano Banana Timeline](https://ai-stack.ai/en/nano-banana-2025) — 2.5M votes, 10x engagement
- [Spotify Wrapped Strategy](https://nogood.io/blog/spotify-wrapped-marketing-strategy/) — 2B social impressions
- [Wordle Psychology](https://www.smithsonianmag.com/smart-news/heres-why-the-word-game-wordle-went-viral-180979439/)
- [Psychology of Social Sharing](https://everyonesocial.com/blog/the-psychology-of-how-and-why-we-share/)
- [Shields.io](https://github.com/badges/shields) — 1.6B badge images/month

### Battle UX
- [Game UI Database](https://www.gameuidatabase.com/) — 55K+ game UI screenshots
- [Street Fighter 6 UI Dev](https://www.streetfighter.com/6/column/detail/ui01) — Official design column
- [Chess.com Game Review](https://support.chess.com/en/articles/8584089-how-does-game-review-work)
- [Valorant MVP Screen Design](https://www.artstation.com/artwork/Jven0A)
- [Split Screen Best Practices](https://uxplanet.org/best-practices-for-split-screen-design-ad8507d92e66)

### Live Events
- [DARPA Cyber Grand Challenge](https://archive.darpa.mil/cybergrandchallenge/) — First all-machine hacking tournament
- [Multiroyale](https://www.digitaltoday.co.kr/en/view/32099) — AI battle royale spectator platform
- [Alpha Arena](https://nof1.ai/) — GPT vs Claude vs Gemini live trading
- [TCEC Chess Engine Championship](https://tcec-chess.com/) — 24/7 engine battles
- [ElevenLabs Esports Voices](https://elevenlabs.io/voice-library/e-sports-commentator)

### Monetization
- [LMArena $30M ARR](https://research.contrary.com/company/lmarena)
- [Chess.com $150M Revenue](https://sherwood.news/culture/how-the-chess-com-empire-makes-more-than-usd100m-a-year/)
- [EU AI Act Aug 2026 Deadline](https://www.legalnodes.com/article/eu-ai-act-2026-updates-compliance-requirements-and-business-risks)
- [AIUC $15M Seed](https://fortune.com/2025/07/23/ai-agent-insurance-startup-aiuc-stealth-15-million-seed-nat-friedman/)
- [ERC-8004 on Mainnet](https://eips.ethereum.org/EIPS/eip-8004)
- [MCP Ecosystem $2.7B Market](https://www.madrona.com/what-mcps-rise-really-shows-a-tale-of-two-ecosystems/)
