# Hallucination Sentinel — Project Plan

## The Idea

A lightweight tool that detects LLM hallucinations using entropy analysis (CES algorithm). Based on the top-ranked Kurate.org paper "Entropy Distribution as a Fingerprint for Hallucinations in Generative Models."

---

## Thinking Skills Analysis

### 1. FIRST PRINCIPLES — What problem does this solve?

**The problem:** LLMs generate confident-sounding but factually wrong text. Nobody can tell the difference without manual verification.

**The solution:** Analyze the entropy (uncertainty) of token probabilities. Hallucinated text has higher entropy because the model is "guessing" rather than "knowing."

**Why existing solutions fail:**
- Multi-sample methods (generate 5-10 times, compare) = expensive, slow
- Human verification = doesn't scale
- Trust the model = dangerous

**Why CES works:**
- Single forward pass = fast
- Black-box = works with any model
- Formal guarantees = statistical confidence

---

### 2. INVERSION — What would kill this project?

**What would make this fail:**
1. The algorithm doesn't work in practice (paper says median AUROC ~0.65, which is modest)
2. API providers don't expose logprobs (some models don't give token probabilities)
3. Someone builds a better tool first
4. The OpenAI grant doesn't materialize

**How to mitigate:**
1. Test extensively, be honest about limitations
2. Support multiple backends (OpenAI logprobs, HuggingFace logits, local models)
3. Speed to market — build fast, ship fast
4. The grant is a bonus, not the only goal

---

### 3. SECOND ORDER EFFECTS — What happens after launch?

**Positive:**
- You become "the hallucination detection person" on Twitter
- Companies reach out for consulting/integration
- Contributors improve the tool
- Daemons gets a trust layer (see below)

**Negative:**
- Users expect maintenance
- API providers change logprob formats
- Competitors copy the approach

**The key insight:** The value is not the tool — it's the reputation. Even if the tool doesn't get massive adoption, you've demonstrated you can implement cutting-edge research.

---

### 4. OPPORTUNITY COST — What are we giving up?

**Time cost:** 2-3 days for OSS version

**What else could that time do?**
- Build another API wrapper (Listfix) = low value
- Build agent eval combo = medium value
- Study more papers = learning but no shipping

**Verdict:** This is the highest-value use of 2-3 days. The ROI is:
- $26,200 OpenAI grant (if stars materialize)
- Portfolio credibility
- Daemons trust layer
- Twitter reach

---

### 5. REVERSIBILITY — Can we change course?

**Highly reversible:**
- Can delete the repo anytime
- Can pivot to a different approach
- Can abandon if it doesn't get traction

**Not reversible:**
- Time spent (but only 2-3 days)
- Reputation if tool gives bad results (mitigate with honest docs)

---

### 6. LINDY EFFECT — Will this be useful in 5 years?

**The problem is Lindy-compatible:** As long as LLMs hallucinate, we need detection.

**The specific tool is NOT Lindy-compatible:** CES algorithm may be superseded by better methods.

**What IS Lindy-compatible:**
- The concept of entropy-based detection
- The CLI/API interface pattern
- The reputation as "hallucination detection person"

---

### 7. VIA NEGATIVA — What should we NOT include?

**Do NOT include:**
- ❌ Dashboard in v1 (CLI-first)
- ❌ Cloud hosting (let users run locally)
- ❌ Support for every model (start with OpenAI + HuggingFace)
- ❌ Custom data format (JSON output)
- ❌ Enterprise features (save for later)

**The bloat test:** If it doesn't directly detect hallucinations, cut it.

---

### 8. LEVERAGE POINTS — Where to add the most value?

**Highest leverage, lowest effort:**
1. **CLI tool** — `hallucination-sentinel check "text"` → risk score
2. **OpenAI integration** — Use logprobs API
3. **Clear README** — Examples, installation, usage
4. **Demo mode** — Show how it works without API key

**Lowest leverage, highest effort:**
- Dashboard
- Enterprise features
- Multi-model support

---

### 9. THEORY OF CONSTRAINTS — What's the bottleneck?

**The bottleneck is NOT coding.** The algorithm is clear.

**The bottleneck is:**
1. Getting logprobs from APIs (not all providers expose them)
2. Calibration data (need reference distributions)
3. Documentation and examples

**How to mitigate:**
1. Start with OpenAI (they expose logprobs)
2. Use the paper's calibration approach
3. Write clear docs with real examples

---

### 10. REGRET MINIMIZATION — Will we regret this?

**You will NOT regret building this.** Even if it gets 0 stars:
- You learned to implement a research paper
- You have a portfolio piece
- You can always pivot

**You WILL regret not building it.** The window is open:
- CES paper just published
- No good implementation exists
- OpenAI grant is available

---

## Daemons Connection

**How hallucination-sentinel connects to Daemons:**

Daemons = AI agents working for employees. These agents make decisions, write code, send emails, etc.

**The problem:** How do you trust that Daemons agents aren't hallucinating?

**The solution:** hallucination-sentinel as a Daemons module:

```
Daemon agent generates output
    ↓
hallucination-sentinel checks output
    ↓
If hallucination risk > threshold:
  - Flag for human review
  - Don't execute action
  - Log the risk
    ↓
If hallucination risk < threshold:
  - Execute action
  - Log confidence
```

**Daemons integration:**
1. **Trust layer** — Every agent output goes through sentinel
2. **Confidence scoring** — "This email is 92% likely to be accurate"
3. **Human-in-the-loop** — Flag uncertain outputs for review
4. **Audit trail** — Log all hallucination risks for compliance

**The tweet:**
> "hallucination-sentinel is the trust layer for AI agents. It checks if your AI is making things up before it takes action."

---

## Execution Plan

### Phase 1: OSS Version (Days 1-3)

**Day 1: Core Algorithm**
- [ ] Implement entropy calculation
- [ ] Implement CES formula (geometric mean of mean + max)
- [ ] Build calibration against reference CDF
- [ ] Unit tests

**Day 2: API Integration**
- [ ] OpenAI API (logprobs support)
- [ ] CLI interface with click
- [ ] JSON output format
- [ ] Demo mode (mock data)

**Day 3: Polish + Ship**
- [ ] README with examples
- [ ] Installation instructions
- [ ] Push to GitHub
- [ ] Tweet about it

### Phase 2: Growth (Days 4-14)

**Day 4-7:**
- [ ] Write blog post explaining CES
- [ ] Share on Hacker News, Reddit
- [ ] Engage with comments
- [ ] Fix bugs

**Day 8-14:**
- [ ] Apply for OpenAI grant
- [ ] Add HuggingFace support
- [ ] Add local model support
- [ ] Write documentation

### Phase 3: Product (Month 2+)

**Month 2:**
- [ ] Build hosted API
- [ ] Free tier (100 checks/day)
- [ ] Pro tier ($49/mo)

**Month 3:**
- [ ] Daemons integration
- [ ] Enterprise features
- [ ] Dashboard

---

## Tech Stack

**Language:** Python (matches research ecosystem, easy to install)

**Dependencies:**
- `openai` — API access
- `transformers` — Local model support
- `numpy` — Entropy calculations
- `scipy` — Statistical tests
- `click` — CLI interface
- `rich` — Terminal formatting

**Project structure:**
```
hallucination-sentinel/
├── README.md
├── PLAN.md (this file)
├── setup.py
├── requirements.txt
├── hallucination_sentinel/
│   ├── __init__.py
│   ├── cli.py          # CLI interface
│   ├── core.py         # CES algorithm
│   ├── openai_backend.py  # OpenAI integration
│   ├── hf_backend.py   # HuggingFace integration
│   ├── calibration.py  # Reference CDF calibration
│   └── utils.py        # Helpers
├── tests/
│   ├── test_core.py
│   ├── test_openai.py
│   └── test_calibration.py
└── examples/
    ├── basic_usage.py
    └── openai_example.py
```

---

## Success Metrics

**Week 1:**
- [ ] GitHub repo live
- [ ] 50+ stars
- [ ] 10+ Twitter engagements

**Month 1:**
- [ ] 200+ stars
- [ ] OpenAI grant application submitted
- [ ] 5+ contributors

**Month 3:**
- [ ] 500+ stars
- [ ] First paying customer
- [ ] Daemons integration live

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Algorithm doesn't work well | Be honest about limitations, test extensively |
| No logprobs from API | Support multiple backends, start with OpenAI |
| Competitor ships first | Speed to market, better docs |
| OpenAI grant rejected | Still have portfolio piece + reputation |
| No users | Focus on Twitter reach, Hacker News |

---

## The Tweet Thread

**Tweet 1:**
> I open-sourced hallucination-sentinel — a tool that detects LLM hallucinations using entropy analysis.
>
> Single forward pass. Black-box. Works with any model.
>
> Based on @KurateOrg top-ranked paper.
>
> github.com/Heman10x-NGU/hallucination-sentinel

**Tweet 2:**
> How it works:
>
> 1. Get token probabilities from LLM
> 2. Calculate entropy per token
> 3. Compute Calibrated Entropy Score (CES)
> 4. Flag high-entropy segments (potential hallucinations)
>
> No more guessing if the AI is making things up.

**Tweet 3:**
> This is the trust layer for AI agents.
>
> @DaemonsAI agents use hallucination-sentinel to check their outputs before taking action.
>
> If hallucination risk > threshold → flag for human review.
>
> If hallucination risk < threshold → execute.

---

## Next Steps

1. **You:** Review this plan, suggest changes
2. **Me:** Execute the plan (build the tool)
3. **You:** Push to GitHub, tweet about it
4. **Me:** Help with Daemons integration

---

## Questions for You

1. Should we start with Python or Go? (Python matches research ecosystem, Go matches your strength)
2. Should we include the paper's math in the README, or keep it simple?
3. Should we name the CLI `hallucination-sentinel` or shorter like `hs-check`?
4. How much should we emphasize the Daemons connection in the launch?

---

**Ready to execute when you approve the plan.**
