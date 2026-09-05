# ChargeShield — 5 minute video script

Keep this open on a second screen or phone while recording.
Short sentences. Pause between them. Do not rush.

Record each segment separately. Record **Segment 5 first**, while you are fresh.

---

## Segment 1 — Problem (15 seconds)

**Screen:** `demo/cards.html` fullscreen, then your README.

> A customer disputes a payment. The bank takes the money back.
> This is called a chargeback.
>
> The merchant has three bad choices. Give up and lose the money.
> Fight every case, and waste money on cases they cannot win.
> Or check every case by hand, which is slow and expensive.
>
> Nobody tells them which cases are worth fighting. That is the gap.

**Then say this. It is important:**

> One thing to notice. This works backwards from a normal fraud model.
> If fraud is likely, you should *not* fight it. You will lose, and you
> still pay the cost of fighting.

---

## Segment 2 — How it works (35 seconds)

**Screen:** `docs/ARCHITECTURE.md`

> Here is the design. The AI writes a suggestion. It does not decide anything.
>
> A separate piece of code makes the decision. I call it the policy engine.
> It is plain Python. It has no internet access and it cannot read files.
> Everything it needs is passed in as arguments.
>
> So the AI cannot cause an action on its own. That is not a promise in
> the README. There is a test that removes network access and file access
> from the code, and then checks the engine still works.

---

## Segment 3 — The rules running (70 seconds)

**Screen:** terminal, `make demo ARGS=--pause`

> This runs with no internet and no API key. Let me show three cases.

**Scenario 1 — missing evidence:**

> The delivery proof is missing. So we cannot prove the customer got the
> item. The engine sends this to a human instead of fighting it.

**Press Enter to scenario 5 — amount cap:**

> This dispute is ninety thousand rupees. That is above the limit I set.
> A human always reviews those, no matter what the score says.

**Press Enter to scenario 8 — slow down here:**

> Now the interesting one. Same fraud score, 0.88. Same evidence.
> Only the amount is different.
>
> At six thousand rupees, it goes to a human.
> At two thousand rupees, we just accept it.
>
> Why? Because fighting costs a fixed five hundred rupees. On a small
> dispute that is not worth it. On a bigger one it is. The cut-off comes
> from the cost, not from a number I picked.

**One line while it finishes:**

> This demo also checks itself. If I change a rule and forget to update it,
> it fails and the build breaks.

---

## Segment 4 — The main point (60 seconds)

**Screen:** `make demo ARGS="--only 2"`, then the browser page

> This is the case I care about most.
>
> Here all the evidence is present. So normally we would fight this dispute.
>
> But look at what the AI suggested. It says the customer passed a 3-D Secure
> check. That record does not exist. We never collected it. The AI made it up.
>
> The engine catches it. Rule two stops it, names the invented document,
> and sends the case to a human. The suggestion is rejected.

**Then say this immediately. Do not skip it:**

> I should be clear about one thing. I wrote that suggestion myself, to test
> the system. I also ran the real AI on the same case, and it did *not* make
> anything up. It correctly said the evidence was missing.
>
> So the check exists because an AI *might* do this. Not because mine did.

---

## Segment 5 — The numbers (70 seconds)

**Screen:** the page, sections 1 and 3

> Now the results. The test data is the last fifteen percent by date.
> The model never sees it during training.

> PR-AUC is 0.543. A simpler model gets 0.223 on the same data.
> Only three and a half percent of transactions are fraud, so 0.543 is
> about fifteen times better than guessing.
>
> I also set myself a warning line. If I ever got above 0.90, I would go
> looking for a mistake, not celebrate.

> On calibration I report 0.048, not 0.005. The smaller number looks better
> but it is misleading. Ninety-two percent of transactions score very low,
> where no decision is made. So I measure only the range where the engine
> actually decides.

**Now the honest number. Say it calmly:**

> And here is the result that does not flatter me.
>
> Across all disputes, my engine costs eighteen rupees *more* per dispute
> than a simple rule with no model at all.
>
> I report that first because it is the true number.
>
> But look closer. Where the engine has what it needs, it is sixty-nine
> rupees better. Where evidence is missing, it is eighty-three rupees worse.
> That is because it pays a hundred and fifty rupees for a human review,
> instead of filing a claim it cannot prove.
>
> I chose that. I would rather lose eighteen rupees than send a false claim
> to a bank.

---

## Segment 6 — Close (50 seconds)

**Screen:** `FAILURES.md`, then terminal, then closing card

> Seven things broke while building this. All of them are written down.
>
> The worst one. I reported that my engine saved eighty-four rupees per
> dispute. It did not. That number came from a different, simpler rule
> that I had labelled with my product's name.
>
> Every test passed while that was wrong. The tests checked the maths was
> correct. None of them checked that I was measuring the right thing.
> I found it, fixed it, and the real number is the minus eighteen I showed you.

**Final line, over `make all` or the test count:**

> Everything here can be checked. Clone the repository and run one command.
> Eighty-three seconds. Two hundred and twenty-four tests pass. And every
> number I showed you comes out the same, to six decimal places.
>
> Thank you.

---

## Reminders

- Speak slower than feels natural. Pause after each full stop.
- Do not say "unfortunately" or "I did not have time". Say "I cut that, and
  here is why."
- Do not apologise for the minus eighteen. It is your strongest moment.
- Same chair, same mic distance, all six segments, or the cuts will sound odd.
- Leave half a second of silence at the start and end of each segment.
