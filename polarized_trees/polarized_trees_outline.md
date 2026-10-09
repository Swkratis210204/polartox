# Polarized Trees: Paper Outline

**Abstract**
Subjective annotation can produce polarization, where annotators form opposing perspectives on the same item. This is not necessarily noise: it often reflects a deeper polyphony of legitimate viewpoints, and understanding it requires identifying the annotator characteristics behind each perspective. Existing approaches can test predefined socio-demographic (SCD) dimensions, but require guessing which ones matter and cannot reveal how several interact. We introduce Polarized Trees, a decision-tree framework that automatically discovers this structure by identifying the SCD dimensions and their intersections associated with opposing perspectives. Each split is chosen according to how strongly it explains the polarization, producing interpretable multidimensional subgroups and quantifying their contribution to it. Because real annotation data lack ground truth for the underlying structure, we first evaluate the method on synthetic data with known structure and then apply the selected configuration to a real-world dataset. The resulting trees show that polarization often cannot be explained by a single dimension, but emerges from intersections of several characteristics, with recurring patterns linking specific traits to particular perspectives. Since the splitting criterion decides every split, we also show how different formulations of it steer the explanation of the same polarized annotations: which traits are chosen, how deep the trees go, and which subgroups and poles emerge.

**1 Introduction**
- Disagreement in subjective annotation is meaningful, and when it follows annotator groups it becomes polarization.
- Current methods test one predefined trait at a time, so they need guesses and miss trait combinations.
- We propose Polarized Trees, which discover these combinations automatically and hierarchically.
- We validate on synthetic data with known structure, apply the selected configuration to DICES, and study how the PEG criterion steers the explanation of the same polarized annotations.
- We list our contributions in itemized form.
- **RQ1:** How can we explain annotation polarization multidimensionally, through the socio-demographic characteristics of annotators?
- **RQ2:** Given the same polarized annotations, how do different PEG criteria steer the explanation: which traits are chosen, how deep the trees go, and which subgroups and poles emerge?

**2 Related Work**
- **2.1 Disagreement as information:** annotator disagreement is signal worth keeping, not noise.
- **2.2 Measuring polarization:** DFU measures polarization. Existing attribution (AU) checks one trait at a time.
- **2.3 Annotator demographics:** traits shape annotations, but real data has no ground truth, which motivates synthetic evaluation.
- **2.4 Recursive partitioning:** decision trees and subgroup discovery, adapted here to a polarization objective.

**3 Data**
- **3.1 Synthetic data:** annotators with known traits and planted polarization, so we can check whether the method recovers it.
- **3.2 DICES:** a real, dense dataset with many annotations per item and annotator demographics.

**4 Method.** *Four stages: find → build → split/recurse → interpret.*
- **4.1 Finding polarized texts:** keep only texts that are polarized enough to explain.
- **4.2 Selecting a configuration and building one tree per text:** settings are chosen by a rule, not a fixed value. Whichever setting recovers the planted synthetic structure best is the one used to build a tree, one per text, starting at the root.
- **4.3 Splitting and recursing:**
  - *The criterion:* Polarization Explanation Gain (PEG): how much of a node's polarization a trait explains. Four variants exist (min, max, weighted, harmonic), plus a possible fifth: the mean of the three base variants (min, max, weighted), for a formulation that doesn't lean toward any single view of the split.
  - *The recursion:* at each node, split on the trait with the best PEG, drop it from that branch, repeat inside each resulting subgroup until size/depth/gain stopping rules trigger.
- **4.4 Labeling leaves and interpreting results:**
  - Each leaf gets a pole by majority vote.
  - Every setting, including PEG, is fixed at whatever performed best on synthetic data (§4.2), not at what's provably correct. PEG is the one we then vary: since it decides every split, changing it alone can change which traits get picked, how deep trees go, and how a text gets explained. §5 studies that behavior directly, from synthetic data to DICES, tracking its shifts, insights, and trends, not just which formulation "wins."
  - Reading that behavior off individual trees doesn't scale, so we aggregate across all of them into three interpretable summaries: 𝓕 (which traits get picked), 𝒞 (which groups lean to which pole), 𝒫 (which groups explain the most polarization). This aggregation is what lets us interpret results dataset-wide, per formulation, instead of tree by tree.

**5 Evaluation and Findings.** *No single check can fully validate an unsupervised method, so we check trust from several angles: synthetic vs. real data, quantitative vs. qualitative, formulation vs. hyperparameters. The goal is not to declare one PEG formulation "best." Instead, answering RQ2, we show how each formulation steers the explanation of the same polarized annotations: its behavior, its trade-offs, and why it behaves that way. This lets users pick what fits their needs. 5.1 supports RQ1 and the trust in the chosen configuration; 5.2–5.4 answer RQ2.*
- **5.1 Synthetic recovery, as a confidence check, not proof:**
  - Recovery is trustworthy on the data we built ourselves, where we planted the polarization and know its true drivers.
  - It says nothing directly about real datasets: those have no known polarization drivers and no ground truth to check against. We only borrow confidence from how well the method does here.
  - None of this makes DICES results untrustworthy by default. Whatever formulation or configuration we end up using, the patterns it surfaces on real data are still worth interpreting. A pattern doesn't need a validated method behind it to be useful. It needs to be read carefully and reported as what it is: a signal, not a proof.
  - This is where §4.2's rule gets applied: we run the search, see which setting wins, and report it. Several configurations score similarly, so the winner is "best on synthetic," not "correct." Of everything it sets, we single out the PEG formulation for closer study next.
  - *Experiment (exists):* Jaccard, precision, recall (mean/median/std/min), per synthetic corpus.
  - *Why:* quantify how much to trust the recovered structure.

- **5.2 Comparing PEG formulations, overall (RQ2: how deep and how similar):**
  - Question: on the same texts, with other hyperparameters fixed, does each formulation still need multiple traits to explain polarization, and how much does each explain? And where the formulations disagree, how much do they disagree, and on what?
  - *Experiment (planned):* tree depth, leaf count, and group size per formulation, plus accumulated-depth distribution per formulation, and pairwise tree similarity (ARI) between formulations.
  - *Statistical test (second tier):* a test on whether formulations differ (e.g., a permutation test on the ARI) is not part of the main experiments. It is deferred to future work, or run in response to reviewers (rebuttal) if requested.
  - *Why:* depth alone doesn't say whether formulations *agree* on structure.

- **5.3 Quantitative, dataset-wide results with 𝓕, 𝒞, 𝒫 (RQ2: which traits, subgroups and poles):**
  - *Problem:* a dataset produces one tree per text and that can scale to hundreds (like it did with Dices). Reading them one by one doesn't scale, and without a shared summary, subgroups from different trees, at different depths, can't be compared to each other at all.
  - *Solution:* three dataset-level metrics turn many individual trees into a small set of comparable numbers, one per formulation:
    - 𝓕: for each formulation, which trait is behind the split at each depth, and how often. Example: under PEGmax, age drives 50% of depth-1 splits but only 10% of depth-2 splits, while another trait takes over deeper in the tree. Read across depths, this tells us where people diverge the most at each stage: the first split is where the sharpest division sits, and whatever trait dominates deeper down is refining an already-narrowed disagreement, not opening a new one. Comparing this across formulations then shows whether they agree on where the strongest divide is, or locate it differently.
    - 𝒞: our goal is to find subgroups that consistently diverge to the same pole, not just once but systematically. One occurrence in a single tree isn't evidence, it's a coincidence. Pooling the same subgroup across every tree turns repeated occurrences into a signal, even if not a statistical guarantee. Example: "Man, Asian, Millennial" toxic in 10/10 occurrences, "Woman, Gen Z" civil in 8/8, both visible only once counted across the whole dataset.
    - 𝒫: same logic, applied to how much a subgroup's split explains. One split's polarization reduction, seen once, could be a fluke. If averaged over every time that subgroup appears across the dataset, it becomes a dataset-wide answer to which subgroups explain most of the polarization we find, and which explain almost none. And since it's an average over occurrences rather than a raw count, it stays comparable across subgroups found at different depths or in different trees.
  - *Why this matters:* trees from different texts are structurally incomparable on their own. A subgroup at depth 2 in one tree and depth 4 in another, or found under one formulation but not another, has no shared basis to be judged against another until it's expressed in the same terms. 𝓕/𝒞/𝒫 are three such shared terms. They are not the only ones that could be defined, and nothing stops us from introducing more dataset-level metrics later if a different comparison is needed.
  - *Experiment:* F/C/P tables, one set per PEG formulation.
  - *Why:* check whether formulations agree on *which* traits/subgroups matter, not just how deep they go.

- **5.4 Qualitative comparison across PEG approaches (RQ2: the same text, explained differently):**
  - Take the same handful of real texts and build a tree for each one under every formulation. For each text, walk through what each formulation actually did, which trait it split on first, whether it kept going or stopped and what subgroup and pole it landed on.
  - For each text, check whether that matches what we'd expect from the text itself, and interpret the result either way. The match is a sanity check, a mismatch is itself a finding worth explaining.
  - Example (illustrative, not a real result): a comment that's clearly misogynistic in content. We'd expect gender to show up early and the split to be decisive. If PEGmax does exactly that but PEGweighted needs a second trait to agree, that's a concrete, readable difference between the two, not an abstract one.
  - The aim is depth, not breadth: a close reading of a few texts, showing what each formulation does with them and why, and letting the disagreements between formulations (when they happen) speak for themselves.
  - *Experiment (planned):* the same set of texts, one tree per formulation per text, read side by side.
  - *Why:* numbers in 5.1–5.3 say formulations differ. This section shows what that difference actually looks like on real text.

**6 Discussion**
Polarization becomes a structure to explain rather than a number. The PEG criterion is a lens on that structure: different formulations steer the explanation of the same disagreement in different ways, so the choice should follow what the user needs. The method is practical (no training, interpretable), and findings apply to DICES only, not to the groups themselves.

**7 Conclusion**
Recap of the method and main findings, including how the PEG criteria steer the explanation.

**8 Future Work**
More data, stability tests, human evaluation and method extensions.

**9 Limitations**
Synthetic validation only, greedy splits, categorical traits, partial setting search, and no use of the text content itself.

**10 Ethics**
Real DICES demographics are used responsibly. The findings are associations, not causes. Small groups are suggestive only.

**Appendices**
- **A:** acronyms.
- **B:** why DFU over other polarization measures.
- **C:** an example real tree.
- **D:** how the synthetic data is generated.
- **E:** DICES statistics and how many distinct annotators back each group.
- **F:** settings and the search.
- **G:** more results: synthetic inference, other top configurations, and the full PEG-formulation results.
