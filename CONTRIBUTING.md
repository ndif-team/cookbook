# Implementation Guide

## Overview
The purpose of this document is to define the expectations for executing and submitting paper implementations within the NDIF Cookbook project. This includes guidance on claiming papers, tooling, repository organization, acceptance criteria, and review process.

## Claiming a paper
- Look through the papers spreadsheet, any rows with an empty A cell are free game
- State "I'm claiming (paper title)" in your channel, then woog will mark it on the spreadsheet. If two people attempt to claim the same one, it goes to who claimed first.

## Compute and tooling
- You must use [nnterp](https://github.com/ndif-team/nnterp) to perform all experiments. Comes bundled with [nnsight](https://github.com/ndif-team/nnsight)
- You must sign up for an NDIF api key here: https://login.ndif.us/
- Preferably you sign up using a gmail account
- Under user profile, if you have nothing original to say enter these defaults:
  - Research Fields: Computer science > Mechanistic Interpretability
  - Research Topic: Helping woog with implementing mech interp papers
  - Use Case: Patch scopes, causal interventions, whatever methods come up in the paper
  - Alternatively, copy paste the 1-2 sentence description in column i corresponding to the paper you claimed
  - Estimated Usage: Daily sparse use from now until 2 weeks from now, potentially longer
- At the "US-based User Attestation" part if you are not US-based, pick option 2: you have a US based researcher you are collaborating with
  - DM @woog on Discord to get their information for this part (personal details not included here to keep this document publicly shareable)
- After signing up, share the email you registered with in your personal channel
- Within 24 hours, your key will be upgraded to support remote with any HF model
- See https://nnsight.net/notebooks/features/remote_execution/ to get started

## Submission Process
1. Fork repo, create branch: `replications/paper-name`
2. Complete your replication with required files (see below)
3. Fill out checklist in your PR
4. Submit pull request

## Repository structure
```
root/
├── replications/
│   ├── paper_name/
│   │   ├── README.md           # MMVE results file
│   │   ├── paper_name.ipynb    # Core Jupyter Notebook
│   │   ├── paper_name.py       # Core script
│   │   ├── utils.py            # Helper functions (if needed)
│   │   ├── config.json         # Stores experiment hyperparameters
│   │   ├── figures/            # Any generated plots or graphs
│   │   ├── results/            # Outputs and logs
│   ├── ... (other paper implementations)
```

## Submission checklist

### Required
- README.md with setup and usage instructions included
- Implemented scope declared in README.md and at top of main.ipynb notebook
  - Which experiments, figures, claims are implemented
  - Which are not implemented (with reason)
- Notebook runs without errors if ran in Google colab on free cpu mode
- README links to view-only colab run through
- Uses nnterp methods (logit lens, attention_probabilities, etc.) when possible
- Uses nnsight for all interventions when nnterp is unavailable
- Uses NDIF remote execution
- No hard-coded local paths or credentials
- Fixed seeds
- Which model used, whether it matches the paper, explanation provided if different
- Which datasets used, their size, if you ran on a smaller subset
- Equivalent prompt formatting
- Key result reproduced (matching paper's qualitative or quantitative claim)
- Output artifacts saved (figures, tables, logs, serialized results)
- Docstrings for all major functions and classes
- Comments explaining non-obvious implementation choices
- Comparison table showing replicated vs original results
- Notes on any deviations from the original paper (with justification)
- No dead or commented-out experimental code

## Review process

### Reviewer responsibilities:
- Verify execution: Run notebook in colab. Does it actually work?
- Validate checklist: Check that marked items are genuinely complete
- Assess scope: Does it reproduce at least 1 core claim from the paper?
- Check quality: is the code readable? Results honestly compared? Deviations explained?
- Comment appropriately: use the Github pull request conversation features to give feedback to the author.

### Approval criteria:
- Checklist items marked complete are actually complete
- Reproduces at least one key result that tests the paper's main claim
- Code quality allows others to understand and build on it
- Partial implementations clearly document what's missing and why

### Review outcomes:
- **Approve**: Merge (label as complete or partial in spreadsheet)
- **Request changes**: specific changes needed (e.g., "Notebook crashes at cell 5", "Need justification for using smaller model")
- **Suggest rescope**: Implementation too limited, recommend claiming different paper or expanding scope

### For partial implementations:
- Create GitHub issue tracking remaining work (label: help-wanted)
- Update papers spreadsheet status to Partial with link to PR
- Add implementation status section to paper's README

### Common rejection reasons:
- Doesn't use nnterp/nnsight
- Checklist marked complete but notebook crashes
- Implemented trivial subset without justification
- Code is unreadable or poorly documented
- Uses different model/dataset with no explanation of impact
- Absent figure generation