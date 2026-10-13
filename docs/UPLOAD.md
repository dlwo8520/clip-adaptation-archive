# GitHub publication and future updates

Both repositories were created and published as **private** repositories under `dlwo8520` on 2026-10-13 KST, after interactive GitHub CLI authentication. Their default branch is `main`. The initial uploaded heads were `120669c0fc0b503dc32f658b4e4d1db4234017b3` (CLIP) and `6d155e8b93f40dd90d158b16c68fb387e85750f1` (CIL). Remote main SHAs and all three curated CIL history tags were verified against the local refs. Large external evidence and private-history bundles remain local.

For future work, review and commit changes, then run `git push origin main` from the corresponding repository. Each local `main` tracks `origin/main`. Do not recreate these repositories.

If using another machine, install the GitHub CLI through its official distribution, then authenticate interactively:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

The following commands document the initial private publication. They have already been executed; do not rerun repository creation:

```bash
cd clip-adaptation-archive
gh repo create dlwo8520/clip-adaptation-archive --private --source=. --remote=origin
git push -u origin main
cd ../efcil-structure-research
gh repo create dlwo8520/efcil-structure-research --private --source=. --remote=origin
git push -u origin main
# Only the curated safe history tags; do not publish local private-history bundles.
git push origin 'refs/tags/history/*:refs/tags/history/*'
```

If a repository with the name already exists, inspect it with `gh repo view dlwo8520/NAME` and `git ls-remote` first. Reuse a verified intended empty repository by adding its URL as `origin`; for a nonempty repository, integrate history on a new branch and review before pushing. Never use force push, reset existing remote history or delete branches. Do not rerun `gh repo create` blindly.

Verify each upload:

```bash
gh repo view dlwo8520/NAME --json url,visibility,defaultBranchRef
git fetch origin main
git rev-parse main
git rev-parse origin/main
git status --short
```

Both SHAs must match; visibility must be PRIVATE. The verified repositories are [CLIP archive](https://github.com/dlwo8520/clip-adaptation-archive) and [CIL/EFCIL research](https://github.com/dlwo8520/efcil-structure-research). README, results and provenance manifests are versioned there. Publication confirms file and history transfer; it does not establish full GPU experiment reproducibility.

Command references: [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create), [interactive authentication](https://cli.github.com/manual/gh_auth_login), [Git credential setup](https://cli.github.com/manual/gh_auth_setup-git).
