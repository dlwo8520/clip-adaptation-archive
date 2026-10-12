# Publish the prepared repositories privately

The user selected local preparation and upload instructions. No GitHub repository was created and no push was attempted. The connected account was `dlwo8520`; the visible repository list had no matching research repository. The GitHub connector could list/read existing repositories, but exposed no repository-creation action. The server had no authenticated GitHub CLI.

Install the GitHub CLI through its official distribution, then authenticate interactively on your machine:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

From the parent directory containing both prepared repositories:

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

Both SHAs must match; visibility must be PRIVATE. Inspect README, results and excluded-file manifests on GitHub. Intended URLs are `https://github.com/dlwo8520/clip-adaptation-archive` and `https://github.com/dlwo8520/efcil-structure-research`; they are not verified existing repositories.

Command references: [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create), [interactive authentication](https://cli.github.com/manual/gh_auth_login), [Git credential setup](https://cli.github.com/manual/gh_auth_setup-git).
