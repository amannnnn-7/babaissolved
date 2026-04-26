# Push to a personal GitHub account (with office credentials globally configured)

> Your global git config currently has:
> ```
> user.name  = Aman Paliwal
> user.email = t-apaliwal@microsoft.com
> credential.https://github.com.helper = !/usr/bin/gh auth git-credential
> ```
> The `gh` CLI is currently authenticated to your office account. We will
> override **both identity and credentials per-repo** so commits and pushes
> from this directory go to your personal account, **without touching the
> global config** that other repos depend on.

You only need to do **Option A** *or* **Option B**, not both. Option A is the
simplest if you don't already have a personal SSH key on this box.

---

## One-time prep (do this in either option)

```bash
cd ~/baba-rlvr

# 1. Per-repo identity — your commits will be authored as your personal account.
git config user.name  "<your-personal-display-name>"
git config user.email "<your-personal-email-or-noreply>"
# Tip: you can use GitHub's privacy-preserving noreply address, found at
#   https://github.com/settings/emails  (e.g. 12345678+username@users.noreply.github.com)

# 2. Stage everything we just produced and commit.
git add .gitignore pyproject.toml \
        ARCHITECTURE.md ROADMAP.md HANDOVER.md PUSH_TO_PERSONAL_GITHUB.md \
        scripts/ \
        src/baba_rlvr/engine/__init__.py \
        src/baba_rlvr/engine/world.py \
        src/baba_rlvr/levels/loader.py \
        src/baba_rlvr/levels/map_writer.py \
        levels/templates/

# Stage the YAML deletions explicitly so the repo no longer ships them.
git add -u levels/templates/

git commit -m "pivot: wrap C++ pyBaba engine; drop YAML levels for baba-is-auto .txt format

- vendor/baba-is-auto integrated as a clone-and-patch step (see scripts/setup_vendor.sh)
- engine/world.py rewritten as a pyBaba.Game adapter; preserves public World API
- levels: handcrafted maps converted to baba-is-auto integer-ID .txt format
- loader scans both custom and vendored maps (vendor_* prefix)
- pybind11 binding patches saved at scripts/patches/
- new docs: ARCHITECTURE.md, ROADMAP.md, HANDOVER.md
- all 29 tests still passing"
```

---

## Option A — HTTPS push with a Personal Access Token (recommended for office laptops)

A scoped token is the lowest-blast-radius way to push from a box that's
already logged into a different GitHub account.

### A.1 — Create the token (browser, ~30 s)

1. Open https://github.com/settings/tokens?type=beta in a browser logged into
   **your personal account**.
2. *Generate new token* → **Fine-grained personal access token**.
3. *Resource owner* = your personal account; *Repository access* = "Only
   select repositories" → pick the repo you're about to create (or "All
   repositories" if you prefer).
4. *Repository permissions* → **Contents: Read and write**, plus
   **Metadata: Read-only** (auto-selected). Nothing else.
5. *Expiration* = 7 days is plenty for the hackathon. Click **Generate**.
6. Copy the token (`github_pat_...`). You won't see it again.

### A.2 — Create an empty repo (no README, no license)

```bash
# In your browser: https://github.com/new
# Owner = your personal account, repo name = baba-rlvr (or whatever).
# Important: leave it empty (no README/.gitignore/license) so we can push.
```

### A.3 — Wire the remote and push (back in the terminal)

```bash
PERSONAL_USER="<your-personal-github-username>"
PERSONAL_REPO="baba-rlvr"
# Paste the fine-grained token you just created. Note the colon and `@` —
# the token replaces your password in the URL.
TOKEN="github_pat_paste_here"

git remote add origin "https://${PERSONAL_USER}:${TOKEN}@github.com/${PERSONAL_USER}/${PERSONAL_REPO}.git"

# Override the gh credential helper *for this repo only* so it doesn't
# inject your office account's credentials when pushing.
git config --local --replace-all credential.https://github.com.helper ""
git config --local credential.useHttpPath true

# First push.
git branch -M main
git push -u origin main
```

### A.4 — Clean up afterwards (optional but tidy)

```bash
# Remove the token from the remote URL so it's not stored on disk.
git remote set-url origin "https://github.com/${PERSONAL_USER}/${PERSONAL_REPO}.git"
# Future pushes will prompt for a username/password — paste username and
# the token as the password. Or store it via `gh auth login` (Option C).
```

If you accidentally pushed the token-bearing URL into a commit message or a
file, **revoke the token immediately** at the same GitHub settings page and
generate a new one.

---

## Option B — SSH push with a personal deploy key

Use this if you already have, or are happy to generate, a dedicated SSH key
for your personal account on this laptop.

```bash
# 1. Generate a key just for this account (don't reuse your office key).
ssh-keygen -t ed25519 -C "personal-github" -f ~/.ssh/id_ed25519_personal -N ""

# 2. Add the public key to your personal account:
#    https://github.com/settings/keys  (paste the contents of ~/.ssh/id_ed25519_personal.pub)
cat ~/.ssh/id_ed25519_personal.pub

# 3. Tell SSH to use that key for a personal-only host alias.
cat >> ~/.ssh/config <<'EOF'

Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# 4. Wire the remote using the alias.
PERSONAL_USER="<your-personal-github-username>"
PERSONAL_REPO="baba-rlvr"
git remote add origin "git@github-personal:${PERSONAL_USER}/${PERSONAL_REPO}.git"
git branch -M main
git push -u origin main
```

This isolates the personal key behind the `github-personal` alias, so it
never collides with your office GitHub Enterprise key.

---

## Option C — `gh` CLI multi-account (gh ≥ 2.40)

```bash
# Adds a second account alongside your office one. Auth flow is interactive.
gh auth login --hostname github.com --git-protocol https

# Confirm both accounts are visible.
gh auth status

# When pushing, prepend GH_TOKEN scoped to your personal account, OR rely
# on `gh repo create` (which uses the currently active gh account):
gh auth switch --user "<your-personal-github-username>"
gh repo create baba-rlvr --public --source=. --remote=origin --push
```

This rewrites the gh credential helper to your personal account *for this
shell session*. Useful if you'd rather not deal with PATs.

---

## After-push checklist

```bash
# Verify your commit looks right (author = personal email).
git log -1 --pretty=fuller

# Verify the remote URL is clean (no embedded token).
git remote -v
```

If anything pushed to the wrong account, tell me what happened — we can
amend the commit author, force-push to the correct repo, and revoke the
PAT/SSH key cleanly.
