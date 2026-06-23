# NyayaChakshu — Command Centre (static site)

Self-contained single-page app. All CSS, JS and images are inlined in `index.html`
— no build step, no external requests. Deploy as a static site.

## Deploy to Vercel (CLI)

```bash
cd nyayachakshu-console
npx vercel login      # one-time: pick GitHub / Google / Email, confirm in browser
npx vercel --prod     # deploys; accept the defaults — outputs your live URL
```

First run asks a few setup questions — accept defaults:
- Set up and deploy? **Y**
- Which scope? **(your account)**
- Link to existing project? **N**
- Project name? **nyayachakshu-console** (or anything)
- Directory with code? **./**  (just press enter)
- Modify settings? **N**

Re-deploy after edits: `npx vercel --prod` again.

## Alternative: GitHub + Vercel dashboard
1. Put this folder in a GitHub repo.
2. vercel.com → Add New → Project → import the repo.
3. Framework preset: **Other**. Root: this folder. Deploy.
