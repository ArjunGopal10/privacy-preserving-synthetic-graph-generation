# Project website

Live site: **https://arjungopalevolveproject.netlify.app/** (interactive explainer: live randomized-response demo, modularity vs. epsilon, NSGA-II frontier, limitations, references).

## Status: source files are NOT included here
The website source (HTML/CSS/JS or framework project) was not available when this repository was assembled — only the deployed page could be read — so no site source is included and none is claimed.

## How to add it (no rebuild needed)
1. Copy the original site folder into this directory (`website/`), keeping its own `index.html` / `package.json`.
2. **Static site (plain HTML/JS):** run locally with `python -m http.server 8000 --directory website` and open http://localhost:8000.
   **Framework site (Vite/Next/Astro…):** `cd website && npm install && npm run dev`; build with `npm run build`.
3. **Netlify:** link the repo, set *Base directory* `website`, and the build command / publish directory of your framework (for plain static: no build command, publish directory `website`).
   **GitHub Pages:** Settings → Pages → deploy from a branch / GitHub Action using the `website` folder (plain static) or the framework's `dist` output.
4. If the site already lives in another repository, add it as a submodule instead: `git submodule add <site-repo-url> website/site`.

## Connecting the real experiment results
The page states its NSGA-II curves are browser simulations and that it will load `data/generation_history_nsga2.csv` and `data/generation_history_ga_combined.csv` if present next to the page. This repository produces the real pymoo run at `results/csv/nsga2_generation_history.csv` (columns: generation, hypervolume, front_size, best_expected_q, …). The site's expected CSV column names are not known to us, so the file may need renaming/reshaping before it loads.
