# Chilli Intelligence Web

The browser-accessible counterpart to the desktop app
(`chilli_desktop`, launched with `python -m chilli_desktop.main`). Same
workbook, same `chilli_desktop.data_loader` / `preprocessing` / `analytics` /
`forecasting` / `insights` modules, same 12 pages, same global filters, same
dark/light theme -- only the presentation layer differs (Dash instead of Qt
widgets). Nothing here is a redesign or a mockup: every chart, table and
forecast is computed by the exact same business logic the desktop app calls,
at request time, from `Chilli mastersheet for dashboard.xlsx`.

## Run locally

```bash
.venv\Scripts\pip install -r chilli_web\requirements.txt
.venv\Scripts\python.exe run_web.py
```

Open **http://127.0.0.1:8060**. This uses Flask's development server
(auto-reload on code changes) -- fine for local iteration, not for sharing.

## Run on the network (share with another computer on your LAN)

```bash
.venv\Scripts\python.exe run_web_production.py --host 0.0.0.0 --port 8060
```

`run_web_production.py` serves the same app through **waitress** (a
production-grade WSGI server -- no Werkzeug debugger, safe to expose).
Anyone on the same network can then open `http://<your-machine-ip>:8060`.
Find your IP with `ipconfig` (look for "IPv4 Address"). Make sure Windows
Firewall allows inbound connections on the port you chose.

## Deploy publicly

### Option A -- any host that runs a Python web service (Render, Railway, Fly.io, a VPS, ...)

The repo is deployment-ready:

- **`chilli_web/requirements.txt`** -- pinned dependencies.
- **`Procfile`** -- `web: python run_web_production.py --host 0.0.0.0 --port $PORT`
  (works on any Procfile-reading platform).
- **`render.yaml`** -- a [Render Blueprint](https://render.com/docs/blueprint-spec).
  Push this repo to GitHub, connect it at
  https://dashboard.render.com/blueprints, and Render builds/runs it
  automatically from this file.
- **`Dockerfile`** -- builds a self-contained image (workbook included) for
  any container host:
  ```bash
  docker build -t chilli-web .
  docker run -p 8060:8060 chilli-web
  ```

None of these were pushed live from this environment -- it has no
docker/git/gh/flyctl/render/railway CLI or account access. Pick a host, add
this repo (`git init && git add . && git commit` first if it isn't a repo
yet), and either connect `render.yaml` or point the host at the `Dockerfile`.

### Option B -- expose the machine that's already running it (Cloudflare Tunnel)

`tools/cloudflared.exe` is bundled in this repo. With
`run_web_production.py` already running locally:

```bash
tools\cloudflared.exe tunnel --url http://127.0.0.1:8060
```

This prints a random `https://<something>.trycloudflare.com` URL, live
immediately, with no Cloudflare account needed. It only stays up while both
`run_web_production.py` and the `cloudflared` process keep running on this
machine, and the URL changes every time you restart the tunnel -- fine for
sharing with a few people right now, not a substitute for a real host if you
need a stable, permanent address.

## Configuration

Copy `.env.example` to `.env` (or set these as real environment variables /
platform secrets):

| Variable | Purpose | Default |
|---|---|---|
| `CHILLI_WORKBOOK` | Full path to the master workbook | searches project root, then `chilli_desktop/` |
| `HOST` | Bind address for `run_web_production.py` | `127.0.0.1` |
| `PORT` | Bind port for `run_web_production.py` | `8060` (or the platform's `$PORT`) |
| `CHILLI_WEB_USERNAME` / `CHILLI_WEB_PASSWORD` | Enable HTTP Basic Auth | unset = no auth (matches the desktop app) |

**Authentication:** neither the desktop app nor this web port has a login
system by default. The workbook contains proprietary AgriWatch market data,
so set `CHILLI_WEB_USERNAME` and `CHILLI_WEB_PASSWORD` (both must be set)
before exposing this beyond a trusted network -- every request is then
challenged with a standard browser Basic Auth prompt. No secrets are ever
sent to or embedded in the frontend JavaScript; the check happens entirely
in `chilli_web/app.py` on the server.

## Public URL

_Fill in here once deployed to a permanent host:_ `https://` **(none yet --
see "Deploy publicly" above)**

## Architecture

```
chilli_desktop/   business logic (unchanged, still runs as the desktop app)
chilli_web/
├── app.py            Dash app factory, background-callback manager, optional Basic Auth
├── server_state.py   one shared DataService per process (same object desktop pages use)
├── layout_shell.py   sidebar nav + global filters shell (mirrors ui.py's MainWindow chrome)
├── components.py     reusable Dash components (cards, tables, notes) mirroring charts.py/insights.py output
├── plotly_charts.py  Plotly figure builders parallel to chilli_desktop/charts.py (matplotlib)
├── theme.py          reads chilli_desktop.settings.THEMES so colours match exactly
├── filters_io.py      FilterState <-> dcc.Store JSON round-trip (per-browser-session, not global)
├── pages/             one Dash page per desktop page (12), registered at matching routes
└── assets/            generated CSS (from theme.py) + client-side JS
```

The workbook is read once per process into a shared `DataService` (same
memoisation the desktop app's `MainWindow` uses); each browser's filter
selections live only in that browser's own `dcc.Store`, so multiple people
can use the app at once without stepping on each other's filters. The
**"Reload workbook"** button re-reads the file for everyone at once, exactly
like the desktop app's own reload action affects only the one window it runs
in.
