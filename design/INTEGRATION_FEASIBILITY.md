# NUDGE ⇄ Claude integration — feasibility memo (Phase A gate)

**Question the build hangs on:** *Can NUDGE actually be driven by Claude
("Claude Science" / the Life-Sciences hackathon) through a custom MCP server —
and if so, what is the exact, real connection recipe?*

**Verdict: YES — verified, not assumed.** A custom, self-authored MCP server is a
first-class, currently-shipping extension point across every relevant Claude
surface, including the **Claude Science** workbench that `WORKING_BACKWARDS.md`
Part 2 targets. A single `stdio` server (the standard local transport) is
directly connectable from Claude Code, Claude Desktop, **and Claude Science's
"Local command" connector**; a remote HTTPS deployment additionally reaches
`claude.ai` web. This memo records the exact recipes and the caveats, so Phase D
builds to what was verified.

Researched 2026-07-09 via the official docs below and cross-checked with the
`claude-code-guide` agent. Where a fact is version-dependent or unverifiable it
is flagged as such.

---

## 1. The three questions, answered

### Is MCP the right surface? — Yes.
The Model Context Protocol is Anthropic's standard for exposing tools to Claude.
A server declares tools; a Claude *client* connects to it and the model calls the
tools in plain language. NUDGE's verbs (`attribute`, plus the abstention/card
lookups) map cleanly onto MCP tools — this is exactly the "thin adapter over the
same engine" that Part 2 of `WORKING_BACKWARDS.md` describes.

### Does Claude Science accept a custom server? — Yes, both transports.
The Claude Science **Custom connectors** doc is explicit:

> "Add any Model Context Protocol (MCP) server as a **Remote (HTTPS web server)**
> or **Local command (program on your computer)**." — *Settings › Connectors ›
> Add connector*, name = lowercase/digits/hyphens. Remote takes a URL (transport
> **SSE** or **Streamable HTTP**, optional OAuth); Local command takes the
> command + args + env. "Local-command connectors run inside the sandbox … with a
> per-connector writable directory."

So NUDGE does **not** need to be hosted to be used inside the workbench — a local
`stdio` command connector works. (Secrets in a local connector's `env` are stored
unencrypted per-account; NUDGE needs none, so this is moot for us.)

### Is stdio enough, or must we host? — stdio covers 3 of 4 surfaces.
`stdio` (a subprocess speaking over stdin/stdout) is the default local transport
and is accepted by Claude Code, Claude Desktop, and Claude Science's Local-command
connector. Only **`claude.ai` web** is remote-only (it cannot spawn a local
process), so reaching the browser app requires hosting the server over HTTPS
(Streamable HTTP / SSE). **Recommendation: ship the `stdio` server now** (real,
runnable, zero-infra, demoable today); document the remote path as a follow-on.

---

## 2. Exact connection recipes (copy-paste)

NUDGE exposes the entry point `nudge-mcp` (see `pyproject.toml` `[project.scripts]`)
which runs the FastMCP `stdio` server in `nudge.mcp.server`. Equivalent:
`uv run nudge-mcp` or `uv run python -m nudge.mcp.server`.

### A. Claude Code CLI (fastest to demo)
```bash
# from the repo root, project scope (writes ./.mcp.json, shareable with the team)
claude mcp add --scope project nudge -- uv run nudge-mcp
claude mcp list          # expect:  nudge  ✓ Connected
# then, in a session:
#   "Use the nudge server to list its mechanisms, then explain a gain⇄threshold abstention."
```
- No `--transport` flag ⇒ default `stdio`. Everything after `--` is the launch
  command. Scopes: `local` → `~/.claude.json` (this project, you); `project` →
  `./.mcp.json` (committed, team); `user` → `~/.claude.json` (all your projects).
- Verify per-server detail / errors: `claude mcp get nudge`.

A ready-to-commit project entry (`.mcp.json`) is shipped at the repo root:
```json
{ "mcpServers": { "nudge": { "type": "stdio", "command": "uv",
  "args": ["run", "nudge-mcp"] } } }
```

### B. Claude Desktop
Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`;
Windows: `%APPDATA%\Claude\`; Linux: the XDG config dir), then fully quit + reopen:
```json
{ "mcpServers": { "nudge": { "type": "stdio", "command": "uv",
  "args": ["run", "--directory", "/abs/path/to/NUDGE", "nudge-mcp"] } } }
```

### C. Claude Science workbench (the on-thesis surface)
*Settings › Connectors › Add connector › **Local command***. Name `nudge`,
command `uv run nudge-mcp` (working dir = the repo). Each tool starts at **Ask
each time**; set `attribute` etc. to *Always allow* once trusted. (Or **Remote**
with the hosted URL from §D.)

### D. claude.ai web (requires hosting — follow-on)
Host the server over HTTPS with Streamable HTTP transport
(`mcp.run(transport="streamable-http")` in the SDK), then *claude.ai/customize/
connectors › Add custom connector › URL*. Free tier = 1 custom connector; Pro/Max/
Team/Enterprise more. Beta. Connectors added here also auto-load in the Claude Code
CLI when signed in with the same account.

---

## 3. The SDK we build on

- **Package:** `mcp` (PyPI `mcp`), already declared as the optional extra
  `nudge-bio[mcp]` in `pyproject.toml`. Import guarded so the core install stays
  light.
- **Idiom:** the `FastMCP` decorator API (`from mcp.server.fastmcp import
  FastMCP`; `@mcp.tool()`; `mcp.run(transport="stdio")`). Minimal, stable, and the
  currently-recommended way to author a server.
- **Transports the SDK supports:** `stdio`, Streamable HTTP, SSE. NUDGE ships
  `stdio` and leaves a one-line switch to Streamable HTTP for the hosted path.
- **Version caveat (flagged):** an SDK v2 reorganization (FastMCP → `MCPServer`)
  was reported around the 2026-07 spec refresh; the `@mcp.tool()` decorator idiom
  is described as backward-compatible. We pin `mcp>=1.0` and import defensively;
  if `mcp.server.fastmcp` moves, the adapter is ~40 lines to retarget. **Not a
  blocker**, but a known churn point.

---

## 4. Recommended integration architecture

```
        nudge CLI (typer)                 nudge MCP server (FastMCP, stdio)
   load / attribute / mechanisms / explain     attribute · explain_abstention
              │                                 list_mechanisms · get_mechanism_card
              └──────────────┬──────────────────────────┘
                             ▼   both are THIN adapters over the tested API
      inference.pipeline.attribute_across_operating_points · lyapunov.* ·
      data.loaders.* · data.decoys.DECOY_BATTERY · docs/mechanism_cards/*
```
Both the CLI and the MCP server are thin, no-logic-of-their-own layers over the
same tested engine and the same Mechanism-Card knowledge base — so Claude gets
exactly the honest, abstaining output a human gets, with the card + decoy that
explains any abstention. Build order: CLI first (core, testable headless), MCP
server as the same verbs re-exposed, skills composing both.

**Bottom line:** the integration imagined in `WORKING_BACKWARDS.md` is real and
buildable as specified. Proceed to Phase D with a `stdio` FastMCP server; hosting
is an optional reach for the browser app, not a prerequisite.

---

## Sources
- Claude Science — Custom connectors: <https://claude.com/docs/claude-science/custom-connectors>
- Claude Code — Connect to MCP servers (add/scope/verify, `.mcp.json`): <https://code.claude.com/docs/en/mcp-quickstart.md>
- Custom connectors via remote MCP (claude.ai): <https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp>
- MCP Python SDK: <https://github.com/modelcontextprotocol/python-sdk> · <https://pypi.org/project/mcp/>
- MCP spec / remote servers: <https://modelcontextprotocol.io/docs/develop/connect-remote-servers>
