# Making JAMS workflows runnable from Jarvis

## Why the button didn't work

Your discovery workflow has two triggers:

| Trigger | Fires when | Reachable over HTTP? |
|---|---|---|
| `Daily 7am` (schedule) | its own clock | **no** |
| `Run manually` (manual) | you click *Test workflow* in n8n | **no** |

A manual trigger has no URL. Nothing outside the n8n editor — not Jarvis, not
curl, not cron — can start it. Only a **Webhook** node gives a workflow an HTTP
entry point.

## Fix: add a webhook trigger

For each workflow (discovery, responses, actions):

1. In n8n: **Workflows → ⋯ → Download** to export the JSON.
2. Patch it:

   ```bash
   python -m tools.n8n_trigger ~/Downloads/discovery.json --path discovery
   ```

   That adds a `POST /webhook/discovery` node wired to **the same node your
   manual trigger already feeds**, so firing it runs exactly what *Test
   workflow* runs. It's idempotent — safe to run twice.

3. Import the `.jarvis.json` back into n8n.
4. **Switch the workflow to Active.** An inactive workflow registers no
   webhook and the URL 404s. This is the single most common reason a correct
   setup still fails.

Repeat with `--path responses` and `--path actions`. Then reload
`/jams` — the buttons appear automatically for whatever answers.

## Checking it worked

```bash
./probe_jams.sh                        # what Jarvis can reach
curl -X POST localhost:5678/webhook/discovery
```

A registered webhook returns `{"message":"Workflow was started"}`. An
unregistered one returns a 404 saying the webhook is not registered.

## Paths that aren't the defaults

Jarvis looks for `discovery`, `responses` and `actions`. If yours differ:

```bash
JAMS_EXTRA_WEBHOOKS="discovery:job-discovery,responses:check-inbox"
```

## Why the webhook responds immediately

The added node uses `responseMode: onReceived`, so the HTTP call returns as
soon as the run starts. Discovery takes minutes; holding the connection open
that long means any timeout in between reads as a failure while the run is
actually fine. The JAMS button polls for the results instead and shows elapsed
time while it waits.

## One thing to fix while you're in there

The `Config` node holds your API keys as plain Set values. Those are exported
verbatim with the workflow — every download, backup and paste carries them.
Move them to n8n **Credentials** or `$env`. The patcher prints a redacted list
of what it finds so you can see which ones are affected.
