# Team quickstart — running the demo at the booth

This is the short version for anyone staffing the booth who just needs to run the demo,
not build or troubleshoot it in depth. For the full picture see
[`README.md`](../README.md) ("Booth-day runbook") and
[`docs/PRESENTER-GUIDE.md`](PRESENTER-GUIDE.md) (talking points and visitor Q&A).

**Server:** `cg2600remote1` — it boots and starts the demo automatically. You should not
normally need to start anything yourself.

## To show the demo

1. On the booth machine, open a browser to **http://localhost:7860**
2. Go full screen (F11)
3. Walk the demo:
   - **Status** tab — show the hardware (96 cores, 1 TB RAM)
   - **Dataset** tab — select **HG002 chr20**
   - **Scaling race** tab — press **RACE: 16 cores vs 96 cores**, wait ~6–8 min for both legs
   - **Results** — same variant counts either way; the full machine wins on time, not accuracy

## If the demo isn't loading

1. Check the machine is powered on and networked.
2. Wait about a minute after any reboot — it starts itself, no login needed.
3. If it's still down, from a terminal on that box:
   ```bash
   sudo systemctl status genomics-demo
   sudo systemctl restart genomics-demo
   ```
4. If Docker itself is the problem: `sudo systemctl restart docker`, then restart the
   service above.

## Between visitors

Nothing to reset — just click the race button again. Each run writes to its own folder,
so nothing gets overwritten or needs clearing.

## If a run fails mid-demo

The error shows on screen. Just press **START** again.

## End of day

Nothing required — leave it running. It's safe to power off, power on, or reboot the
machine at any time; it comes back on its own.

## Who to call

If `sudo systemctl status genomics-demo` doesn't show `active (running)`, or the page
still won't load after a restart, escalate rather than debugging live during show hours.

<!-- TODO: add booth contact name/number here -->
