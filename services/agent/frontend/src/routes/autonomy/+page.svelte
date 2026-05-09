<script>
	import { onMount, onDestroy } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import AgendaForm from './AgendaForm.svelte';
	import SpeakToDevice from '$lib/components/SpeakToDevice.svelte';

	let status = $state(null);
	let events = $state(null);
	let items = $state([]);
	let runs = $state([]);
	let liveRuns = $state([]);
	let awaiting = $state([]);
	let error = $state('');
	let loading = $state(true);

	// Run history filters
	let filterKind = $state('');
	let filterStatus = $state('');
	let filterSource = $state('');

	// Form modal
	let formOpen = $state(false);
	let formItem = $state(null);

	// WS state
	let ws = null;
	let wsStatus = $state('connecting');
	let reconnectTimer = null;

	async function fetchJSON(path, init) {
		const r = await fetch(path, init);
		if (!r.ok) throw new Error(`${path} → ${r.status}`);
		return r.json();
	}

	async function refresh() {
		try {
			const [s, ev, it, rr, aw] = await Promise.all([
				fetchJSON('/api/autonomy/status'),
				fetchJSON('/api/autonomy/events/summary'),
				fetchJSON('/api/autonomy/items'),
				fetchJSON(buildRunsPath()),
				fetchJSON('/api/autonomy/runs/awaiting'),
			]);
			status = s;
			events = ev;
			items = it.items ?? [];
			runs = rr.runs ?? [];
			awaiting = aw.runs ?? [];
		} catch (e) {
			error = e.message;
		} finally {
			loading = false;
		}
	}

	async function confirmRun(runId, approved) {
		const params = new URLSearchParams(location.search);
		const token = params.get('token') ?? null;
		try {
			const r = await fetch(`/api/autonomy/runs/${runId}/confirm`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ approved, token }),
			});
			if (!r.ok) {
				const text = await r.text();
				throw new Error(`${r.status}: ${text}`);
			}
			// Clear deep-link query params so a reload doesn't re-pop the banner.
			if (params.has('confirm') || params.has('token')) {
				history.replaceState(null, '', location.pathname);
			}
			await refresh();
		} catch (e) {
			error = e.message;
		}
	}

	function buildRunsPath() {
		const qs = new URLSearchParams();
		qs.set('limit', '50');
		if (filterKind) qs.set('kind', filterKind);
		if (filterStatus) qs.set('status', filterStatus);
		if (filterSource) qs.set('trigger_source', filterSource);
		return `/api/autonomy/runs?${qs}`;
	}

	async function reloadRuns() {
		try {
			const rr = await fetchJSON(buildRunsPath());
			runs = rr.runs ?? [];
		} catch (e) {
			error = e.message;
		}
	}

	async function pauseResume() {
		try {
			const path = status?.paused ? '/api/autonomy/resume' : '/api/autonomy/pause';
			await fetchJSON(path, { method: 'POST' });
			await refresh();
		} catch (e) {
			error = e.message;
		}
	}

	async function triggerItem(item, bypassQuiet = false) {
		try {
			const qs = bypassQuiet ? '?bypass_quiet=true' : '';
			await fetchJSON(`/api/autonomy/trigger/${item.id}${qs}`, { method: 'POST' });
			await refresh();
		} catch (e) {
			error = e.message;
		}
	}

	async function toggleEnabled(item) {
		try {
			await fetchJSON(`/api/autonomy/items/${item.id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled: !item.enabled }),
			});
			await refresh();
		} catch (e) {
			error = e.message;
		}
	}

	async function deleteItem(item) {
		if (!confirm(`Delete "${item.name || item.kind}"?`)) return;
		try {
			await fetchJSON(`/api/autonomy/items/${item.id}`, { method: 'DELETE' });
			await refresh();
		} catch (e) {
			error = e.message;
		}
	}

	function openCreate() {
		formItem = null;
		formOpen = true;
	}

	function openEdit(item) {
		formItem = item;
		formOpen = true;
	}

	async function onFormSaved() {
		formOpen = false;
		formItem = null;
		await refresh();
	}

	// --- Live runs WebSocket ---
	function connectWS() {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		wsStatus = 'connecting';
		ws = new WebSocket(`${proto}//${location.host}/ws/autonomy/runs`);
		ws.onopen = () => {
			wsStatus = 'connected';
		};
		ws.onmessage = (e) => {
			let msg;
			try {
				msg = JSON.parse(e.data);
			} catch {
				return;
			}
			if (msg.type === 'run' && msg.run) {
				liveRuns = [msg.run, ...liveRuns.slice(0, 49)];
			}
		};
		ws.onerror = () => {
			wsStatus = 'error';
		};
		ws.onclose = () => {
			wsStatus = 'disconnected';
			ws = null;
			reconnectTimer = setTimeout(connectWS, 3000);
		};
	}

	onMount(() => {
		refresh();
		connectWS();
		// If arriving via a confirmation deep-link, scroll the pending card
		// into view once data lands.
		const params = new URLSearchParams(location.search);
		if (params.has('confirm')) {
			queueMicrotask(() => {
				const el = document.querySelector('.pending-list');
				if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
			});
		}
	});

	onDestroy(() => {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		try {
			ws?.close();
		} catch {}
	});

	function formatTime(iso) {
		if (!iso) return '—';
		try {
			return new Date(iso).toLocaleString();
		} catch {
			return iso;
		}
	}

	function formatDuration(ms) {
		if (ms == null) return '—';
		if (ms < 1000) return `${ms}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function runStatusClass(s) {
		if (!s) return '';
		if (s === 'ok' || s === 'notified' || s === 'completed') return 'ok';
		if (s === 'error' || s === 'failed') return 'err';
		if (s.startsWith('skipped')) return 'skip';
		if (s === 'scheduled' || s === 'in_flight') return 'pending';
		return '';
	}

	function describeTrigger(item) {
		if (item.schedule_cron) return `cron: ${item.schedule_cron}`;
		const ts = item.trigger_spec;
		if (!ts) return '—';
		if (ts.source === 'mqtt') return `mqtt: ${ts.match?.topic ?? '?'}`;
		if (ts.source === 'webhook') return `webhook: ${ts.match?.name ?? '?'}`;
		return JSON.stringify(ts);
	}
</script>

<div class="page">
	<div class="page-header">
		<h1 class="page-title">Autonomy</h1>
		<div class="header-actions">
			{#if status}
				<StatusBadge
					status={status.running && !status.paused ? 'healthy' : 'unhealthy'}
					label={status.paused ? 'Paused' : status.running ? 'Running' : 'Stopped'}
				/>
				<button class="btn ghost" onclick={pauseResume}>
					{status.paused ? 'Resume' : 'Pause'}
				</button>
			{/if}
			<button class="btn primary" onclick={openCreate}>New item</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">
			{error}
			<button class="dismiss" onclick={() => (error = '')}>×</button>
		</div>
	{/if}

	{#if loading}
		<div class="muted">Loading…</div>
	{:else}
		<!-- Section 1: Engine stats -->
		<div class="stats">
			<Card title="Runs (last hour)">
				<div class="big-number">{status?.runs_last_hour ?? 0}</div>
				<div class="muted">of {status?.rate_limit_per_hour ?? '—'} allowed</div>
			</Card>
			<Card title="Deferred (quiet hours)">
				<div class="big-number">{status?.deferred_runs_pending ?? 0}</div>
				<div class="muted">pending resume</div>
			</Card>
			<Card title="MQTT">
				<StatusBadge
					status={status?.mqtt_connected ? 'healthy' : 'unhealthy'}
					label={status?.mqtt_connected ? 'Connected' : 'Offline'}
				/>
				<div class="muted">{status?.subscribed_topics?.length ?? 0} topics subscribed</div>
			</Card>
			<Card title="Live feed">
				<StatusBadge
					status={wsStatus === 'connected' ? 'healthy' : 'unhealthy'}
					label={wsStatus}
				/>
				<div class="muted">WS /ws/autonomy/runs</div>
			</Card>
		</div>

		{#if awaiting.length > 0}
			<Card title="Pending confirmations ({awaiting.length})">
				<div class="pending-list">
					{#each awaiting as run (run.id)}
						<div class="pending-row">
							<div class="pending-meta">
								<div class="mono small">{run.id.slice(0, 8)}… · {formatTime(run.triggered_at)}</div>
								<div class="pending-summary">{run.summary || '(no summary)'}</div>
								{#if run.action_audit && run.action_audit.length > 0}
									<ul class="audit">
										{#each run.action_audit as step}
											<li class:pending={step.outcome === 'pending'} class:skipped={step.outcome !== 'pending'}>
												<span class="mono">{step.tool}</span>
												{#if step.rationale}<span class="muted"> — {step.rationale}</span>{/if}
												{#if step.outcome !== 'pending'}<span class="muted"> [{step.outcome}]</span>{/if}
											</li>
										{/each}
									</ul>
								{/if}
							</div>
							<div class="pending-actions">
								<button class="btn primary sm" onclick={() => confirmRun(run.id, true)}>Approve</button>
								<button class="btn danger sm" onclick={() => confirmRun(run.id, false)}>Deny</button>
							</div>
						</div>
					{/each}
				</div>
			</Card>
		{/if}

		<SpeakToDevice />

		<!-- Section 2: Agenda items -->
		<Card title="Agenda items">
			{#if items.length === 0}
				<p class="muted">No agenda items yet. Click "New item" to create one.</p>
			{:else}
				<div class="table-wrapper">
					<table class="table">
						<thead>
							<tr>
								<th>Name</th>
								<th>Kind</th>
								<th>Trigger</th>
								<th>Level</th>
								<th>Next</th>
								<th>Last</th>
								<th>Enabled</th>
								<th></th>
							</tr>
						</thead>
						<tbody>
							{#each items as item (item.id)}
								<tr class:disabled={!item.enabled}>
									<td>{item.name || '—'}</td>
									<td><span class="chip">{item.kind}</span></td>
									<td class="mono">{describeTrigger(item)}</td>
									<td>{item.autonomy_level}</td>
									<td class="mono">{formatTime(item.next_fire_at)}</td>
									<td class="mono">{formatTime(item.last_fired_at)}</td>
									<td>
										<button
											class="toggle"
											class:on={item.enabled}
											onclick={() => toggleEnabled(item)}
										>
											{item.enabled ? 'on' : 'off'}
										</button>
									</td>
									<td class="actions">
										<button class="btn ghost sm" onclick={() => triggerItem(item)}>Run</button>
										<button class="btn ghost sm" onclick={() => openEdit(item)}>Edit</button>
										<button class="btn danger sm" onclick={() => deleteItem(item)}>Delete</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</Card>

		<!-- Section 3: Live feed + Reactive sources -->
		<div class="two-col">
			<Card title="Live run feed">
				{#if liveRuns.length === 0}
					<p class="muted">Waiting for autonomy runs…</p>
				{:else}
					<div class="live-list">
						{#each liveRuns as run (run.id)}
							<div class="live-row">
								<span class="ts mono">{formatTime(run.triggered_at ?? run.started_at)}</span>
								<span class="chip">{run.kind}</span>
								<span class="src">{run.trigger_source ?? 'cron'}</span>
								<span class="status-pill {runStatusClass(run.status)}">{run.status}</span>
								<span class="summary">{run.summary ?? ''}</span>
							</div>
						{/each}
					</div>
				{/if}
			</Card>

			<Card title="Reactive sources">
				<div class="src-block">
					<div class="src-title">MQTT</div>
					<StatusBadge
						status={events?.mqtt?.connected ? 'healthy' : 'unhealthy'}
						label={events?.mqtt?.connected ? 'connected' : 'offline'}
					/>
					{#if events?.mqtt?.subscribed_topics?.length}
						<ul class="topic-list">
							{#each events.mqtt.subscribed_topics as topic}
								<li class="mono">{topic}</li>
							{/each}
						</ul>
					{:else}
						<p class="muted">No topics subscribed.</p>
					{/if}
				</div>

				<div class="src-block">
					<div class="src-title">Webhook</div>
					{#if events?.webhook?.items?.length}
						<ul class="topic-list">
							{#each events.webhook.items as w}
								<li class="mono">POST /api/autonomy/webhook/{w.name}</li>
							{/each}
						</ul>
					{:else}
						<p class="muted">No webhook items configured.</p>
					{/if}
				</div>

				<div class="src-block">
					<div class="src-title">Runs by source (24h)</div>
					<div class="source-counts">
						{#each Object.entries(events?.runs_last_24h_by_source ?? {}) as [src, count]}
							<div class="sc-row">
								<span>{src}</span>
								<span class="tier-count">{count}</span>
							</div>
						{:else}
							<p class="muted">No runs in last 24h.</p>
						{/each}
					</div>
				</div>
			</Card>
		</div>

		<!-- Section 4: Run history with filters -->
		<Card title="Run history">
			<div class="filters">
				<select class="input" bind:value={filterKind} onchange={reloadRuns}>
					<option value="">All kinds</option>
					<option value="briefing">briefing</option>
					<option value="anomaly_sweep">anomaly_sweep</option>
					<option value="memory_review">memory_review</option>
					<option value="reminder">reminder</option>
					<option value="warmup">warmup</option>
					<option value="watch">watch</option>
					<option value="routine">routine</option>
				</select>
				<select class="input" bind:value={filterStatus} onchange={reloadRuns}>
					<option value="">All statuses</option>
					<option value="ok">ok</option>
					<option value="notified">notified</option>
					<option value="error">error</option>
					<option value="skipped_quiet_hours">skipped_quiet_hours</option>
					<option value="skipped_rate_limit">skipped_rate_limit</option>
					<option value="skipped_trigger_mismatch">skipped_trigger_mismatch</option>
					<option value="scheduled">scheduled</option>
				</select>
				<select class="input" bind:value={filterSource} onchange={reloadRuns}>
					<option value="">All sources</option>
					<option value="cron">cron</option>
					<option value="mqtt">mqtt</option>
					<option value="webhook">webhook</option>
					<option value="manual">manual</option>
					<option value="deferred">deferred</option>
				</select>
			</div>

			{#if runs.length === 0}
				<p class="muted">No runs match.</p>
			{:else}
				<div class="table-wrapper">
					<table class="table">
						<thead>
							<tr>
								<th>When</th>
								<th>Kind</th>
								<th>Source</th>
								<th>Status</th>
								<th>Duration</th>
								<th>Summary</th>
							</tr>
						</thead>
						<tbody>
							{#each runs as run (run.id)}
								<tr>
									<td class="mono">{formatTime(run.triggered_at ?? run.started_at)}</td>
									<td><span class="chip">{run.kind}</span></td>
									<td class="mono">{run.trigger_source ?? 'cron'}</td>
									<td>
										<span class="status-pill {runStatusClass(run.status)}">{run.status}</span>
									</td>
									<td class="mono">{formatDuration(run.duration_ms)}</td>
									<td>{run.summary ?? run.error ?? ''}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</Card>
	{/if}
</div>

{#if formOpen}
	<AgendaForm
		item={formItem}
		timezone={status?.timezone ?? 'UTC'}
		onclose={() => {
			formOpen = false;
			formItem = null;
		}}
		onsaved={onFormSaved}
	/>
{/if}

<style>
	.page {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}

	.page-title {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.header-actions {
		display: flex;
		gap: 10px;
		align-items: center;
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 12px;
	}

	.big-number {
		font-size: 28px;
		font-weight: 700;
		color: #a78bfa;
		margin-bottom: 4px;
	}

	.muted {
		color: #6b7280;
		font-size: 13px;
	}

	.error-banner {
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid #7f1d1d;
		color: #f87171;
		padding: 12px 16px;
		border-radius: 8px;
		font-size: 14px;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.dismiss {
		background: none;
		border: none;
		color: #f87171;
		font-size: 18px;
		cursor: pointer;
	}

	.btn {
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 8px 14px;
		font-size: 13px;
		cursor: pointer;
		font-weight: 500;
		transition: all 0.15s;
	}

	.btn.sm {
		padding: 4px 10px;
		font-size: 12px;
	}

	.btn.primary {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border-color: transparent;
	}

	.btn.primary:hover {
		opacity: 0.9;
	}

	.btn.ghost {
		background: #252a3e;
		color: #c9cdd5;
	}

	.btn.ghost:hover {
		background: #2d3148;
	}

	.btn.danger {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		border-color: #7f1d1d;
	}

	.btn.danger:hover {
		background: rgba(239, 68, 68, 0.2);
	}

	.pending-list {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.pending-row {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 14px;
		padding: 10px 12px;
		border: 1px solid #3b3d55;
		border-radius: 8px;
		background: rgba(167, 139, 250, 0.05);
	}

	.pending-meta {
		display: flex;
		flex-direction: column;
		gap: 4px;
		flex: 1;
	}

	.pending-summary {
		font-size: 13px;
		color: #e1e4e8;
	}

	.audit {
		margin: 4px 0 0;
		padding-left: 18px;
		font-size: 12px;
	}

	.audit li.pending {
		color: #d1d5db;
	}

	.audit li.skipped {
		color: #6b7280;
	}

	.pending-actions {
		display: flex;
		gap: 6px;
	}

	.small {
		font-size: 11px;
		color: #6b7280;
	}

	.table-wrapper {
		overflow-x: auto;
	}

	.table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}

	.table th,
	.table td {
		text-align: left;
		padding: 8px 10px;
		border-bottom: 1px solid #1e2235;
		vertical-align: middle;
	}

	.table th {
		color: #9ca3af;
		font-weight: 500;
		font-size: 12px;
		text-transform: uppercase;
		letter-spacing: 0.02em;
	}

	.table tr.disabled {
		opacity: 0.5;
	}

	.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 12px;
		color: #c9cdd5;
	}

	.chip {
		display: inline-block;
		padding: 2px 8px;
		border-radius: 10px;
		background: #252a3e;
		color: #a78bfa;
		font-size: 11px;
		font-weight: 500;
	}

	.toggle {
		background: #252a3e;
		color: #6b7280;
		border: 1px solid #2d3148;
		border-radius: 10px;
		padding: 3px 10px;
		font-size: 11px;
		cursor: pointer;
		text-transform: uppercase;
	}

	.toggle.on {
		background: rgba(74, 222, 128, 0.1);
		color: #4ade80;
		border-color: rgba(74, 222, 128, 0.3);
	}

	.actions {
		display: flex;
		gap: 6px;
		justify-content: flex-end;
	}

	.two-col {
		display: grid;
		grid-template-columns: 2fr 1fr;
		gap: 16px;
	}

	@media (max-width: 900px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}

	.live-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
		max-height: 360px;
		overflow-y: auto;
	}

	.live-row {
		display: grid;
		grid-template-columns: 150px auto auto auto 1fr;
		gap: 8px;
		align-items: center;
		padding: 6px 8px;
		border-radius: 6px;
		font-size: 12px;
	}

	.live-row:nth-child(odd) {
		background: #1a1d2c;
	}

	.live-row .ts {
		color: #6b7280;
	}

	.live-row .src {
		color: #8b5cf6;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 11px;
	}

	.live-row .summary {
		color: #c9cdd5;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.status-pill {
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 11px;
		background: #252a3e;
		color: #9ca3af;
	}

	.status-pill.ok {
		background: rgba(74, 222, 128, 0.1);
		color: #4ade80;
	}

	.status-pill.err {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
	}

	.status-pill.skip {
		background: rgba(234, 179, 8, 0.1);
		color: #facc15;
	}

	.status-pill.pending {
		background: rgba(99, 102, 241, 0.1);
		color: #a78bfa;
	}

	.src-block {
		margin-bottom: 14px;
	}

	.src-block:last-child {
		margin-bottom: 0;
	}

	.src-title {
		font-size: 13px;
		font-weight: 600;
		color: #c9cdd5;
		margin-bottom: 6px;
	}

	.topic-list {
		list-style: none;
		padding: 0;
		margin: 6px 0 0;
		max-height: 120px;
		overflow-y: auto;
	}

	.topic-list li {
		padding: 3px 0;
		color: #8b5cf6;
		font-size: 11px;
	}

	.source-counts {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.sc-row {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
	}

	.tier-count {
		color: #a78bfa;
		font-weight: 600;
	}

	.filters {
		display: flex;
		gap: 8px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}

	.input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 6px;
		padding: 6px 10px;
		font-size: 13px;
	}
</style>
