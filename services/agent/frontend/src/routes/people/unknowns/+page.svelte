<script>
	import { onMount, onDestroy } from 'svelte';
	import {
		listDetections,
		listPeople,
		confirmDetection,
		rejectDetection,
		detectionSnapshotUrl,
		startRescanUnknowns,
		getFaceJob,
		bulkDeleteDetections,
	} from '$lib/api';
	import ImageLightbox from '$lib/components/ImageLightbox.svelte';

	let unknowns = $state([]);
	let people = $state([]);
	let loading = $state(true);
	let error = $state('');
	let statusMessage = $state('');
	// Per-row UI state: which mode is selected (existing|new) + form values.
	// Keyed by detection id. Reset whenever the queue refreshes.
	let rowState = $state({});
	let busyId = $state(null);
	let lightboxSrc = $state(null);

	let rescanJob = $state(null); // FaceJob | null
	let rescanPolling = $state(false);
	let rescanTimer = null;
	let bulkBusy = $state(false);
	let statusTimer = null;

	function flashStatus(msg) {
		statusMessage = msg;
		if (statusTimer) clearTimeout(statusTimer);
		statusTimer = setTimeout(() => (statusMessage = ''), 8000);
	}

	async function refresh() {
		loading = true;
		error = '';
		try {
			const [u, p] = await Promise.all([
				listDetections({ unknowns_only: true, limit: 100 }),
				listPeople(),
			]);
			unknowns = u;
			people = p;
			rowState = Object.fromEntries(
				u.map((d) => [d.id, { mode: 'existing', personId: '', name: '' }]),
			);
		} catch (e) {
			error = e.message ?? String(e);
		}
		loading = false;
	}

	async function confirmTo(id) {
		const s = rowState[id];
		if (!s) return;
		if (s.mode === 'existing' && !s.personId) {
			error = 'Pick a person from the dropdown.';
			return;
		}
		if (s.mode === 'new' && !s.name.trim()) {
			error = 'Type a new person name.';
			return;
		}
		busyId = id;
		error = '';
		try {
			const body =
				s.mode === 'existing' ? { person_id: s.personId } : { name: s.name.trim() };
			const r = await confirmDetection(id, body);
			if (!r.embedding_contributed) {
				error = `Confirmed as ${r.person_name}, but no face cleared the quality floor on the snapshot — embedding not added.`;
			}
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		busyId = null;
	}

	async function ignore(id) {
		busyId = id;
		error = '';
		try {
			await rejectDetection(id);
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		busyId = null;
	}

	function formatTime(iso) {
		if (!iso) return '';
		const d = new Date(iso);
		return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
	}

	function shortCamera(entity) {
		return entity.replace(/^camera\./, '').replace(/_fluent$/, '');
	}

	async function startRescan() {
		error = '';
		try {
			const r = await startRescanUnknowns();
			rescanPolling = true;
			rescanJob = null;
			pollRescan(r.job_id);
		} catch (e) {
			const msg = e.message ?? String(e);
			// 409 conflict — surface the in-flight job_id and start polling it.
			const match = msg.match(/"job_id"\s*:\s*"([0-9a-f-]+)"/i);
			if (match) {
				rescanPolling = true;
				rescanJob = null;
				pollRescan(match[1]);
				flashStatus('Rescan already in progress — attaching to it.');
			} else {
				error = msg;
			}
		}
	}

	function pollRescan(jobId) {
		const tick = async () => {
			try {
				const job = await getFaceJob(jobId);
				rescanJob = job;
				if (job.status !== 'running') {
					rescanPolling = false;
					if (rescanTimer) {
						clearInterval(rescanTimer);
						rescanTimer = null;
					}
					if (job.status === 'done') {
						const t = job.totals ?? {};
						flashStatus(
							`Rescanned ${t.examined ?? 0}: ${t.matched ?? 0} re-matched ` +
								`(${t.contributed ?? 0} new gallery embeddings), ` +
								`${t.no_match ?? 0} unmatched, ` +
								`${t.skipped_missing_snapshot ?? 0} skipped, ` +
								`${t.errors ?? 0} errors.`,
						);
					} else {
						error = `Rescan failed: ${(job.errors ?? []).map((x) => x.detail || x.reason).join('; ') || 'unknown error'}`;
					}
					await refresh();
				}
			} catch (e) {
				rescanPolling = false;
				if (rescanTimer) {
					clearInterval(rescanTimer);
					rescanTimer = null;
				}
				error = e.message ?? String(e);
			}
		};
		tick();
		rescanTimer = setInterval(tick, 1500);
	}

	async function clearRejected() {
		if (!confirm('Permanently delete all rejected unknowns? Files and rows are removed.')) {
			return;
		}
		bulkBusy = true;
		error = '';
		try {
			const r = await bulkDeleteDetections('rejected');
			flashStatus(`Deleted ${r.rows_deleted} rejected rows (${r.files_unlinked} files).`);
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		bulkBusy = false;
	}

	async function deleteAllUnknowns() {
		const proceed = confirm(
			"This deletes ALL unknown detections — including ones you haven't reviewed. Continue?",
		);
		if (!proceed) return;
		const typed = prompt('Type DELETE (uppercase) to confirm:');
		if (typed !== 'DELETE') {
			flashStatus('Aborted — confirmation text did not match.');
			return;
		}
		bulkBusy = true;
		error = '';
		try {
			const r = await bulkDeleteDetections('all_unknowns');
			flashStatus(`Deleted ${r.rows_deleted} unknown rows (${r.files_unlinked} files).`);
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		bulkBusy = false;
	}

	onMount(refresh);
	onDestroy(() => {
		if (rescanTimer) clearInterval(rescanTimer);
		if (statusTimer) clearTimeout(statusTimer);
	});
</script>

<div class="page">
	<header class="page-header">
		<div>
			<h1 class="page-title">Unknowns</h1>
			<p class="page-subtitle">Detections that didn't match a known person — assign or ignore.</p>
		</div>
		<nav class="sub-nav">
			<a href="/people">Known</a>
			<a href="/people/detections">Detections</a>
			<a href="/people/unknowns" class="active">Unknowns</a>
		</nav>
	</header>

	<div class="toolbar">
		<button
			class="action"
			onclick={startRescan}
			disabled={rescanPolling || bulkBusy}
			title="Re-run face matching on every unknown using the current embedding index"
		>
			{rescanPolling ? 'Rescanning...' : 'Rescan against current index'}
		</button>
		<button
			class="action"
			onclick={clearRejected}
			disabled={rescanPolling || bulkBusy}
			title="Delete every detection marked rejected"
		>
			Clear rejected
		</button>
		<button
			class="action danger"
			onclick={deleteAllUnknowns}
			disabled={rescanPolling || bulkBusy}
			title="Delete EVERY unknown detection, reviewed or not"
		>
			Delete ALL unknowns
		</button>
	</div>

	{#if rescanPolling && rescanJob}
		<div class="status-banner">
			Rescanning... examined {rescanJob.totals?.examined ?? 0},
			matched {rescanJob.totals?.matched ?? 0},
			contributed {rescanJob.totals?.contributed ?? 0}.
		</div>
	{/if}

	{#if statusMessage}
		<div class="status-banner">{statusMessage}</div>
	{/if}

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading}
		<p class="muted">Loading review queue...</p>
	{:else if unknowns.length === 0}
		<p class="muted">Nothing to review. All recent detections are either matched or rejected.</p>
	{:else}
		<div class="list">
			{#each unknowns as d (d.id)}
				{@const s = rowState[d.id] ?? { mode: 'existing', personId: '', name: '' }}
				<article class="row">
					<button
						type="button"
						class="thumb"
						onclick={() => (lightboxSrc = detectionSnapshotUrl(d.id))}
						title="Click to view full snapshot"
					>
						<img src={detectionSnapshotUrl(d.id)} alt="snapshot" loading="lazy" />
					</button>
					<div class="body">
						<div class="meta-line">
							<span class="cam">{shortCamera(d.camera)}</span>
							<span class="time">{formatTime(d.captured_at)}</span>
							{#if d.confidence != null}
								<span class="conf">closest match {d.confidence.toFixed(2)}</span>
							{/if}
							{#if d.quality_score != null}
								<span class="conf">quality {d.quality_score.toFixed(2)}</span>
							{/if}
						</div>

						<div class="actions">
							<div class="mode-tabs">
								<button
									class:active={s.mode === 'existing'}
									onclick={() => (rowState[d.id] = { ...s, mode: 'existing' })}
								>
									Assign to existing
								</button>
								<button
									class:active={s.mode === 'new'}
									onclick={() => (rowState[d.id] = { ...s, mode: 'new' })}
								>
									New person
								</button>
							</div>

							{#if s.mode === 'existing'}
								<select
									bind:value={s.personId}
									onchange={(e) => (rowState[d.id] = { ...s, personId: e.target.value })}
									disabled={busyId === d.id}
								>
									<option value="">Pick someone...</option>
									{#each people as p (p.id)}
										<option value={p.id}>{p.name}</option>
									{/each}
								</select>
							{:else}
								<input
									type="text"
									placeholder="New person name"
									bind:value={s.name}
									oninput={(e) => (rowState[d.id] = { ...s, name: e.target.value })}
									disabled={busyId === d.id}
								/>
							{/if}

							<button
								class="primary"
								onclick={() => confirmTo(d.id)}
								disabled={busyId === d.id}
							>
								{busyId === d.id ? 'Saving...' : 'This is them'}
							</button>
							<button class="ghost" onclick={() => ignore(d.id)} disabled={busyId === d.id}>
								Ignore
							</button>
						</div>
					</div>
				</article>
			{/each}
		</div>
	{/if}
</div>

{#if lightboxSrc}
	<ImageLightbox src={lightboxSrc} alt="full snapshot" onclose={() => (lightboxSrc = null)} />
{/if}

<style>
	.page {
		max-width: 1100px;
		margin: 0 auto;
	}
	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		gap: 16px;
		margin-bottom: 20px;
	}
	.page-title {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}
	.page-subtitle {
		color: #9ca3af;
		font-size: 13px;
		margin-top: 2px;
	}
	.sub-nav {
		display: flex;
		gap: 4px;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 4px;
	}
	.sub-nav a {
		padding: 6px 12px;
		border-radius: 6px;
		font-size: 13px;
		color: #9ca3af;
	}
	.sub-nav a:hover {
		color: #e1e4e8;
	}
	.sub-nav a.active {
		background: #252a3e;
		color: #a78bfa;
	}
	.error-banner {
		background: rgba(239, 68, 68, 0.08);
		border: 1px solid rgba(239, 68, 68, 0.4);
		color: #fca5a5;
		padding: 10px 14px;
		border-radius: 8px;
		margin-bottom: 16px;
		font-size: 13px;
	}
	.status-banner {
		background: rgba(34, 197, 94, 0.08);
		border: 1px solid rgba(34, 197, 94, 0.35);
		color: #86efac;
		padding: 10px 14px;
		border-radius: 8px;
		margin-bottom: 12px;
		font-size: 13px;
	}
	.toolbar {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		margin-bottom: 14px;
	}
	.toolbar .action {
		background: #2d3148;
		color: #e1e4e8;
		border: 0;
		padding: 8px 14px;
		border-radius: 6px;
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
	}
	.toolbar .action:hover:not(:disabled) {
		background: #3a3f56;
	}
	.toolbar .action.danger {
		background: rgba(239, 68, 68, 0.15);
		color: #f87171;
	}
	.toolbar .action.danger:hover:not(:disabled) {
		background: rgba(239, 68, 68, 0.25);
	}
	.toolbar .action:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.muted {
		color: #6b7280;
		font-size: 13px;
	}
	.list {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.row {
		display: flex;
		gap: 16px;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		padding: 14px;
	}
	.thumb {
		flex-shrink: 0;
		width: 180px;
		aspect-ratio: 4 / 3;
		background: #0f1117;
		border-radius: 6px;
		overflow: hidden;
		padding: 0;
		border: 0;
		cursor: zoom-in;
	}
	.thumb img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		display: block;
	}
	.body {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.meta-line {
		display: flex;
		gap: 14px;
		align-items: center;
		font-size: 12px;
		color: #9ca3af;
	}
	.cam {
		font-family: ui-monospace, SFMono-Regular, monospace;
		color: #a78bfa;
	}
	.time {
		color: #9ca3af;
	}
	.conf {
		color: #6b7280;
	}
	.actions {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	.mode-tabs {
		display: flex;
		gap: 0;
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 2px;
	}
	.mode-tabs button {
		background: transparent;
		color: #9ca3af;
		border: 0;
		padding: 6px 10px;
		font-size: 12px;
		border-radius: 4px;
		cursor: pointer;
	}
	.mode-tabs button.active {
		background: #2d3148;
		color: #e1e4e8;
	}
	select,
	input[type='text'] {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		padding: 7px 10px;
		border-radius: 6px;
		font-size: 13px;
		min-width: 200px;
	}
	button.primary {
		background: #6366f1;
		color: white;
		border: 0;
		padding: 7px 14px;
		border-radius: 6px;
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
	}
	button.primary:hover:not(:disabled) {
		background: #5558e6;
	}
	button.ghost {
		background: transparent;
		color: #9ca3af;
		border: 1px solid #2d3148;
		padding: 7px 14px;
		border-radius: 6px;
		font-size: 13px;
		cursor: pointer;
	}
	button.ghost:hover:not(:disabled) {
		background: #1e2235;
		color: #e1e4e8;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
