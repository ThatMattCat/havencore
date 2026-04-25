<script>
	import { onMount } from 'svelte';
	import {
		listDetections,
		listPeople,
		confirmDetection,
		rejectDetection,
		detectionSnapshotUrl,
	} from '$lib/api';

	let unknowns = $state([]);
	let people = $state([]);
	let loading = $state(true);
	let error = $state('');
	// Per-row UI state: which mode is selected (existing|new) + form values.
	// Keyed by detection id. Reset whenever the queue refreshes.
	let rowState = $state({});
	let busyId = $state(null);

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

	onMount(refresh);
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
					<div class="thumb">
						<img src={detectionSnapshotUrl(d.id)} alt="snapshot" loading="lazy" />
					</div>
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
	}
	.thumb img {
		width: 100%;
		height: 100%;
		object-fit: cover;
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
