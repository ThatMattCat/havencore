<script>
	import { onMount } from 'svelte';
	import { listDetections, detectionSnapshotUrl } from '$lib/api';
	import ImageLightbox from '$lib/components/ImageLightbox.svelte';

	let detections = $state([]);
	let loading = $state(true);
	let error = $state('');
	let limit = $state(50);
	let cameraFilter = $state('');
	let stateFilter = $state(''); // '' | 'auto' | 'confirmed' | 'rejected' | 'pending'
	let lightboxSrc = $state(null);

	async function refresh() {
		loading = true;
		error = '';
		try {
			detections = await listDetections({
				limit,
				camera: cameraFilter || undefined,
				review_state: stateFilter || undefined,
			});
		} catch (e) {
			error = e.message ?? String(e);
		}
		loading = false;
	}

	function formatTime(iso) {
		if (!iso) return '';
		const d = new Date(iso);
		return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
	}

	function shortCamera(entity) {
		// camera.front_duo_3_clear → front_duo_3
		return entity.replace(/^camera\./, '').replace(/_(clear|fluent|main|sub)$/, '');
	}

	function formatDemographic(d) {
		// Informational only — InsightFace estimates from the genderage
		// submodel. Hidden when no face was detected (no_face events).
		if (d.age == null && !d.sex) return null;
		const parts = [];
		if (d.sex) parts.push(d.sex);
		if (d.age != null) parts.push(`~${d.age}`);
		return parts.join(' ');
	}

	function stateClass(s) {
		return `state state-${s}`;
	}

	onMount(refresh);
</script>

<div class="page">
	<header class="page-header">
		<div>
			<h1 class="page-title">Detections</h1>
			<p class="page-subtitle">Recent face-recognition events across all cameras</p>
		</div>
		<nav class="sub-nav">
			<a href="/people">Known</a>
			<a href="/people/detections" class="active">Detections</a>
			<a href="/people/unknowns">Unknowns</a>
		</nav>
	</header>

	<div class="filter-row">
		<input
			type="text"
			placeholder="Filter by camera entity (e.g. camera.front_duo_3_fluent)"
			bind:value={cameraFilter}
			onchange={refresh}
		/>
		<select bind:value={stateFilter} onchange={refresh}>
			<option value="">All states</option>
			<option value="auto">Auto (no review)</option>
			<option value="confirmed">Confirmed</option>
			<option value="rejected">Rejected</option>
			<option value="pending">Pending</option>
		</select>
		<select bind:value={limit} onchange={refresh}>
			<option value={20}>20 rows</option>
			<option value={50}>50 rows</option>
			<option value={100}>100 rows</option>
			<option value={200}>200 rows</option>
		</select>
		<button onclick={refresh} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</button>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading}
		<p class="muted">Loading...</p>
	{:else if detections.length === 0}
		<p class="muted">No detections match these filters.</p>
	{:else}
		<div class="list">
			{#each detections as d (d.id)}
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
						<div class="line top">
							{#if d.person_id}
								<a href={`/people/${d.person_id}`} class="name">{d.person_name}</a>
							{:else}
								<span class="name unknown">unknown</span>
							{/if}
							<span class={stateClass(d.review_state)}>{d.review_state}</span>
						</div>
						<div class="line">
							<span class="cam">{shortCamera(d.camera)}</span>
							<span class="muted">·</span>
							<span class="time">{formatTime(d.captured_at)}</span>
						</div>
						<div class="line meta">
							{#if d.confidence != null}
								<span>conf {d.confidence.toFixed(2)}</span>
							{/if}
							{#if d.quality_score != null}
								<span>quality {d.quality_score.toFixed(2)}</span>
							{/if}
							{#if formatDemographic(d)}
								<span class="demo" title="InsightFace estimate — display only">{formatDemographic(d)}</span>
							{/if}
							{#if d.embedding_contributed}
								<span class="contrib">embedding contributed</span>
							{/if}
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
	.filter-row {
		display: flex;
		gap: 8px;
		align-items: center;
		margin-bottom: 16px;
		flex-wrap: wrap;
	}
	.filter-row input,
	.filter-row select {
		background: #161822;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		padding: 8px 10px;
		border-radius: 6px;
		font-size: 13px;
	}
	.filter-row input {
		flex: 1;
		min-width: 240px;
	}
	.filter-row button {
		background: #2d3148;
		color: #e1e4e8;
		border: 0;
		padding: 8px 14px;
		border-radius: 6px;
		font-size: 13px;
		cursor: pointer;
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
		gap: 8px;
	}
	.row {
		display: flex;
		gap: 14px;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 10px;
		padding: 10px;
	}
	.thumb {
		flex-shrink: 0;
		width: 120px;
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
		justify-content: center;
		gap: 4px;
	}
	.line {
		display: flex;
		gap: 10px;
		align-items: center;
		font-size: 13px;
		color: #c9cdd5;
	}
	.line.top {
		gap: 12px;
	}
	.name {
		font-weight: 600;
		color: #e1e4e8;
	}
	.name.unknown {
		color: #fbbf24;
		font-style: italic;
	}
	.cam {
		font-family: ui-monospace, SFMono-Regular, monospace;
		font-size: 12px;
		color: #a78bfa;
	}
	.time {
		color: #9ca3af;
		font-size: 12px;
	}
	.meta {
		font-size: 11px;
		color: #6b7280;
		gap: 14px;
	}
	.contrib {
		color: #4ade80;
	}
	.demo {
		color: #94a3b8;
		font-variant-numeric: tabular-nums;
	}
	.state {
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 10px;
		font-weight: 500;
		text-transform: lowercase;
	}
	.state-auto {
		background: rgba(99, 102, 241, 0.15);
		color: #a5b4fc;
	}
	.state-confirmed {
		background: rgba(34, 197, 94, 0.15);
		color: #4ade80;
	}
	.state-rejected {
		background: rgba(239, 68, 68, 0.15);
		color: #f87171;
	}
	.state-pending {
		background: rgba(251, 191, 36, 0.15);
		color: #fbbf24;
	}
</style>
