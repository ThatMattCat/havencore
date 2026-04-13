<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	let stats = $state(null);
	let l4 = $state([]);
	let proposals = $state([]);
	let runs = $state([]);
	let l3Entries = $state([]);
	let l3Offset = $state(0);
	let l3HasMore = $state(false);
	let sourcesModal = $state(null);
	let loading = $state(true);
	let error = $state('');
	let triggering = $state(false);
	let lastRunTime = $state(null);
	let newL4Text = $state('');
	let newL4Importance = $state(5);

	let searchQ = $state('');
	let searchTiers = $state({ L2: true, L3: true, L4: true });
	let searchResults = $state(null);
	let searchLoading = $state(false);
	let searchError = $state('');

	const PAGE_SIZE = 50;

	async function fetchJSON(path, init) {
		const res = await fetch(path, init);
		if (!res.ok) throw new Error(`${res.status}: ${await res.text().catch(() => res.statusText)}`);
		return res.json();
	}

	async function refreshL3() {
		const r = await fetchJSON(`/api/memory/l3?limit=${PAGE_SIZE}&offset=${l3Offset}`);
		l3Entries = r.entries ?? [];
		l3HasMore = !!r.has_more;
	}

	async function refresh() {
		loading = true;
		error = '';
		try {
			const [s, l, p, r] = await Promise.all([
				fetchJSON('/api/memory/stats'),
				fetchJSON('/api/memory/l4'),
				fetchJSON('/api/memory/l4/proposals'),
				fetchJSON('/api/memory/runs?limit=20'),
			]);
			stats = s;
			l4 = l.entries ?? [];
			proposals = p.proposals ?? [];
			runs = r.runs ?? [];
			lastRunTime = runs.length ? runs[0].triggered_at : null;
			await refreshL3();
		} catch (e) {
			error = e?.message ?? String(e);
		} finally {
			loading = false;
		}
	}

	async function showSources(id) {
		try {
			const r = await fetchJSON(`/api/memory/l3/${id}/sources`);
			sourcesModal = r.sources ?? [];
		} catch (e) {
			error = e?.message ?? String(e);
		}
	}

	async function deleteL3(id) {
		if (!confirm('Delete this L3 entry? Source L2 entries will remain untouched.')) return;
		await fetch(`/api/memory/l3/${id}`, { method: 'DELETE' });
		await refreshL3();
	}

	async function approve(id) {
		await fetch(`/api/memory/l4/proposals/${id}/approve`, { method: 'POST' });
		await refresh();
	}

	async function reject(id) {
		await fetch(`/api/memory/l4/proposals/${id}/reject`, { method: 'POST' });
		await refresh();
	}

	async function deleteL4(id) {
		if (!confirm('Remove this L4 entry? It will be demoted to L3, not deleted.')) return;
		await fetch(`/api/memory/l4/${id}`, { method: 'DELETE' });
		await refresh();
	}

	async function addL4() {
		if (!newL4Text.trim()) return;
		await fetch('/api/memory/l4', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ text: newL4Text, importance: newL4Importance, tags: [] }),
		});
		newL4Text = '';
		await refresh();
	}

	async function runNow() {
		triggering = true;
		try {
			await fetch('/api/memory/runs/trigger', { method: 'POST' });
			for (let i = 0; i < 60; i++) {
				await new Promise((r) => setTimeout(r, 3000));
				await refresh();
				if (runs[0]?.triggered_at !== lastRunTime) break;
			}
		} finally {
			triggering = false;
		}
	}

	async function runSearch() {
		const tiers = Object.entries(searchTiers).filter(([, v]) => v).map(([k]) => k);
		if (!searchQ.trim() || tiers.length === 0) return;
		searchLoading = true;
		searchError = '';
		try {
			const r = await fetchJSON('/api/memory/search', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ q: searchQ, tiers, limit: 25 }),
			});
			searchResults = r.results ?? [];
		} catch (e) {
			searchError = e?.message ?? String(e);
			searchResults = null;
		} finally {
			searchLoading = false;
		}
	}

	function clearSearch() {
		searchQ = '';
		searchResults = null;
		searchError = '';
	}

	function onSearchKey(e) {
		if (e.key === 'Enter') runSearch();
	}

	function prevPage() {
		l3Offset = Math.max(0, l3Offset - PAGE_SIZE);
		refreshL3();
	}

	function nextPage() {
		l3Offset += PAGE_SIZE;
		refreshL3();
	}

	function fmtDate(iso) {
		if (!iso) return '';
		return iso.slice(0, 10);
	}

	function fmtTime(iso) {
		if (!iso) return '';
		return new Date(iso).toLocaleString();
	}

	let tokensWarn = $derived(stats && stats.l4_est_tokens > 1500);

	onMount(refresh);
</script>

<svelte:head><title>Memory — Selene</title></svelte:head>

<div class="page">
	<h1 class="page-title">Memory</h1>
	<p class="subtitle">Tiered memory: L2 episodic logs, L3 consolidated summaries, L4 persistent context.</p>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<div class="stats">
		<Card title="L2 episodic">
			<div class="stat">{stats?.l2_count ?? '–'}</div>
		</Card>
		<Card title="L3 consolidated">
			<div class="stat">{stats?.l3_count ?? '–'}</div>
		</Card>
		<Card title="L4 persistent">
			<div class="stat">{stats?.l4_count ?? '–'}</div>
		</Card>
		<Card title="Pending proposals">
			<div class="stat" class:warn={proposals.length > 0}>{stats?.pending_proposals ?? 0}</div>
		</Card>
		<Card title="L4 ~tokens">
			<div class="stat" class:danger={tokensWarn}>{stats?.l4_est_tokens ?? 0}</div>
		</Card>
	</div>

	<div class="action-bar">
		<button class="btn primary" onclick={runNow} disabled={triggering}>
			{triggering ? 'Running…' : 'Run consolidation now'}
		</button>
		<span class="muted">Last run: {lastRunTime ? fmtTime(lastRunTime) : 'never'}</span>
	</div>

	<Card title="Semantic search">
		<div class="search-form">
			<input
				type="text"
				class="input"
				placeholder="Search memories by meaning (e.g. 'plex media setup', 'user's coffee preference')"
				bind:value={searchQ}
				onkeydown={onSearchKey}
			/>
			<button class="btn primary" onclick={runSearch} disabled={searchLoading || !searchQ.trim()}>
				{searchLoading ? 'Searching…' : 'Search'}
			</button>
			{#if searchResults !== null || searchError}
				<button class="btn ghost" onclick={clearSearch}>Clear</button>
			{/if}
		</div>

		<div class="search-tiers">
			<label class="tier-toggle">
				<input type="checkbox" bind:checked={searchTiers.L2} />
				<span>L2 episodic</span>
			</label>
			<label class="tier-toggle">
				<input type="checkbox" bind:checked={searchTiers.L3} />
				<span>L3 consolidated</span>
			</label>
			<label class="tier-toggle">
				<input type="checkbox" bind:checked={searchTiers.L4} />
				<span>L4 persistent</span>
			</label>
		</div>

		{#if searchError}
			<div class="error-banner search-error">{searchError}</div>
		{/if}

		{#if searchResults !== null}
			{#if searchResults.length === 0}
				<p class="muted">No matches.</p>
			{:else}
				<div class="search-results">
					{#each searchResults as r}
						<div class="result" class:r-l4={r.tier === 'L4'} class:r-l3={r.tier === 'L3'}>
							<div class="result-header">
								<span class="tier-pill tier-{r.tier?.toLowerCase()}">{r.tier}</span>
								<span class="result-score" title="cosine similarity">{r.score?.toFixed(3)}</span>
								<span class="muted result-date">{fmtDate(r.timestamp)}</span>
								{#if r.tier === 'L3' && r.source_ids?.length}
									<button class="link result-sources" onclick={() => showSources(r.id)}>
										{r.source_ids.length} sources
									</button>
								{/if}
							</div>
							<div class="result-text">{r.text}</div>
						</div>
					{/each}
				</div>
			{/if}
		{/if}
	</Card>

	<Card title="L4 — Persistent context">
		<div class="l4-form">
			<textarea
				class="input"
				rows="2"
				placeholder="Add an L4 entry (injected into every system prompt)"
				bind:value={newL4Text}
			></textarea>
			<div class="l4-form-controls">
				<label class="field">
					<span class="field-label">Importance</span>
					<input
						type="number"
						min="1"
						max="5"
						class="input narrow"
						bind:value={newL4Importance}
					/>
				</label>
				<button class="btn success" onclick={addL4}>Add</button>
			</div>
		</div>

		{#if l4.length === 0}
			<p class="muted">No L4 entries yet.</p>
		{:else}
			<div class="table-wrapper">
				<table class="table">
					<thead>
						<tr>
							<th>Text</th>
							<th class="num">Importance</th>
							<th class="num">Age</th>
							<th></th>
						</tr>
					</thead>
					<tbody>
						{#each l4 as e}
							<tr>
								<td class="wrap">{e.text}</td>
								<td class="num">{e.importance}</td>
								<td class="num">{fmtDate(e.timestamp)}</td>
								<td class="actions">
									<button class="link danger" onclick={() => deleteL4(e.id)}>Remove</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</Card>

	<Card title="Pending proposals ({proposals.length})">
		{#if proposals.length === 0}
			<p class="muted">No pending proposals.</p>
		{:else}
			<div class="proposal-list">
				{#each proposals as p}
					<div class="proposal">
						<div class="proposal-text">{p.text}</div>
						{#if p.proposal_rationale}
							<div class="proposal-rationale">{p.proposal_rationale}</div>
						{/if}
						<div class="proposal-meta">
							<StatusBadge status="unknown" label="sources: {p.source_ids.length}" />
							<StatusBadge status="unknown" label="importance: {p.importance}" />
						</div>
						<div class="proposal-actions">
							<button class="btn success" onclick={() => approve(p.id)}>Approve</button>
							<button class="btn ghost" onclick={() => reject(p.id)}>Reject</button>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</Card>

	<Card title="L3 — Consolidated memories">
		{#if l3Entries.length === 0}
			<p class="muted">No L3 entries yet.</p>
		{:else}
			<div class="table-wrapper">
				<table class="table">
					<thead>
						<tr>
							<th>Text</th>
							<th class="num">Importance (eff)</th>
							<th class="num">Sources</th>
							<th class="num">Age</th>
							<th></th>
						</tr>
					</thead>
					<tbody>
						{#each l3Entries as e}
							<tr>
								<td class="wrap">{e.text}</td>
								<td class="num">{e.importance_effective?.toFixed(2)}</td>
								<td class="num">
									<button class="link" onclick={() => showSources(e.id)}>
										{e.source_ids.length}
									</button>
								</td>
								<td class="num">{fmtDate(e.timestamp)}</td>
								<td class="actions">
									<button class="link danger" onclick={() => deleteL3(e.id)}>Delete</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<div class="pager">
				<button class="btn ghost" disabled={l3Offset === 0} onclick={prevPage}>Prev</button>
				<span class="muted">Showing {l3Offset + 1}–{l3Offset + l3Entries.length}</span>
				<button class="btn ghost" disabled={!l3HasMore} onclick={nextPage}>Next</button>
			</div>
		{/if}
	</Card>

	<Card title="Consolidation runs">
		{#if runs.length === 0}
			<p class="muted">No runs yet.</p>
		{:else}
			<div class="table-wrapper">
				<table class="table">
					<thead>
						<tr>
							<th>When</th>
							<th>Status</th>
							<th>Summary</th>
							<th class="num">Stats</th>
						</tr>
					</thead>
					<tbody>
						{#each runs as r}
							<tr class:row-error={r.status === 'error'}>
								<td>{fmtTime(r.triggered_at)}</td>
								<td>
									<StatusBadge
										status={r.status === 'error' ? 'unhealthy' : r.status === 'success' ? 'healthy' : 'unknown'}
										label={r.status}
									/>
								</td>
								<td class="wrap">{r.summary}</td>
								<td class="num metrics-cell">
									<span class="chip">L3+{r.metrics?.l3_created ?? 0}</span>
									<span class="chip">L4?{r.metrics?.l4_proposed ?? 0}</span>
									<span class="chip">pruned {r.metrics?.l2_pruned ?? 0}</span>
									<span class="chip">{r.metrics?.total_ms ?? 0}ms</span>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</Card>
</div>

{#if sourcesModal}
	<div
		class="modal-backdrop"
		role="presentation"
		onclick={() => (sourcesModal = null)}
	>
		<div
			class="modal"
			role="dialog"
			aria-modal="true"
			aria-label="Source L2 entries"
			onclick={(e) => e.stopPropagation()}
		>
			<div class="modal-header">
				<h3>Source L2 entries</h3>
				<button class="link" onclick={() => (sourcesModal = null)}>Close</button>
			</div>
			<div class="modal-body">
				{#each sourcesModal as src}
					<div class="source-entry">
						<div class="source-text">{src.text}</div>
						<div class="source-meta">{fmtTime(src.timestamp)}</div>
					</div>
				{/each}
				{#if sourcesModal.length === 0}
					<p class="muted">No source entries.</p>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.page {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.page-title {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.subtitle {
		color: #6b7280;
		font-size: 14px;
		margin-top: -8px;
	}

	.error-banner {
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid #7f1d1d;
		color: #f87171;
		padding: 12px 16px;
		border-radius: 8px;
		font-size: 14px;
	}

	.muted {
		color: #6b7280;
		font-size: 13px;
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 12px;
	}

	.stat {
		font-size: 28px;
		font-weight: 700;
		color: #a78bfa;
	}

	.stat.warn {
		color: #fbbf24;
	}

	.stat.danger {
		color: #f87171;
	}

	.action-bar {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.btn {
		padding: 8px 14px;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 500;
		border: 1px solid transparent;
		cursor: pointer;
		transition: all 0.15s;
		font-family: inherit;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn.primary {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
	}

	.btn.primary:hover:not(:disabled) {
		filter: brightness(1.1);
	}

	.btn.success {
		background: rgba(34, 197, 94, 0.15);
		border-color: rgba(34, 197, 94, 0.3);
		color: #4ade80;
	}

	.btn.success:hover:not(:disabled) {
		background: rgba(34, 197, 94, 0.25);
	}

	.btn.ghost {
		background: #1e2235;
		border-color: #2d3148;
		color: #c9cdd5;
	}

	.btn.ghost:hover:not(:disabled) {
		background: #252a3e;
	}

	.l4-form {
		display: flex;
		gap: 12px;
		align-items: flex-start;
		margin-bottom: 16px;
	}

	.input {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		color: #e1e4e8;
		padding: 8px 10px;
		font-size: 13px;
		font-family: inherit;
		flex: 1;
		resize: vertical;
	}

	.input:focus {
		outline: none;
		border-color: #6366f1;
	}

	.input.narrow {
		width: 70px;
		flex: none;
	}

	.l4-form-controls {
		display: flex;
		flex-direction: column;
		gap: 8px;
		flex-shrink: 0;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.field-label {
		font-size: 11px;
		color: #6b7280;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.table-wrapper {
		overflow-x: auto;
	}

	.table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}

	.table th {
		text-align: left;
		color: #6b7280;
		font-weight: 500;
		padding: 8px;
		border-bottom: 1px solid #2d3148;
	}

	.table th.num,
	.table td.num {
		text-align: right;
		white-space: nowrap;
	}

	.table td {
		padding: 8px;
		border-bottom: 1px solid #1e2235;
		color: #c9cdd5;
		vertical-align: top;
	}

	.table tr:last-child td {
		border-bottom: none;
	}

	.table td.wrap {
		white-space: pre-wrap;
		word-break: break-word;
	}

	.table td.actions {
		text-align: right;
		white-space: nowrap;
	}

	.row-error td {
		color: #f87171;
	}

	.link {
		background: none;
		border: none;
		color: #a78bfa;
		cursor: pointer;
		padding: 0;
		font-size: inherit;
		font-family: inherit;
	}

	.link:hover {
		text-decoration: underline;
	}

	.link.danger {
		color: #f87171;
	}

	.proposal-list {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.proposal {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 12px 14px;
	}

	.proposal-text {
		font-size: 14px;
		color: #e1e4e8;
		font-weight: 500;
	}

	.proposal-rationale {
		font-size: 12px;
		color: #9ca3af;
		font-style: italic;
		margin-top: 4px;
	}

	.proposal-meta {
		display: flex;
		gap: 6px;
		margin-top: 8px;
		flex-wrap: wrap;
	}

	.proposal-actions {
		display: flex;
		gap: 8px;
		margin-top: 10px;
	}

	.pager {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: 12px;
	}

	.metrics-cell {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
		justify-content: flex-end;
	}

	.chip {
		background: #252a3e;
		color: #c9cdd5;
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 11px;
	}

	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.6);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 50;
		padding: 24px;
	}

	.modal {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		max-width: 720px;
		width: 100%;
		max-height: 80vh;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 14px 18px;
		border-bottom: 1px solid #2d3148;
	}

	.modal-header h3 {
		font-size: 14px;
		font-weight: 600;
		color: #c9cdd5;
	}

	.modal-body {
		padding: 8px 18px 18px;
		overflow-y: auto;
	}

	.source-entry {
		padding: 10px 0;
		border-bottom: 1px solid #1e2235;
	}

	.source-entry:last-child {
		border-bottom: none;
	}

	.source-text {
		font-size: 13px;
		color: #c9cdd5;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.source-meta {
		font-size: 11px;
		color: #6b7280;
		margin-top: 4px;
	}

	.search-form {
		display: flex;
		gap: 8px;
		margin-bottom: 10px;
	}

	.search-form .input {
		flex: 1;
	}

	.search-tiers {
		display: flex;
		gap: 14px;
		margin-bottom: 14px;
		flex-wrap: wrap;
	}

	.tier-toggle {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: #c9cdd5;
		cursor: pointer;
		user-select: none;
	}

	.tier-toggle input[type='checkbox'] {
		accent-color: #8b5cf6;
		cursor: pointer;
	}

	.search-error {
		margin-bottom: 10px;
	}

	.search-results {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.result {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-left: 3px solid #2d3148;
		border-radius: 8px;
		padding: 10px 12px;
	}

	.result.r-l3 {
		border-left-color: #6366f1;
	}

	.result.r-l4 {
		border-left-color: #a78bfa;
	}

	.result-header {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 6px;
		flex-wrap: wrap;
	}

	.tier-pill {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.04em;
		padding: 2px 6px;
		border-radius: 4px;
		background: #252a3e;
		color: #c9cdd5;
	}

	.tier-pill.tier-l3 {
		background: rgba(99, 102, 241, 0.15);
		color: #818cf8;
	}

	.tier-pill.tier-l4 {
		background: rgba(167, 139, 250, 0.15);
		color: #a78bfa;
	}

	.result-score {
		font-size: 11px;
		color: #6b7280;
		font-variant-numeric: tabular-nums;
		font-family: monospace;
	}

	.result-date {
		font-size: 11px;
	}

	.result-sources {
		font-size: 11px;
		margin-left: auto;
	}

	.result-text {
		font-size: 13px;
		color: #e1e4e8;
		white-space: pre-wrap;
		word-break: break-word;
		line-height: 1.5;
	}
</style>
