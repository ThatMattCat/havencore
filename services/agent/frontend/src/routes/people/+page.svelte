<script>
	import { onMount } from 'svelte';
	import {
		listPeople,
		listDetections,
		createPerson,
		faceImageUrl,
	} from '$lib/api';

	let people = $state([]);
	let lastSeen = $state({}); // person_id → ISO timestamp of most recent detection
	let primaryImageId = $state({}); // person_id → face_image_id of primary
	let loading = $state(true);
	let error = $state('');
	let creating = $state(false);
	let newName = $state('');

	async function refresh() {
		loading = true;
		error = '';
		try {
			const list = await listPeople();
			people = list;
			// Look up the primary image and last-seen in parallel. Both are
			// O(people) round-trips and the list is small (single-digits in v1).
			const detail = await Promise.all(
				people.map((p) => fetch(`/api/face/people/${p.id}`).then((r) => r.json())),
			);
			const primaries = {};
			for (const d of detail) {
				const prim = (d.images ?? []).find((i) => i.is_primary);
				if (prim) primaries[d.id] = prim.id;
			}
			primaryImageId = primaries;

			const seenEntries = await Promise.all(
				people.map(async (p) => {
					const rows = await listDetections({ person_id: p.id, limit: 1 });
					return [p.id, rows[0]?.captured_at ?? null];
				}),
			);
			lastSeen = Object.fromEntries(seenEntries);
		} catch (e) {
			error = e.message ?? String(e);
		}
		loading = false;
	}

	async function submitCreate(event) {
		event.preventDefault();
		const name = newName.trim();
		if (!name) return;
		creating = true;
		error = '';
		try {
			await createPerson({ name });
			newName = '';
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		creating = false;
	}

	function formatRelative(iso) {
		if (!iso) return 'never';
		const ms = Date.now() - new Date(iso).getTime();
		const s = Math.floor(ms / 1000);
		if (s < 60) return `${s}s ago`;
		const m = Math.floor(s / 60);
		if (m < 60) return `${m}m ago`;
		const h = Math.floor(m / 60);
		if (h < 24) return `${h}h ago`;
		const d = Math.floor(h / 24);
		return `${d}d ago`;
	}

	function accessClass(level) {
		return `chip chip-${level}`;
	}

	onMount(refresh);
</script>

<div class="page">
	<header class="page-header">
		<div>
			<h1 class="page-title">People</h1>
			<p class="page-subtitle">Known faces and their recent activity</p>
		</div>
		<nav class="sub-nav">
			<a href="/people" class="active">Known</a>
			<a href="/people/detections">Detections</a>
			<a href="/people/unknowns">Unknowns</a>
		</nav>
	</header>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<form class="create-row" onsubmit={submitCreate}>
		<input
			type="text"
			placeholder="Add new person by name..."
			bind:value={newName}
			disabled={creating}
		/>
		<button type="submit" disabled={creating || !newName.trim()}>
			{creating ? 'Creating...' : 'Create'}
		</button>
	</form>

	{#if loading}
		<p class="muted">Loading people...</p>
	{:else if people.length === 0}
		<p class="muted">
			No people enrolled yet. Create one above, then upload reference photos
			from their detail page.
		</p>
	{:else}
		<div class="grid">
			{#each people as p (p.id)}
				<a href={`/people/${p.id}`} class="card">
					<div class="avatar">
						{#if primaryImageId[p.id]}
							<img src={faceImageUrl(primaryImageId[p.id])} alt={p.name} />
						{:else}
							<span class="avatar-initial">{p.name[0]?.toUpperCase() ?? '?'}</span>
						{/if}
					</div>
					<div class="card-body">
						<div class="row">
							<span class="name">{p.name}</span>
							<span class={accessClass(p.access_level)}>{p.access_level}</span>
						</div>
						<div class="meta">
							<span>{p.image_count} {p.image_count === 1 ? 'image' : 'images'}</span>
							<span>· last seen {formatRelative(lastSeen[p.id])}</span>
						</div>
					</div>
				</a>
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
		margin-bottom: 24px;
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
	.create-row {
		display: flex;
		gap: 8px;
		margin-bottom: 20px;
	}
	.create-row input {
		flex: 1;
		background: #161822;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 14px;
	}
	.create-row input:focus {
		outline: none;
		border-color: #6366f1;
	}
	.create-row button {
		background: #6366f1;
		color: white;
		border: 0;
		padding: 10px 18px;
		border-radius: 8px;
		font-size: 14px;
		font-weight: 500;
		cursor: pointer;
	}
	.create-row button:disabled {
		background: #2d3148;
		color: #6b7280;
		cursor: not-allowed;
	}
	.muted {
		color: #6b7280;
		font-size: 14px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
		gap: 16px;
	}
	.card {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		transition: border-color 0.15s, transform 0.15s;
	}
	.card:hover {
		border-color: #4b5168;
		transform: translateY(-2px);
	}
	.avatar {
		aspect-ratio: 16 / 10;
		background: #0f1117;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
	}
	.avatar img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}
	.avatar-initial {
		font-size: 56px;
		font-weight: 700;
		color: #4b5168;
	}
	.card-body {
		padding: 12px 14px;
	}
	.row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 6px;
	}
	.name {
		font-size: 15px;
		font-weight: 600;
		color: #e1e4e8;
	}
	.meta {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
		font-size: 12px;
		color: #9ca3af;
	}
	.chip {
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 10px;
		font-weight: 500;
		text-transform: lowercase;
	}
	.chip-unknown {
		background: rgba(156, 163, 175, 0.15);
		color: #9ca3af;
	}
	.chip-resident {
		background: rgba(34, 197, 94, 0.15);
		color: #4ade80;
	}
	.chip-guest {
		background: rgba(251, 191, 36, 0.15);
		color: #fbbf24;
	}
	.chip-blocked {
		background: rgba(239, 68, 68, 0.15);
		color: #f87171;
	}
</style>
