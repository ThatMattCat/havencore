<script>
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import {
		getPerson,
		updatePerson,
		deletePerson,
		enrollImage,
		setPrimaryImage,
		deleteFaceImage,
		faceImageUrl,
		ACCESS_LEVELS,
	} from '$lib/api';

	let personId = $derived(page.params.id);
	let person = $state(null);
	let loading = $state(true);
	let error = $state('');
	let busy = $state(false);

	let uploadFile = $state(null);
	let uploadPrimary = $state(false);
	let uploading = $state(false);

	let accessDraft = $state('unknown');
	let notesDraft = $state('');
	let metaSaving = $state(false);

	async function refresh() {
		loading = true;
		error = '';
		try {
			person = await getPerson(personId);
			accessDraft = person.access_level;
			notesDraft = person.notes ?? '';
		} catch (e) {
			error = e.message ?? String(e);
			person = null;
		}
		loading = false;
	}

	async function saveMeta() {
		if (!person) return;
		metaSaving = true;
		error = '';
		try {
			const updated = await updatePerson(personId, {
				access_level: accessDraft,
				notes: notesDraft,
			});
			person = { ...person, ...updated };
		} catch (e) {
			error = e.message ?? String(e);
		}
		metaSaving = false;
	}

	function pickFile(event) {
		const f = event.target.files?.[0];
		uploadFile = f ?? null;
	}

	async function submitUpload(event) {
		event.preventDefault();
		if (!uploadFile) return;
		uploading = true;
		error = '';
		try {
			await enrollImage(personId, uploadFile, { isPrimary: uploadPrimary });
			uploadFile = null;
			uploadPrimary = false;
			// Reset the file input by selector — Svelte 5 controlled-file is awkward.
			const input = document.querySelector('input[type="file"]');
			if (input) input.value = '';
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		uploading = false;
	}

	async function makePrimary(imageId) {
		busy = true;
		error = '';
		try {
			await setPrimaryImage(personId, imageId);
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		busy = false;
	}

	async function removeImage(imageId) {
		if (!confirm('Delete this image? This also removes its embedding from the index.')) return;
		busy = true;
		error = '';
		try {
			await deleteFaceImage(personId, imageId);
			await refresh();
		} catch (e) {
			error = e.message ?? String(e);
		}
		busy = false;
	}

	async function destroyPerson() {
		if (!person) return;
		const prompt1 = `Delete ${person.name}? This removes all images and embeddings.`;
		if (!confirm(prompt1)) return;
		busy = true;
		error = '';
		try {
			await deletePerson(personId);
			await goto('/people');
		} catch (e) {
			error = e.message ?? String(e);
			busy = false;
		}
	}

	function formatTime(iso) {
		if (!iso) return '';
		return new Date(iso).toLocaleString();
	}

	function sourceLabel(s) {
		const map = {
			upload: 'upload',
			detection_auto: 'auto-improvement',
			detection_confirmed: 'confirmed unknown',
			agent_enroll: 'agent enrolled',
		};
		return map[s] ?? s;
	}

	onMount(refresh);
</script>

<div class="page">
	<div class="breadcrumb">
		<a href="/people">← People</a>
	</div>

	{#if loading}
		<p class="muted">Loading...</p>
	{:else if !person}
		<div class="error-banner">{error || 'Person not found.'}</div>
	{:else}
		<header class="page-header">
			<div>
				<h1 class="page-title">{person.name}</h1>
				<p class="page-subtitle">
					{person.image_count} {person.image_count === 1 ? 'image' : 'images'}
					· created {formatTime(person.created_at)}
				</p>
			</div>
		</header>

		{#if error}
			<div class="error-banner">{error}</div>
		{/if}

		<section class="meta-card">
			<div class="meta-row">
				<label>
					Access level
					<select bind:value={accessDraft} disabled={metaSaving}>
						{#each ACCESS_LEVELS as lvl}
							<option value={lvl}>{lvl}</option>
						{/each}
					</select>
				</label>
				<label class="grow">
					Notes
					<input type="text" bind:value={notesDraft} disabled={metaSaving} placeholder="Optional notes" />
				</label>
				<button
					class="primary"
					onclick={saveMeta}
					disabled={metaSaving ||
						(accessDraft === person.access_level && (notesDraft || '') === (person.notes ?? ''))}
				>
					{metaSaving ? 'Saving...' : 'Save'}
				</button>
			</div>
		</section>

		<section class="upload-card">
			<form onsubmit={submitUpload}>
				<div class="upload-row">
					<input type="file" accept="image/*" onchange={pickFile} disabled={uploading} />
					<label class="checkbox">
						<input type="checkbox" bind:checked={uploadPrimary} disabled={uploading} />
						Set as primary
					</label>
					<button type="submit" class="primary" disabled={uploading || !uploadFile}>
						{uploading ? 'Enrolling...' : 'Enroll image'}
					</button>
				</div>
				<p class="hint">
					Single image · best face is auto-detected and embedded into the index.
				</p>
			</form>
		</section>

		<h2 class="section-title">Gallery</h2>
		{#if person.images.length === 0}
			<p class="muted">No images yet — upload one above to start identifying this person.</p>
		{:else}
			<div class="gallery">
				{#each person.images as img (img.id)}
					<div class="thumb" class:primary={img.is_primary}>
						<img src={faceImageUrl(img.id)} alt={person.name} />
						<div class="thumb-meta">
							<span class="src">{sourceLabel(img.source)}</span>
							{#if img.quality_score != null}
								<span class="q">q={img.quality_score.toFixed(2)}</span>
							{/if}
						</div>
						<div class="thumb-actions">
							{#if img.is_primary}
								<span class="primary-chip">primary</span>
							{:else}
								<button onclick={() => makePrimary(img.id)} disabled={busy}>
									Set primary
								</button>
								<button class="danger" onclick={() => removeImage(img.id)} disabled={busy}>
									Delete
								</button>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<section class="danger-zone">
			<h2 class="section-title">Danger zone</h2>
			<p class="muted">
				Permanently removes this person, every enrolled image, and their entries
				in the embedding index. Detection rows survive but lose their identity.
			</p>
			<button class="danger" onclick={destroyPerson} disabled={busy}>
				Delete {person.name}
			</button>
		</section>
	{/if}
</div>

<style>
	.page {
		max-width: 1100px;
		margin: 0 auto;
	}
	.breadcrumb {
		margin-bottom: 12px;
	}
	.breadcrumb a {
		font-size: 13px;
		color: #9ca3af;
	}
	.breadcrumb a:hover {
		color: #e1e4e8;
	}
	.page-header {
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
	.error-banner {
		background: rgba(239, 68, 68, 0.08);
		border: 1px solid rgba(239, 68, 68, 0.4);
		color: #fca5a5;
		padding: 10px 14px;
		border-radius: 8px;
		margin-bottom: 16px;
		font-size: 13px;
	}
	.meta-card,
	.upload-card {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		padding: 16px;
		margin-bottom: 20px;
	}
	.meta-row {
		display: flex;
		gap: 12px;
		align-items: flex-end;
		flex-wrap: wrap;
	}
	.meta-row label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 12px;
		color: #9ca3af;
	}
	.meta-row .grow {
		flex: 1;
		min-width: 200px;
	}
	.meta-row select,
	.meta-row input,
	.upload-row input[type='file'] {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		padding: 8px 10px;
		border-radius: 6px;
		font-size: 13px;
	}
	.upload-row {
		display: flex;
		gap: 12px;
		align-items: center;
		flex-wrap: wrap;
	}
	.checkbox {
		display: flex;
		gap: 6px;
		align-items: center;
		font-size: 13px;
		color: #c9cdd5;
		cursor: pointer;
	}
	.hint {
		margin-top: 8px;
		font-size: 12px;
		color: #6b7280;
	}
	button {
		background: #2d3148;
		color: #e1e4e8;
		border: 0;
		padding: 8px 14px;
		border-radius: 6px;
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
	}
	button:hover:not(:disabled) {
		background: #3a3f56;
	}
	button.primary {
		background: #6366f1;
		color: white;
	}
	button.primary:hover:not(:disabled) {
		background: #5558e6;
	}
	button.danger {
		background: rgba(239, 68, 68, 0.15);
		color: #f87171;
	}
	button.danger:hover:not(:disabled) {
		background: rgba(239, 68, 68, 0.25);
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.section-title {
		font-size: 16px;
		font-weight: 600;
		color: #e1e4e8;
		margin: 24px 0 12px;
	}
	.muted {
		color: #6b7280;
		font-size: 13px;
		margin-bottom: 8px;
	}
	.gallery {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
		gap: 12px;
	}
	.thumb {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 10px;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}
	.thumb.primary {
		border-color: #6366f1;
		box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.4);
	}
	.thumb img {
		width: 100%;
		aspect-ratio: 1;
		object-fit: cover;
		background: #0f1117;
	}
	.thumb-meta {
		padding: 8px 10px 0;
		display: flex;
		justify-content: space-between;
		font-size: 11px;
		color: #9ca3af;
	}
	.thumb-actions {
		padding: 8px 10px 10px;
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.thumb-actions button {
		padding: 4px 10px;
		font-size: 12px;
	}
	.primary-chip {
		font-size: 11px;
		font-weight: 600;
		color: #a78bfa;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.danger-zone {
		margin-top: 32px;
		padding: 16px;
		background: rgba(239, 68, 68, 0.04);
		border: 1px solid rgba(239, 68, 68, 0.2);
		border-radius: 12px;
	}
</style>
