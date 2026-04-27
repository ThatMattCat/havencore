<script>
	import { onMount } from 'svelte';
	import { listCameraZones, setCameraZone, clearCameraZone } from '$lib/api';

	let cameras = $state([]); // CameraZoneRow[]
	let knownZones = $state([]); // string[]
	let loading = $state(true);
	let error = $state('');

	// Per-row draft state — keyed by camera_entity. Lets the user type without
	// committing until they hit Save (or focus moves away).
	let drafts = $state({}); // entity → { zone, zone_label }
	let savingEntity = $state(''); // entity currently being saved
	let flashEntity = $state(''); // entity that just saved (for the green checkmark)
	let flashTimer = null;

	async function refresh() {
		loading = true;
		error = '';
		try {
			const data = await listCameraZones();
			cameras = data.cameras;
			knownZones = data.zones;
			drafts = Object.fromEntries(
				cameras.map((c) => [
					c.camera_entity,
					{ zone: c.zone ?? '', zone_label: c.zone_label ?? '' },
				]),
			);
		} catch (e) {
			error = e?.message ?? String(e);
		}
		loading = false;
	}

	function flashSaved(entity) {
		flashEntity = entity;
		if (flashTimer) clearTimeout(flashTimer);
		flashTimer = setTimeout(() => {
			flashEntity = '';
		}, 1500);
	}

	async function save(entity) {
		const draft = drafts[entity];
		if (!draft) return;
		const zone = (draft.zone ?? '').trim();
		const zoneLabel = (draft.zone_label ?? '').trim();
		savingEntity = entity;
		error = '';
		try {
			if (!zone) {
				// Empty zone means clear assignment.
				await clearCameraZone(entity);
			} else {
				await setCameraZone(entity, {
					zone,
					zone_label: zoneLabel || null,
				});
			}
			await refresh();
			flashSaved(entity);
		} catch (e) {
			error = e?.message ?? String(e);
		}
		savingEntity = '';
	}

	function isDirty(c) {
		const d = drafts[c.camera_entity] ?? { zone: '', zone_label: '' };
		const cleanZone = (d.zone ?? '').trim();
		const cleanLabel = (d.zone_label ?? '').trim();
		return cleanZone !== (c.zone ?? '') || cleanLabel !== (c.zone_label ?? '');
	}

	onMount(refresh);
</script>

<div class="page">
	<header class="page-header">
		<div>
			<h1 class="page-title">Cameras</h1>
			<p class="page-subtitle">
				Map each camera to a zone (front_door, backyard, driveway). The autonomy
				engine reasons about zones — not raw camera entity_ids — so the same
				notification logic generalizes across deployments.
			</p>
		</div>
	</header>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading}
		<p class="muted">Loading cameras…</p>
	{:else if cameras.length === 0}
		<p class="muted">
			No cameras discovered yet. Make sure face-recognition is running and at
			least one HA camera entity exists (paired with a binary_sensor.&lt;base&gt;_person).
		</p>
	{:else}
		<datalist id="known-zones">
			{#each knownZones as z}
				<option value={z}></option>
			{/each}
		</datalist>

		<div class="rows">
			{#each cameras as cam (cam.camera_entity)}
				{@const draft = drafts[cam.camera_entity] ?? { zone: '', zone_label: '' }}
				<div class="row">
					<div class="row-header">
						<span class="cam-name">{cam.camera_entity}</span>
						<div class="meta">
							{#if !cam.camera_exists}
								<span class="chip chip-warn">orphan</span>
							{:else if cam.current_state}
								<span class="chip">state: {cam.current_state}</span>
							{/if}
							{#if cam.sensor_entity}
								<span class="chip-soft">sensor: {cam.sensor_entity}</span>
							{/if}
						</div>
					</div>
					<div class="row-controls">
						<input
							type="text"
							placeholder="zone slug (e.g. front_door)"
							list="known-zones"
							bind:value={drafts[cam.camera_entity].zone}
							disabled={savingEntity === cam.camera_entity}
						/>
						<input
							type="text"
							placeholder="display label (optional)"
							bind:value={drafts[cam.camera_entity].zone_label}
							disabled={savingEntity === cam.camera_entity}
						/>
						<button
							onclick={() => save(cam.camera_entity)}
							disabled={savingEntity === cam.camera_entity || !isDirty(cam)}
						>
							{#if savingEntity === cam.camera_entity}
								Saving…
							{:else if flashEntity === cam.camera_entity}
								Saved ✓
							{:else}
								Save
							{/if}
						</button>
					</div>
				</div>
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
		margin-top: 6px;
		max-width: 720px;
		line-height: 1.55;
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
		font-size: 14px;
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.row {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		padding: 14px 16px;
	}
	.row-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 10px;
	}
	.cam-name {
		font-family:
			'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
		font-size: 13px;
		color: #e1e4e8;
	}
	.meta {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.chip {
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 10px;
		background: rgba(156, 163, 175, 0.15);
		color: #9ca3af;
	}
	.chip-warn {
		background: rgba(251, 191, 36, 0.15);
		color: #fbbf24;
	}
	.chip-soft {
		font-size: 11px;
		color: #6b7280;
	}
	.row-controls {
		display: grid;
		grid-template-columns: 1.2fr 1.5fr auto;
		gap: 8px;
	}
	.row-controls input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		padding: 8px 10px;
		border-radius: 8px;
		font-size: 13px;
	}
	.row-controls input:focus {
		outline: none;
		border-color: #6366f1;
	}
	.row-controls button {
		background: #6366f1;
		color: white;
		border: 0;
		padding: 0 16px;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		min-width: 88px;
	}
	.row-controls button:disabled {
		background: #2d3148;
		color: #6b7280;
		cursor: not-allowed;
	}
</style>
