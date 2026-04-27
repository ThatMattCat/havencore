<script>
	import { onMount } from 'svelte';
	import Card from './Card.svelte';

	let { defaultVoice = 'af_heart' } = $props();

	let text = $state('');
	let device = $state('');
	let voice = $state(defaultVoice);
	let volume = $state(0.5);
	let voices = $state([{ id: 'af_heart', label: 'af_heart' }]);
	let players = $state([]);
	let loading = $state(false);
	let error = $state('');
	let status = $state('');

	onMount(async () => {
		// Voices
		try {
			const r = await fetch('/api/tts/voices');
			if (r.ok) {
				const j = await r.json();
				if (Array.isArray(j.voices)) voices = j.voices;
			}
		} catch {}
		// Players
		try {
			const r = await fetch('/api/tts/players');
			if (r.ok) {
				const j = await r.json();
				const list = Array.isArray(j.players) ? j.players : [];
				players = list
					.map((p) => (typeof p === 'string' ? { name: p } : p))
					.filter((p) => p && p.name);
				if (!device && players.length > 0) device = players[0].name;
			}
		} catch (e) {
			// allow manual entry
		}
	});

	async function speak() {
		error = '';
		status = '';
		if (!text.trim()) {
			error = 'Text is required.';
			return;
		}
		if (!device.trim()) {
			error = 'Device is required.';
			return;
		}
		loading = true;
		try {
			const r = await fetch('/api/tts/announce', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text,
					device,
					voice: voice || null,
					volume: volume === '' || volume == null ? null : Number(volume),
				}),
			});
			if (!r.ok) {
				const detail = await r.text();
				throw new Error(`${r.status}: ${detail}`);
			}
			status = `Queued on ${device}.`;
		} catch (e) {
			error = e.message || String(e);
		} finally {
			loading = false;
		}
	}
</script>

<Card title="Speak to device">
	<div class="grid">
		<label class="field">
			<span>Text</span>
			<textarea
				class="input"
				rows="3"
				bind:value={text}
				placeholder="Say something on a speaker…"
			></textarea>
		</label>

		<div class="row">
			<label class="field">
				<span>Device</span>
				{#if players.length > 0}
					<select class="input" bind:value={device}>
						{#each players as p}
							<option value={p.name}>{p.name}</option>
						{/each}
					</select>
				{:else}
					<input class="input mono" type="text" bind:value={device} placeholder="Living Room" />
				{/if}
			</label>
			<label class="field">
				<span>Voice</span>
				<select class="input" bind:value={voice}>
					{#each voices as v}
						<option value={v.id}>{v.label || v.id}</option>
					{/each}
				</select>
			</label>
			<label class="field">
				<span>Volume</span>
				<input
					class="input"
					type="number"
					step="0.05"
					min="0"
					max="1"
					bind:value={volume}
				/>
			</label>
		</div>

		{#if error}
			<div class="error">{error}</div>
		{/if}
		{#if status}
			<div class="status">{status}</div>
		{/if}

		<div class="actions">
			<button class="btn primary" onclick={speak} disabled={loading}>
				{loading ? 'Speaking…' : 'Speak'}
			</button>
		</div>
	</div>
</Card>

<style>
	.grid {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
		gap: 10px;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 13px;
	}
	.field > span {
		color: #c9cdd5;
		font-size: 12px;
		font-weight: 500;
	}
	.input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 6px;
		padding: 7px 10px;
		font-size: 13px;
		font-family: inherit;
		width: 100%;
	}
	.input:focus {
		outline: none;
		border-color: #6366f1;
	}
	.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 12px;
	}
	.actions {
		display: flex;
		justify-content: flex-end;
	}
	.btn {
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 7px 14px;
		font-size: 13px;
		cursor: pointer;
		font-weight: 500;
	}
	.btn.primary {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border-color: transparent;
	}
	.btn.primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.error {
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid #7f1d1d;
		color: #f87171;
		padding: 6px 10px;
		border-radius: 6px;
		font-size: 12px;
	}
	.status {
		background: rgba(34, 197, 94, 0.1);
		border: 1px solid #14532d;
		color: #4ade80;
		padding: 6px 10px;
		border-radius: 6px;
		font-size: 12px;
	}
</style>
