<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import { getTtsVoices, ttsSpeak } from '$lib/api';

	let text = $state('Hello, I am Selene. It is good to hear my own voice.');
	let voice = $state('af_heart');
	let format = $state('mp3');
	let speed = $state(1.0);
	let voices = $state([]);
	let formats = $state(['mp3', 'wav', 'opus', 'aac', 'flac', 'pcm']);
	let audioUrl = $state('');
	let loading = $state(false);
	let error = $state('');
	let latencyMs = $state(0);

	onMount(async () => {
		try {
			const res = await getTtsVoices();
			voices = res.voices;
			formats = res.formats;
		} catch (e) {
			// keep defaults
		}
	});

	async function speak() {
		if (!text.trim()) return;
		loading = true;
		error = '';
		if (audioUrl) {
			URL.revokeObjectURL(audioUrl);
			audioUrl = '';
		}
		const started = performance.now();
		try {
			const blob = await ttsSpeak({ text, voice, format, speed });
			audioUrl = URL.createObjectURL(blob);
			latencyMs = Math.round(performance.now() - started);
		} catch (e) {
			error = e.message || String(e);
		} finally {
			loading = false;
		}
	}
</script>

<div class="page">
	<div class="header">
		<a href="/playgrounds" class="back">← Playgrounds</a>
		<h1>Text to Speech</h1>
	</div>

	<div class="layout">
		<Card title="Input">
			<label class="field">
				<span>Text</span>
				<textarea bind:value={text} rows="6" placeholder="Type something to speak..."></textarea>
			</label>

			<div class="row">
				<label class="field">
					<span>Voice</span>
					<select bind:value={voice}>
						{#each voices as v}
							<option value={v.id}>{v.label}</option>
						{/each}
					</select>
				</label>
				<label class="field">
					<span>Format</span>
					<select bind:value={format}>
						{#each formats as f}
							<option value={f}>{f}</option>
						{/each}
					</select>
				</label>
				<label class="field">
					<span>Speed</span>
					<input type="number" min="0.5" max="2" step="0.1" bind:value={speed} />
				</label>
			</div>

			<button onclick={speak} disabled={loading || !text.trim()}>
				{loading ? 'Synthesizing…' : 'Speak'}
			</button>
		</Card>

		<Card title="Output">
			{#if error}
				<div class="error">{error}</div>
			{:else if audioUrl}
				<audio controls src={audioUrl} autoplay></audio>
				<div class="meta">Generated in {latencyMs} ms · {format}</div>
				<a class="download" href={audioUrl} download={`tts.${format}`}>Download</a>
			{:else}
				<p class="muted">Submit text to hear synthesized speech.</p>
			{/if}
		</Card>
	</div>
</div>

<style>
	.header {
		display: flex;
		align-items: baseline;
		gap: 16px;
		margin-bottom: 20px;
	}
	.back {
		color: #a78bfa;
		font-size: 13px;
	}
	h1 {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}
	.layout {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 16px;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		margin-bottom: 12px;
		font-size: 13px;
		color: #c9cdd5;
	}
	textarea, select, input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 14px;
		font-family: inherit;
	}
	textarea { resize: vertical; }
	.row { display: flex; gap: 10px; flex-wrap: wrap; }
	.row .field { flex: 1; min-width: 100px; }
	button {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border: none;
		border-radius: 8px;
		padding: 10px 20px;
		font-size: 14px;
		font-weight: 500;
		cursor: pointer;
	}
	button:disabled { opacity: 0.5; cursor: not-allowed; }
	audio { width: 100%; }
	.meta { margin-top: 10px; color: #6b7280; font-size: 12px; }
	.muted { color: #6b7280; font-size: 13px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
	}
	.download {
		display: inline-block;
		margin-top: 10px;
		color: #a78bfa;
		font-size: 13px;
	}
</style>
