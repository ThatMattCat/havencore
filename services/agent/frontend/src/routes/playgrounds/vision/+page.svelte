<script>
	import Card from '$lib/components/Card.svelte';
	import { visionAsk } from '$lib/api';

	let file = $state(null);
	let preview = $state('');
	let prompt = $state('What do you see in this image?');
	let response = $state('');
	let loading = $state(false);
	let error = $state('');
	let latencyMs = $state(0);
	let usage = $state(null);

	function onFile(e) {
		const f = e.target.files?.[0] || null;
		file = f;
		if (preview) URL.revokeObjectURL(preview);
		preview = f ? URL.createObjectURL(f) : '';
	}

	async function ask() {
		if (!file || !prompt.trim()) return;
		loading = true;
		error = '';
		response = '';
		usage = null;
		try {
			const result = await visionAsk(file, prompt);
			response = result.response;
			latencyMs = result.latency_ms;
			usage = result.usage;
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
		<h1>Vision</h1>
	</div>

	<div class="layout">
		<Card title="Input">
			<label class="field">
				<span>Image</span>
				<input type="file" accept="image/*" onchange={onFile} />
			</label>
			{#if preview}
				<img class="preview" src={preview} alt="preview" />
			{/if}
			<label class="field">
				<span>Prompt</span>
				<textarea rows="4" bind:value={prompt}></textarea>
			</label>
			<button onclick={ask} disabled={loading || !file || !prompt.trim()}>
				{loading ? 'Asking…' : 'Ask'}
			</button>
		</Card>

		<Card title="Response">
			{#if error}<div class="error">{error}</div>
			{:else if response}
				<div class="response">{response}</div>
				<div class="meta">
					Model latency: {latencyMs} ms
					{#if usage?.total_tokens}
						· {usage.total_tokens} tokens
					{/if}
				</div>
			{:else}
				<p class="muted">Upload an image and ask a question.</p>
			{/if}
		</Card>
	</div>
</div>

<style>
	.header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 20px; }
	.back { color: #a78bfa; font-size: 13px; }
	h1 { font-size: 24px; font-weight: 600; color: #f0f0f0; }
	.layout {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
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
	input[type="file"], textarea {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 14px;
		font-family: inherit;
	}
	textarea { resize: vertical; }
	.preview {
		max-width: 100%;
		max-height: 280px;
		object-fit: contain;
		border-radius: 8px;
		border: 1px solid #2d3148;
		margin-bottom: 12px;
	}
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
	.response {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 12px;
		white-space: pre-wrap;
		font-size: 14px;
	}
	.meta { margin-top: 10px; color: #6b7280; font-size: 12px; }
	.muted { color: #6b7280; font-size: 13px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
	}
</style>
