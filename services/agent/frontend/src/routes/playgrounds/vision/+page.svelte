<script>
	import Card from '$lib/components/Card.svelte';
	import { visionAsk } from '$lib/api';

	const PRESETS = [
		{ label: 'Describe scene', prompt: 'Describe what you see in this scene. Note people, animals, vehicles, packages, and anything that looks unusual. Be specific.' },
		{ label: "What's unusual?", prompt: 'What looks unusual or out of place here? If nothing stands out, say so plainly.' },
		{ label: 'Read all text', prompt: 'Transcribe all visible text. Preserve layout where it matters (receipts, forms, code). Mark illegible regions [illegible].' },
		{ label: 'Identify objects', prompt: 'List the main objects visible in the image. For each, give a short label and one-sentence description.' },
	];

	let file = $state(null);
	let preview = $state('');
	let mediaKind = $state(''); // 'image' | 'video' | ''
	let prompt = $state('What do you see in this image?');
	let response = $state('');
	let loading = $state(false);
	let error = $state('');
	let latencyMs = $state(0);
	let usage = $state(null);
	let model = $state('');

	function onFile(e) {
		const f = e.target.files?.[0] || null;
		file = f;
		if (preview) URL.revokeObjectURL(preview);
		preview = f ? URL.createObjectURL(f) : '';
		if (!f) {
			mediaKind = '';
		} else if (f.type.startsWith('video/')) {
			mediaKind = 'video';
		} else if (f.type.startsWith('image/')) {
			mediaKind = 'image';
		} else {
			mediaKind = '';
		}
	}

	function applyPreset(p) {
		prompt = p;
	}

	async function ask() {
		if (!file || !prompt.trim()) return;
		loading = true;
		error = '';
		response = '';
		usage = null;
		model = '';
		try {
			const result = await visionAsk(file, prompt);
			response = result.response;
			latencyMs = result.latency_ms;
			usage = result.usage;
			model = result.model || '';
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
				<span>Image or video</span>
				<input type="file" accept="image/*,video/*" onchange={onFile} />
			</label>
			{#if preview && mediaKind === 'image'}
				<img class="preview" src={preview} alt="preview" />
			{:else if preview && mediaKind === 'video'}
				<!-- svelte-ignore a11y_media_has_caption -->
				<video class="preview" src={preview} controls></video>
			{/if}

			<div class="presets">
				{#each PRESETS as p}
					<button class="preset" type="button" onclick={() => applyPreset(p.prompt)}>{p.label}</button>
				{/each}
			</div>

			<label class="field">
				<span>Prompt</span>
				<textarea rows="4" bind:value={prompt}></textarea>
			</label>
			<button class="primary" onclick={ask} disabled={loading || !file || !prompt.trim()}>
				{loading ? 'Asking…' : 'Ask'}
			</button>
		</Card>

		<Card title="Response">
			{#if error}<div class="error">{error}</div>
			{:else if response}
				<div class="response">{response}</div>
				<div class="meta">
					{#if model}<span class="badge">{model}</span>{/if}
					Latency: {latencyMs} ms
					{#if usage?.total_tokens}
						· {usage.total_tokens} tokens
						{#if usage.prompt_tokens !== undefined && usage.completion_tokens !== undefined}
							({usage.prompt_tokens} in / {usage.completion_tokens} out)
						{/if}
					{/if}
				</div>
			{:else}
				<p class="muted">Upload an image or short video and ask a question.</p>
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
		display: block;
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 12px;
	}
	.preset {
		background: #0f1117;
		color: #c9cdd5;
		border: 1px solid #2d3148;
		border-radius: 999px;
		padding: 4px 12px;
		font-size: 12px;
		cursor: pointer;
		font-family: inherit;
	}
	.preset:hover { border-color: #6366f1; color: #e1e4e8; }
	button.primary {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border: none;
		border-radius: 8px;
		padding: 10px 20px;
		font-size: 14px;
		font-weight: 500;
		cursor: pointer;
	}
	button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
	.response {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 12px;
		white-space: pre-wrap;
		font-size: 14px;
	}
	.meta { margin-top: 10px; color: #6b7280; font-size: 12px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
	.badge {
		background: #1a1d2b;
		border: 1px solid #2d3148;
		color: #a78bfa;
		font-family: ui-monospace, SFMono-Regular, monospace;
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 999px;
	}
	.muted { color: #6b7280; font-size: 13px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
	}
</style>
