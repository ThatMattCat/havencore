<script>
	import Card from '$lib/components/Card.svelte';
	import { comfyGenerate, comfyStatus } from '$lib/api';

	let prompt = $state('a cozy cabin in a snowy forest at dusk, warm light in the windows');
	let negative = $state('');
	let seed = $state('');
	let steps = $state('');
	let advanced = $state(false);

	let jobId = $state('');
	let status = $state('idle'); // idle | pending | done | error
	let images = $state([]);
	let elapsedMs = $state(0);
	let error = $state('');

	let pollTimer = null;

	async function generate() {
		error = '';
		images = [];
		status = 'pending';
		const body = { prompt };
		if (negative.trim()) body.negative_prompt = negative.trim();
		if (seed !== '') body.seed = Number(seed);
		if (steps !== '') body.steps = Number(steps);

		try {
			const res = await comfyGenerate(body);
			jobId = res.job_id;
			poll();
		} catch (e) {
			error = e.message || String(e);
			status = 'error';
		}
	}

	async function poll() {
		if (!jobId) return;
		try {
			const s = await comfyStatus(jobId);
			status = s.status;
			elapsedMs = s.elapsed_ms;
			if (s.status === 'done') {
				images = s.images;
			} else if (s.status === 'error') {
				error = s.error || 'generation failed';
			} else {
				pollTimer = setTimeout(poll, 1000);
			}
		} catch (e) {
			error = e.message || String(e);
			status = 'error';
		}
	}
</script>

<div class="page">
	<div class="header">
		<a href="/playgrounds" class="back">← Playgrounds</a>
		<h1>Image Generation</h1>
	</div>

	<div class="layout">
		<Card title="Prompt">
			<label class="field">
				<span>Prompt</span>
				<textarea rows="5" bind:value={prompt}></textarea>
			</label>

			<button class="toggle" onclick={() => (advanced = !advanced)}>
				{advanced ? '▾' : '▸'} Advanced
			</button>
			{#if advanced}
				<label class="field">
					<span>Negative prompt</span>
					<textarea rows="2" bind:value={negative}></textarea>
				</label>
				<div class="row">
					<label class="field">
						<span>Seed</span>
						<input type="number" bind:value={seed} placeholder="random" />
					</label>
					<label class="field">
						<span>Steps</span>
						<input type="number" bind:value={steps} placeholder="default" />
					</label>
				</div>
			{/if}

			<button onclick={generate} disabled={status === 'pending' || !prompt.trim()}>
				{status === 'pending' ? 'Generating…' : 'Generate'}
			</button>
		</Card>

		<Card title="Result">
			{#if error}
				<div class="error">{error}</div>
			{:else if status === 'pending'}
				<div class="generating">
					<div class="spinner"></div>
					<p>Queued… {Math.round(elapsedMs / 1000)}s</p>
				</div>
			{:else if images.length > 0}
				<div class="meta">Completed in {Math.round(elapsedMs / 1000)}s</div>
				<div class="grid">
					{#each images as img}
						<a href={img.url} target="_blank">
							<img src={img.url} alt={img.filename} />
						</a>
					{/each}
				</div>
			{:else}
				<p class="muted">Submit a prompt to generate an image.</p>
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
	textarea, input[type="number"] {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 14px;
		font-family: inherit;
	}
	textarea { resize: vertical; }
	.row { display: flex; gap: 10px; }
	.row .field { flex: 1; }
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
	button.toggle {
		background: none;
		color: #a78bfa;
		padding: 4px 0;
		font-size: 13px;
		text-align: left;
	}
	button:disabled { opacity: 0.5; cursor: not-allowed; }
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
		gap: 10px;
	}
	img {
		width: 100%;
		border-radius: 8px;
		border: 1px solid #2d3148;
	}
	.muted { color: #6b7280; font-size: 13px; }
	.meta { color: #6b7280; font-size: 12px; margin-bottom: 10px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
	}
	.generating {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		padding: 20px;
		color: #9ca3af;
	}
	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid #2d3148;
		border-top-color: #8b5cf6;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }
</style>
