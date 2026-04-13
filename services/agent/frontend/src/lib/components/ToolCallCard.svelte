<script>
	let { event } = $props();
	let expanded = $state(false);
</script>

<div class="tool-card" class:has-result={event.type === 'tool_result'}>
	<button class="tool-header" onclick={() => expanded = !expanded}>
		<span class="tool-icon">
			{#if event.type === 'tool_call'}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>
			{:else}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
			{/if}
		</span>
		<span class="tool-name">
			{event.type === 'tool_call' ? 'Calling' : 'Result from'} <strong>{event.tool}</strong>
		</span>
		<span class="expand-icon">{expanded ? '-' : '+'}</span>
	</button>

	{#if expanded}
		<div class="tool-body">
			{#if event.type === 'tool_call' && event.args}
				<pre class="tool-data">{JSON.stringify(event.args, null, 2)}</pre>
			{/if}
			{#if event.type === 'tool_result' && event.result}
				<pre class="tool-data">{typeof event.result === 'string' ? event.result : JSON.stringify(event.result, null, 2)}</pre>
			{/if}
		</div>
	{/if}
</div>

<style>
	.tool-card {
		background: #1a1d2e;
		border: 1px solid #2d3148;
		border-radius: 8px;
		margin: 6px 0;
		overflow: hidden;
	}

	.tool-header {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 8px 12px;
		background: none;
		border: none;
		color: #c9cdd5;
		font-size: 12px;
		cursor: pointer;
		text-align: left;
	}

	.tool-header:hover {
		background: #252a3e;
	}

	.tool-icon {
		display: flex;
		align-items: center;
	}

	.tool-name {
		flex: 1;
	}

	.tool-name strong {
		color: #a78bfa;
	}

	.expand-icon {
		color: #6b7280;
		font-size: 16px;
		font-weight: bold;
	}

	.tool-body {
		padding: 0 12px 10px;
	}

	.tool-data {
		background: #0f1117;
		border-radius: 6px;
		padding: 10px;
		font-size: 11px;
		color: #9ca3af;
		overflow-x: auto;
		max-height: 200px;
		overflow-y: auto;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
