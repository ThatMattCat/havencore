<script>
	import { onMount, onDestroy } from 'svelte';

	let { maxLines = 500 } = $props();

	let lines = $state([]);
	let status = $state('connecting');
	let paused = $state(false);
	let filter = $state('');
	let logEl;
	let autoScroll = true;
	let ws = null;
	let reconnectTimer = null;

	function fmtTs(ts) {
		try {
			const d = new Date(ts * 1000);
			return d.toTimeString().slice(0, 8);
		} catch {
			return '';
		}
	}

	function connect() {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		status = 'connecting';
		ws = new WebSocket(`${proto}//${location.host}/ws/logs`);
		ws.onopen = () => {
			status = 'connected';
		};
		ws.onmessage = (ev) => {
			if (paused) return;
			let msg;
			try {
				msg = JSON.parse(ev.data);
			} catch {
				return;
			}
			if (msg.type !== 'log') return;
			lines = [...lines.slice(-(maxLines - 1)), msg];
			queueMicrotask(() => {
				if (autoScroll && logEl) logEl.scrollTop = logEl.scrollHeight;
			});
		};
		ws.onerror = () => {
			status = 'error';
		};
		ws.onclose = () => {
			status = 'disconnected';
			ws = null;
			reconnectTimer = setTimeout(connect, 3000);
		};
	}

	function onScroll() {
		if (!logEl) return;
		const nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
		autoScroll = nearBottom;
	}

	function clear() {
		lines = [];
	}

	function togglePause() {
		paused = !paused;
	}

	let filtered = $derived(
		filter.trim()
			? lines.filter((l) => {
					const q = filter.toLowerCase();
					return (
						l.message?.toLowerCase().includes(q) ||
						l.logger?.toLowerCase().includes(q) ||
						l.level?.toLowerCase().includes(q)
					);
				})
			: lines,
	);

	onMount(() => {
		connect();
	});

	onDestroy(() => {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		try {
			ws?.close();
		} catch {}
	});
</script>

<div class="log-stream">
	<div class="toolbar">
		<span class="status status-{status}">{status}</span>
		<input class="filter" type="text" placeholder="filter…" bind:value={filter} />
		<button onclick={togglePause}>{paused ? 'Resume' : 'Pause'}</button>
		<button onclick={clear}>Clear</button>
		<span class="count">{filtered.length} / {lines.length}</span>
	</div>
	<div class="log-body" bind:this={logEl} onscroll={onScroll}>
		{#if filtered.length === 0}
			<div class="muted">Waiting for log entries…</div>
		{:else}
			{#each filtered as l}
				<div class="log-line level-{l.level?.toLowerCase()}">
					<span class="ts">{fmtTs(l.ts)}</span>
					<span class="level">{l.level}</span>
					<span class="logger">{l.logger}</span>
					<span class="msg">{l.message}</span>
				</div>
			{/each}
		{/if}
	</div>
</div>

<style>
	.log-stream {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.toolbar {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
	}
	.status {
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 11px;
		text-transform: uppercase;
	}
	.status-connected { background: rgba(16, 185, 129, 0.15); color: #34d399; }
	.status-connecting { background: rgba(234, 179, 8, 0.15); color: #facc15; }
	.status-disconnected, .status-error { background: rgba(239, 68, 68, 0.15); color: #f87171; }
	.filter {
		flex: 1;
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 6px;
		padding: 4px 8px;
		font-size: 12px;
	}
	button {
		background: #252a3e;
		color: #c9cdd5;
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 4px 10px;
		font-size: 12px;
		cursor: pointer;
	}
	button:hover { background: #2d3148; }
	.count { color: #6b7280; font-size: 11px; margin-left: auto; }
	.log-body {
		background: #0b0d14;
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 8px 10px;
		height: 320px;
		overflow-y: auto;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 12px;
		line-height: 1.45;
	}
	.log-line {
		display: flex;
		gap: 8px;
		padding: 1px 0;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.ts { color: #6b7280; flex-shrink: 0; }
	.level {
		flex-shrink: 0;
		width: 52px;
		font-weight: 600;
	}
	.logger { color: #8b5cf6; flex-shrink: 0; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.msg { color: #c9cdd5; flex: 1; }
	.level-info .level { color: #60a5fa; }
	.level-warning .level, .level-warn .level { color: #facc15; }
	.level-error .level, .level-critical .level { color: #f87171; }
	.level-error .msg, .level-critical .msg { color: #fca5a5; }
	.level-debug .level { color: #6b7280; }
	.muted { color: #6b7280; font-size: 12px; padding: 8px; }
</style>
