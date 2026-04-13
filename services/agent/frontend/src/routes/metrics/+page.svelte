<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import { getMetricSummary, getMetricTurns, getTopTools } from '$lib/api';

	let summary = $state(null);
	let turns = $state([]);
	let tools = $state([]);
	let error = $state('');

	onMount(async () => {
		try {
			const [s, t, tp] = await Promise.all([
				getMetricSummary(7),
				getMetricTurns(50),
				getTopTools(7, 10),
			]);
			summary = s;
			turns = t.turns;
			tools = tp.tools;
		} catch (e) {
			error = e.message || String(e);
		}
	});

	function formatTime(iso) {
		if (!iso) return '';
		return new Date(iso).toLocaleString();
	}

	function formatMs(n) {
		if (n < 1000) return `${n} ms`;
		return `${(n / 1000).toFixed(2)} s`;
	}

	function chartDays(perDay) {
		// Fill in missing days for the last 14 days so the chart has continuity
		const result = [];
		const byDay = new Map((perDay || []).map((d) => [d.day, d.turns]));
		const today = new Date();
		today.setHours(0, 0, 0, 0);
		for (let i = 13; i >= 0; i--) {
			const d = new Date(today);
			d.setDate(today.getDate() - i);
			const key = d.toISOString().slice(0, 10);
			result.push({ day: key, turns: byDay.get(key) || 0 });
		}
		return result;
	}

	let chart = $derived(chartDays(summary?.per_day));
	let chartMax = $derived(Math.max(1, ...chart.map((d) => d.turns)));
</script>

<div class="page">
	<h1>Metrics</h1>
	<p class="subtitle">Per-turn timings and agent activity over the last 7 days.</p>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	<div class="stats">
		<Card title="Turns today">
			<div class="stat">{summary?.turns_today ?? '–'}</div>
		</Card>
		<Card title="Avg turn (7d)">
			<div class="stat">{summary ? formatMs(summary.avg_total_ms) : '–'}</div>
		</Card>
		<Card title="Avg LLM latency (7d)">
			<div class="stat">{summary ? formatMs(summary.avg_llm_ms) : '–'}</div>
		</Card>
		<Card title="p95 turn (7d)">
			<div class="stat">{summary ? formatMs(summary.p95_total_ms) : '–'}</div>
		</Card>
	</div>

	<div class="row">
		<Card title="Activity (last 14 days)">
			<div class="chart">
				{#each chart as d}
					<div class="bar-col" title="{d.day}: {d.turns} turns">
						<div class="bar" style="height: {(d.turns / chartMax) * 100}%"></div>
						<div class="bar-label">{d.day.slice(5)}</div>
					</div>
				{/each}
			</div>
		</Card>

		<Card title="Top tools (7d)">
			{#if tools.length === 0}
				<p class="muted">No tool calls recorded yet.</p>
			{:else}
				<table class="table">
					<thead>
						<tr><th>Tool</th><th>Count</th><th>Avg ms</th></tr>
					</thead>
					<tbody>
						{#each tools as t}
							<tr>
								<td class="tool-name">{t.name}</td>
								<td>{t.count}</td>
								<td>{t.avg_ms}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</Card>
	</div>

	<Card title="Recent turns">
		{#if turns.length === 0}
			<p class="muted">No turns recorded yet. Send a chat message to populate metrics.</p>
		{:else}
			<div class="table-wrapper">
				<table class="table">
					<thead>
						<tr>
							<th>Time</th>
							<th>Total</th>
							<th>LLM</th>
							<th>Tools</th>
							<th>Iters</th>
							<th>Tools used</th>
						</tr>
					</thead>
					<tbody>
						{#each turns as t}
							<tr>
								<td>{formatTime(t.created_at)}</td>
								<td>{formatMs(t.total_ms)}</td>
								<td>{formatMs(t.llm_ms)}</td>
								<td>{formatMs(t.tool_ms_total)}</td>
								<td>{t.iterations}</td>
								<td class="tools-used">
									{#each t.tool_calls as c}
										<span class="chip">{c.name}:{c.ms}ms</span>
									{/each}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</Card>
</div>

<style>
	h1 { font-size: 24px; font-weight: 600; color: #f0f0f0; }
	.subtitle { color: #6b7280; font-size: 14px; margin-bottom: 20px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
		margin-bottom: 16px;
	}
	.stats {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 12px;
		margin-bottom: 16px;
	}
	.stat {
		font-size: 28px;
		font-weight: 700;
		color: #a78bfa;
	}
	.row {
		display: grid;
		grid-template-columns: 2fr 1fr;
		gap: 16px;
		margin-bottom: 16px;
	}
	@media (max-width: 900px) {
		.row { grid-template-columns: 1fr; }
	}
	.chart {
		display: flex;
		align-items: flex-end;
		gap: 6px;
		height: 180px;
	}
	.bar-col {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		height: 100%;
	}
	.bar {
		width: 100%;
		background: linear-gradient(180deg, #8b5cf6, #6366f1);
		border-radius: 4px 4px 0 0;
		min-height: 2px;
	}
	.bar-label {
		font-size: 10px;
		color: #6b7280;
		writing-mode: vertical-rl;
		transform: rotate(180deg);
	}
	.table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}
	.table th {
		text-align: left;
		color: #6b7280;
		font-weight: 500;
		padding: 6px 8px;
		border-bottom: 1px solid #2d3148;
	}
	.table td {
		padding: 6px 8px;
		border-bottom: 1px solid #1e2235;
		color: #c9cdd5;
	}
	.tool-name { color: #a78bfa; }
	.tools-used {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.chip {
		background: #252a3e;
		color: #c9cdd5;
		padding: 2px 6px;
		border-radius: 10px;
		font-size: 11px;
	}
	.muted { color: #6b7280; font-size: 13px; }
	.table-wrapper { overflow-x: auto; }
</style>
