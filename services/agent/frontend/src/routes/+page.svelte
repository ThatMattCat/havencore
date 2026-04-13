<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import { getStatus, getTools, listConversations, getEntitySummary } from '$lib/api';

	let status = $state(null);
	let tools = $state(null);
	let conversations = $state(null);
	let haSummary = $state(null);
	let memory = $state(null);
	let memoryLastRun = $state(null);
	let error = $state('');

	const L4_TOKEN_BUDGET = 1500;

	onMount(async () => {
		try {
			const [s, t, c] = await Promise.all([
				getStatus(),
				getTools(),
				listConversations(5, 0),
			]);
			status = s;
			tools = t;
			conversations = c.conversations;
		} catch (e) {
			error = e.message;
		}

		// HA summary is optional — don't fail the whole dashboard if HA is down
		try {
			haSummary = await getEntitySummary();
		} catch {}

		// Memory summary is optional — don't fail the dashboard if Qdrant is down
		try {
			const [m, runs] = await Promise.all([
				fetch('/api/memory/stats').then((r) => (r.ok ? r.json() : null)),
				fetch('/api/memory/runs?limit=1').then((r) => (r.ok ? r.json() : null)),
			]);
			memory = m;
			memoryLastRun = runs?.runs?.[0] ?? null;
		} catch {}
	});

	function formatTime(iso) {
		if (!iso) return '';
		const d = new Date(iso);
		return d.toLocaleString();
	}
</script>

<div class="dashboard">
	<h1 class="page-title">Dashboard</h1>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<div class="grid">
		<!-- System Status -->
		<Card title="System Status">
			{#if status}
				<div class="status-grid">
					<div class="status-row">
						<span>Agent</span>
						<StatusBadge status={status.agent.healthy ? 'healthy' : 'unhealthy'} label={status.agent.healthy ? 'Running' : 'Down'} />
					</div>
					<div class="status-row">
						<span>LLM Backend</span>
						<StatusBadge status={status.llm.healthy ? 'healthy' : 'unhealthy'} label={status.llm.healthy ? 'Online' : 'Offline'} />
					</div>
					<div class="status-row">
						<span>Database</span>
						<StatusBadge status={status.database.connected ? 'healthy' : 'unhealthy'} label={status.database.connected ? 'Connected' : 'Disconnected'} />
					</div>
					<div class="status-row">
						<span>MCP Servers</span>
						<StatusBadge
							status={status.mcp.connected_servers?.length > 0 ? 'healthy' : 'unhealthy'}
							label="{status.mcp.connected_servers?.length || 0}/{status.mcp.configured_servers?.length || 0} connected"
						/>
					</div>
					{#if status.mcp.failed_servers && Object.keys(status.mcp.failed_servers).length > 0}
						<div class="failed-servers">
							{#each Object.entries(status.mcp.failed_servers) as [name, err]}
								<div class="failed-item">
									<StatusBadge status="unhealthy" label={name} />
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{:else}
				<div class="loading">Loading...</div>
			{/if}
		</Card>

		<!-- Quick Chat -->
		<Card title="Quick Chat">
			<p class="muted">Talk to {status?.agent?.name || 'Selene'}</p>
			<a href="/chat" class="chat-link">Open Chat</a>
		</Card>

		<!-- Available Tools -->
		<Card title="Available Tools">
			{#if tools}
				<div class="tools-summary">
					<span class="big-number">{tools.total}</span>
					<span class="muted">tools registered</span>
				</div>
				<div class="tools-list">
					{#each Object.entries(tools.tools_by_server) as [server, serverTools]}
						<div class="tool-server">
							<span class="server-name">{server}</span>
							<span class="tool-count">{serverTools.length}</span>
						</div>
					{/each}
				</div>
			{:else}
				<div class="loading">Loading...</div>
			{/if}
		</Card>

		<!-- Recent Conversations -->
		<Card title="Recent Conversations">
			{#if conversations}
				{#if conversations.length === 0}
					<p class="muted">No conversations yet</p>
				{:else}
					<div class="conversation-list">
						{#each conversations as conv}
							<a href="/history" class="conversation-item">
								<span class="conv-time">{formatTime(conv.created_at)}</span>
								<span class="conv-meta">{conv.message_count} messages</span>
							</a>
						{/each}
					</div>
				{/if}
			{:else}
				<div class="loading">Loading...</div>
			{/if}
		</Card>

		<!-- Memory -->
		<Card title="Memory">
			{#if memory}
				<div class="memory-summary">
					<div class="tier-row">
						<span class="tier-label">L4 persistent</span>
						<span class="tier-count">{memory.l4_count}</span>
					</div>
					<div class="tier-row">
						<span class="tier-label">L3 consolidated</span>
						<span class="tier-count">{memory.l3_count}</span>
					</div>
					<div class="tier-row">
						<span class="tier-label">L2 episodic</span>
						<span class="tier-count">{memory.l2_count}</span>
					</div>
				</div>

				<div class="token-meter" title="L4 is injected into every system prompt">
					<div class="token-meter-header">
						<span class="muted">L4 token budget</span>
						<span class="token-meter-value" class:warn={memory.l4_est_tokens > L4_TOKEN_BUDGET}>
							{memory.l4_est_tokens} / {L4_TOKEN_BUDGET}
						</span>
					</div>
					<div class="token-meter-bar">
						<div
							class="token-meter-fill"
							class:warn={memory.l4_est_tokens > L4_TOKEN_BUDGET}
							style="width: {Math.min(100, (memory.l4_est_tokens / L4_TOKEN_BUDGET) * 100)}%"
						></div>
					</div>
				</div>

				{#if memory.pending_proposals > 0}
					<div class="memory-cta">
						<StatusBadge status="unhealthy" label="{memory.pending_proposals} proposals pending review" />
					</div>
				{/if}

				<div class="memory-footer">
					<span class="muted">
						Last consolidation: {memoryLastRun?.triggered_at ? formatTime(memoryLastRun.triggered_at) : 'never'}
					</span>
					<a href="/memory" class="see-all">Open memory</a>
				</div>
			{:else}
				<p class="muted">Memory unavailable</p>
			{/if}
		</Card>

		<!-- Home Assistant -->
		<Card title="Home Summary">
			{#if haSummary}
				<div class="ha-domains">
					{#each haSummary.domains.filter(d => d.total > 0).slice(0, 8) as domain}
						<div class="domain-row">
							<span class="domain-name">{domain.domain}</span>
							<span class="domain-counts">
								{#if domain.active > 0}
									<span class="active-count">{domain.active} active</span>
								{/if}
								<span class="total-count">{domain.total} total</span>
							</span>
						</div>
					{/each}
				</div>
				<a href="/devices" class="see-all">View all devices</a>
			{:else}
				<p class="muted">Home Assistant not available</p>
			{/if}
		</Card>
	</div>
</div>

<style>
	.page-title {
		font-size: 24px;
		font-weight: 600;
		margin-bottom: 20px;
		color: #f0f0f0;
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
		gap: 16px;
	}

	.error-banner {
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid #7f1d1d;
		color: #f87171;
		padding: 12px 16px;
		border-radius: 8px;
		margin-bottom: 16px;
		font-size: 14px;
	}

	.loading {
		color: #6b7280;
		font-size: 13px;
	}

	.muted {
		color: #6b7280;
		font-size: 13px;
	}

	.status-grid {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.status-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 14px;
	}

	.failed-servers {
		border-top: 1px solid #2d3148;
		padding-top: 8px;
		margin-top: 4px;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.chat-link {
		display: inline-block;
		margin-top: 12px;
		padding: 8px 16px;
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border-radius: 8px;
		font-size: 14px;
		font-weight: 500;
		transition: opacity 0.15s;
	}

	.chat-link:hover {
		opacity: 0.9;
	}

	.tools-summary {
		display: flex;
		align-items: baseline;
		gap: 8px;
		margin-bottom: 12px;
	}

	.big-number {
		font-size: 28px;
		font-weight: 700;
		color: #a78bfa;
	}

	.tools-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.tool-server {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
		padding: 4px 0;
	}

	.server-name {
		color: #9ca3af;
	}

	.tool-count {
		color: #6b7280;
		font-size: 12px;
		background: #252a3e;
		padding: 2px 8px;
		border-radius: 10px;
	}

	.conversation-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.conversation-item {
		display: flex;
		justify-content: space-between;
		padding: 8px 0;
		border-bottom: 1px solid #1e2235;
		font-size: 13px;
	}

	.conversation-item:last-child {
		border-bottom: none;
	}

	.conv-time {
		color: #c9cdd5;
	}

	.conv-meta {
		color: #6b7280;
	}

	.ha-domains {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 12px;
	}

	.domain-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 13px;
		padding: 4px 0;
	}

	.domain-name {
		color: #c9cdd5;
		text-transform: capitalize;
	}

	.domain-counts {
		display: flex;
		gap: 8px;
	}

	.active-count {
		color: #4ade80;
		font-size: 12px;
	}

	.total-count {
		color: #6b7280;
		font-size: 12px;
	}

	.see-all {
		font-size: 13px;
		color: #8b5cf6;
	}

	.see-all:hover {
		text-decoration: underline;
	}

	.memory-summary {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 12px;
	}

	.tier-row {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		font-size: 13px;
		padding: 4px 0;
	}

	.tier-label {
		color: #c9cdd5;
	}

	.tier-count {
		color: #a78bfa;
		font-weight: 600;
		font-size: 15px;
	}

	.token-meter {
		margin-bottom: 12px;
	}

	.token-meter-header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 6px;
	}

	.token-meter-value {
		font-size: 12px;
		color: #c9cdd5;
		font-variant-numeric: tabular-nums;
	}

	.token-meter-value.warn {
		color: #f87171;
	}

	.token-meter-bar {
		height: 6px;
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 4px;
		overflow: hidden;
	}

	.token-meter-fill {
		height: 100%;
		background: linear-gradient(90deg, #6366f1, #8b5cf6);
		transition: width 0.3s ease;
	}

	.token-meter-fill.warn {
		background: linear-gradient(90deg, #f59e0b, #f87171);
	}

	.memory-cta {
		margin-bottom: 12px;
	}

	.memory-footer {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
</style>
