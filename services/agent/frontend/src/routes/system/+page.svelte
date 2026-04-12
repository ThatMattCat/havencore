<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import LogStream from '$lib/components/LogStream.svelte';
	import { getStatus, getTools } from '$lib/api';

	let status = $state(null);
	let tools = $state(null);
	let loading = $state(true);
	let error = $state('');

	onMount(async () => {
		try {
			const [s, t] = await Promise.all([getStatus(), getTools()]);
			status = s;
			tools = t;
		} catch (e) {
			error = e.message;
		}
		loading = false;
	});

	function getModelName(models) {
		try {
			return models?.data?.[0]?.id || 'Unknown';
		} catch {
			return 'Unknown';
		}
	}
</script>

<div class="system-page">
	<h1 class="page-title">System</h1>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading}
		<p class="muted">Loading system info...</p>
	{:else}
		<div class="grid">
			<!-- Agent Status -->
			<Card title="Agent">
				{#if status}
					<div class="status-list">
						<div class="status-row">
							<span>Name</span>
							<span class="value">{status.agent.name}</span>
						</div>
						<div class="status-row">
							<span>Health</span>
							<StatusBadge status={status.agent.healthy ? 'healthy' : 'unhealthy'} label={status.agent.healthy ? 'Running' : 'Down'} />
						</div>
					</div>
				{/if}
			</Card>

			<!-- LLM Backend -->
			<Card title="LLM Backend">
				{#if status}
					<div class="status-list">
						<div class="status-row">
							<span>Status</span>
							<StatusBadge status={status.llm.healthy ? 'healthy' : 'unhealthy'} label={status.llm.healthy ? 'Online' : 'Offline'} />
						</div>
						{#if status.llm.healthy && status.llm.models}
							<div class="status-row">
								<span>Model</span>
								<span class="value mono">{getModelName(status.llm.models)}</span>
							</div>
						{/if}
						{#if status.llm.error}
							<div class="error-detail">{status.llm.error}</div>
						{/if}
					</div>
				{/if}
			</Card>

			<!-- Database -->
			<Card title="Database">
				{#if status}
					<div class="status-list">
						<div class="status-row">
							<span>PostgreSQL</span>
							<StatusBadge status={status.database.connected ? 'healthy' : 'unhealthy'} label={status.database.connected ? 'Connected' : 'Disconnected'} />
						</div>
					</div>
				{/if}
			</Card>

			<!-- MCP Servers -->
			<Card title="MCP Servers">
				{#if status}
					<div class="status-list">
						<div class="status-row">
							<span>Connected</span>
							<span class="value">{status.mcp.connected_servers?.length || 0} / {status.mcp.configured_servers?.length || 0}</span>
						</div>
						<div class="status-row">
							<span>Total Tools</span>
							<span class="value">{status.mcp.total_mcp_tools || 0}</span>
						</div>
					</div>

					{#if status.mcp.connected_servers?.length > 0}
						<div class="server-list">
							<h3 class="sub-header">Connected</h3>
							{#each status.mcp.connected_servers as server}
								<div class="server-row">
									<StatusBadge status="healthy" label={server} />
									{#if status.mcp.tools_by_server?.[server]}
										<span class="tool-badge">{status.mcp.tools_by_server[server]} tools</span>
									{/if}
								</div>
							{/each}
						</div>
					{/if}

					{#if status.mcp.failed_servers && Object.keys(status.mcp.failed_servers).length > 0}
						<div class="server-list">
							<h3 class="sub-header">Failed</h3>
							{#each Object.entries(status.mcp.failed_servers) as [name, err]}
								<div class="server-row">
									<StatusBadge status="unhealthy" label={name} />
									<span class="error-text">{err}</span>
								</div>
							{/each}
						</div>
					{/if}
				{/if}
			</Card>

			<!-- Tools by Server -->
			{#if tools}
				{#each Object.entries(tools.tools_by_server) as [server, serverTools]}
					<Card title="{server} ({serverTools.length} tools)">
						<div class="tool-list">
							{#each serverTools as tool}
								<div class="tool-item">
									<span class="tool-name">{tool.name}</span>
									{#if tool.description}
										<span class="tool-desc">{tool.description}</span>
									{/if}
								</div>
							{/each}
						</div>
					</Card>
				{/each}
			{/if}
		</div>

		<div class="logs-section">
			<Card title="Live logs">
				<LogStream />
			</Card>
		</div>
	{/if}
</div>

<style>
	.logs-section {
		margin-top: 16px;
	}
	.page-title {
		font-size: 24px;
		font-weight: 600;
		margin-bottom: 20px;
		color: #f0f0f0;
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

	.muted {
		color: #6b7280;
		font-size: 13px;
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
		gap: 16px;
	}

	.status-list {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.status-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 14px;
		color: #c9cdd5;
	}

	.value {
		color: #e1e4e8;
		font-weight: 500;
	}

	.mono {
		font-family: monospace;
		font-size: 13px;
	}

	.error-detail {
		font-size: 12px;
		color: #f87171;
		background: rgba(239, 68, 68, 0.1);
		padding: 8px 10px;
		border-radius: 6px;
		word-break: break-word;
	}

	.server-list {
		margin-top: 12px;
		padding-top: 10px;
		border-top: 1px solid #2d3148;
	}

	.sub-header {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		color: #6b7280;
		margin-bottom: 8px;
	}

	.server-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 0;
	}

	.tool-badge {
		font-size: 11px;
		color: #6b7280;
		background: #252a3e;
		padding: 2px 8px;
		border-radius: 10px;
	}

	.error-text {
		font-size: 11px;
		color: #f87171;
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.tool-list {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.tool-item {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 6px 0;
		border-bottom: 1px solid #1e2235;
	}

	.tool-item:last-child {
		border-bottom: none;
	}

	.tool-name {
		font-size: 13px;
		font-weight: 500;
		color: #a78bfa;
		font-family: monospace;
	}

	.tool-desc {
		font-size: 12px;
		color: #6b7280;
		line-height: 1.4;
	}
</style>
