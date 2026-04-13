<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import { getEntities, getEntitySummary, getAutomations, getScenes } from '$lib/api';

	let domains = $state([]);
	let selectedDomain = $state(null);
	let entities = $state([]);
	let automations = $state([]);
	let scenes = $state([]);
	let loading = $state(true);
	let loadingEntities = $state(false);
	let error = $state('');

	const DOMAIN_ICONS = {
		light: '\u{1F4A1}',
		switch: '\u{1F50C}',
		media_player: '\u{1F3B5}',
		climate: '\u{1F321}',
		sensor: '\u{1F4CA}',
		binary_sensor: '\u{1F534}',
		cover: '\u{1FA9F}',
		fan: '\u{1F32C}',
		lock: '\u{1F512}',
		camera: '\u{1F4F7}',
		automation: '\u{2699}',
		scene: '\u{1F3AC}',
		person: '\u{1F464}',
		weather: '\u{26C5}',
		input_boolean: '\u{1F518}',
	};

	const ACTIVE_STATES = ['on', 'playing', 'open', 'home', 'active', 'heat', 'cool'];

	onMount(async () => {
		try {
			const [summary, autoData, sceneData] = await Promise.all([
				getEntitySummary(),
				getAutomations().catch(() => ({ automations: [], count: 0 })),
				getScenes().catch(() => ({ scenes: [], count: 0 })),
			]);
			domains = summary.domains.filter(d => d.total > 0);
			automations = autoData.automations;
			scenes = sceneData.scenes;
		} catch (e) {
			error = e.message;
		}
		loading = false;
	});

	async function selectDomain(domain) {
		selectedDomain = domain;
		loadingEntities = true;
		try {
			const data = await getEntities(domain);
			entities = data.entities;
		} catch (e) {
			error = e.message;
		}
		loadingEntities = false;
	}

	function isActive(state) {
		return ACTIVE_STATES.includes(state?.toLowerCase());
	}

	function domainIcon(domain) {
		return DOMAIN_ICONS[domain] || '\u{1F4E6}';
	}

	function formatTime(iso) {
		if (!iso) return '';
		return new Date(iso).toLocaleString();
	}
</script>

<div class="devices-page">
	<h1 class="page-title">Devices</h1>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading}
		<p class="muted">Loading devices...</p>
	{:else}
		<!-- Domain summary cards -->
		<div class="domain-grid">
			{#each domains as d}
				<button
					class="domain-card"
					class:selected={selectedDomain === d.domain}
					onclick={() => selectDomain(d.domain)}
				>
					<span class="domain-icon">{domainIcon(d.domain)}</span>
					<div class="domain-info">
						<span class="domain-name">{d.domain}</span>
						<span class="domain-count">
							{#if d.active > 0}
								<span class="active">{d.active} active</span> /
							{/if}
							{d.total} total
						</span>
					</div>
				</button>
			{/each}
		</div>

		<!-- Entity list for selected domain -->
		{#if selectedDomain}
			<div class="entity-section">
				<h2 class="section-title">{selectedDomain} entities</h2>
				{#if loadingEntities}
					<p class="muted">Loading...</p>
				{:else}
					<div class="entity-grid">
						{#each entities as entity}
							<div class="entity-card" class:active={isActive(entity.state)}>
								<div class="entity-header">
									<span class="entity-name">{entity.friendly_name || entity.entity_id}</span>
									<StatusBadge
										status={isActive(entity.state) ? 'healthy' : 'unhealthy'}
										label={entity.state}
									/>
								</div>
								<div class="entity-id">{entity.entity_id}</div>
								{#if entity.last_changed}
									<div class="entity-changed">Changed {formatTime(entity.last_changed)}</div>
								{/if}
							</div>
						{/each}
					</div>
					{#if entities.length === 0}
						<p class="muted">No entities in this domain</p>
					{/if}
				{/if}
			</div>
		{/if}

		<!-- Automations and Scenes -->
		<div class="extras-grid">
			<Card title="Automations ({automations.length})">
				{#if automations.length === 0}
					<p class="muted">No automations found</p>
				{:else}
					<div class="list">
						{#each automations as auto}
							<div class="list-item">
								<div class="list-name">{auto.friendly_name || auto.entity_id}</div>
								<div class="list-meta">
									<StatusBadge
										status={auto.state === 'on' ? 'healthy' : 'unhealthy'}
										label={auto.state}
									/>
									{#if auto.last_triggered}
										<span class="last-triggered">Last: {formatTime(auto.last_triggered)}</span>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</Card>

			<Card title="Scenes ({scenes.length})">
				{#if scenes.length === 0}
					<p class="muted">No scenes found</p>
				{:else}
					<div class="list">
						{#each scenes as scene}
							<div class="list-item">
								<span class="list-name">{scene.friendly_name || scene.entity_id}</span>
							</div>
						{/each}
					</div>
				{/if}
			</Card>
		</div>
	{/if}
</div>

<style>
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

	.domain-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
		gap: 10px;
		margin-bottom: 24px;
	}

	.domain-card {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 12px;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 10px;
		color: #c9cdd5;
		cursor: pointer;
		text-align: left;
		transition: all 0.15s;
	}

	.domain-card:hover {
		background: #1e2235;
	}

	.domain-card.selected {
		background: #252a3e;
		border-color: #6366f1;
	}

	.domain-icon {
		font-size: 22px;
		flex-shrink: 0;
	}

	.domain-info {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.domain-name {
		font-size: 13px;
		font-weight: 600;
		text-transform: capitalize;
		color: #e1e4e8;
	}

	.domain-count {
		font-size: 11px;
		color: #6b7280;
	}

	.domain-count .active {
		color: #4ade80;
	}

	.section-title {
		font-size: 16px;
		font-weight: 600;
		color: #e1e4e8;
		margin-bottom: 12px;
		text-transform: capitalize;
	}

	.entity-section {
		margin-bottom: 24px;
	}

	.entity-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
		gap: 10px;
	}

	.entity-card {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 10px;
		padding: 14px;
		transition: all 0.15s;
	}

	.entity-card.active {
		border-left: 3px solid #4ade80;
	}

	.entity-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		margin-bottom: 6px;
	}

	.entity-name {
		font-size: 14px;
		font-weight: 500;
		color: #e1e4e8;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.entity-id {
		font-size: 11px;
		color: #4b5563;
		font-family: monospace;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.entity-changed {
		font-size: 11px;
		color: #6b7280;
		margin-top: 4px;
	}

	.extras-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
		gap: 16px;
		margin-top: 8px;
	}

	.list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.list-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 0;
		border-bottom: 1px solid #1e2235;
		font-size: 13px;
	}

	.list-item:last-child {
		border-bottom: none;
	}

	.list-name {
		color: #c9cdd5;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}

	.list-meta {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-shrink: 0;
	}

	.last-triggered {
		font-size: 11px;
		color: #6b7280;
	}
</style>
