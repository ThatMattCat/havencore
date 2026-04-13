<script>
	let { status = 'unknown', label = '' } = $props();
</script>

<span class="badge" class:healthy={status === 'healthy'} class:unhealthy={status === 'unhealthy'} class:unknown={status === 'unknown'}>
	<span class="dot"></span>
	{label}
</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border-radius: 12px;
		font-size: 12px;
		font-weight: 500;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.healthy {
		background: rgba(34, 197, 94, 0.1);
		color: #4ade80;
	}
	.healthy .dot { background: #4ade80; }

	.unhealthy {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
	}
	.unhealthy .dot { background: #f87171; }

	.unknown {
		background: rgba(156, 163, 175, 0.1);
		color: #9ca3af;
	}
	.unknown .dot { background: #9ca3af; }
</style>
