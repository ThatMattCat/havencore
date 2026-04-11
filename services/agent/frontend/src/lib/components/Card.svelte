<script>
	let { title = '', children } = $props();
</script>

<div class="card">
	{#if title}
		<div class="card-header">{title}</div>
	{/if}
	<div class="card-body">
		{@render children()}
	</div>
</div>

<style>
	.card {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		overflow: hidden;
	}

	.card-header {
		padding: 14px 18px;
		font-size: 14px;
		font-weight: 600;
		color: #c9cdd5;
		border-bottom: 1px solid #2d3148;
	}

	.card-body {
		padding: 16px 18px;
	}
</style>
