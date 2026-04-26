<script>
	import { onMount } from 'svelte';

	let { src, alt = '', onclose } = $props();

	function handleKey(e) {
		if (e.key === 'Escape') onclose?.();
	}

	onMount(() => {
		window.addEventListener('keydown', handleKey);
		const prev = document.body.style.overflow;
		document.body.style.overflow = 'hidden';
		return () => {
			window.removeEventListener('keydown', handleKey);
			document.body.style.overflow = prev;
		};
	});
</script>

<div
	class="backdrop"
	onclick={() => onclose?.()}
	role="presentation"
>
	<button
		type="button"
		class="close"
		onclick={(e) => {
			e.stopPropagation();
			onclose?.();
		}}
		aria-label="Close"
	>×</button>
	<img
		{src}
		{alt}
		onclick={(e) => e.stopPropagation()}
	/>
</div>

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.85);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 200;
		padding: 24px;
		cursor: zoom-out;
	}
	img {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
		border-radius: 6px;
		cursor: default;
		box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
	}
	.close {
		position: absolute;
		top: 16px;
		right: 20px;
		background: rgba(22, 24, 34, 0.85);
		color: #e1e4e8;
		border: 1px solid #2d3148;
		width: 36px;
		height: 36px;
		border-radius: 50%;
		font-size: 22px;
		line-height: 1;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.close:hover {
		background: #2d3148;
	}
</style>
