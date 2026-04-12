<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import {
		getTtsHealth,
		getSttHealth,
		getVisionHealth,
		getComfyHealth,
	} from '$lib/api';

	let health = $state({
		tts: 'unknown',
		stt: 'unknown',
		vision: 'unknown',
		comfy: 'unknown',
	});

	async function check(name, fn) {
		try {
			const res = await fn();
			health[name] = res?.status === 'healthy' ? 'healthy' : 'unhealthy';
		} catch {
			health[name] = 'unhealthy';
		}
	}

	onMount(() => {
		check('tts', getTtsHealth);
		check('stt', getSttHealth);
		check('vision', getVisionHealth);
		check('comfy', getComfyHealth);
	});

	const services = [
		{
			key: 'tts',
			href: '/playgrounds/tts',
			title: 'Text to Speech',
			desc: 'Synthesize speech with Kokoro TTS. Pick a voice, choose a format, hear it back.',
		},
		{
			key: 'stt',
			href: '/playgrounds/stt',
			title: 'Speech to Text',
			desc: 'Upload an audio file or stream live from your microphone using Faster Whisper.',
		},
		{
			key: 'vision',
			href: '/playgrounds/vision',
			title: 'Vision',
			desc: 'Ask the Qwen2.5-Omni vision model questions about an image.',
		},
		{
			key: 'comfy',
			href: '/playgrounds/comfy',
			title: 'Image Generation',
			desc: 'Generate images with the ComfyUI stack using text prompts.',
		},
	];
</script>

<div class="page">
	<h1 class="page-title">Service Playgrounds</h1>
	<p class="page-subtitle">Exercise each microservice directly from the dashboard.</p>

	<div class="grid">
		{#each services as svc}
			<a class="card-link" href={svc.href}>
				<Card title={svc.title}>
					<p class="desc">{svc.desc}</p>
					<div class="footer">
						<StatusBadge
							status={health[svc.key]}
							label={health[svc.key] === 'healthy' ? 'Online' : health[svc.key] === 'unhealthy' ? 'Offline' : 'Checking'}
						/>
						<span class="arrow">Open →</span>
					</div>
				</Card>
			</a>
		{/each}
	</div>
</div>

<style>
	.page-title {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}
	.page-subtitle {
		color: #6b7280;
		font-size: 14px;
		margin-bottom: 20px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
		gap: 16px;
	}
	.card-link {
		display: block;
		transition: transform 0.15s ease;
	}
	.card-link:hover {
		transform: translateY(-2px);
	}
	.desc {
		color: #9ca3af;
		font-size: 13px;
		line-height: 1.5;
		margin-bottom: 12px;
	}
	.footer {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.arrow {
		color: #a78bfa;
		font-size: 13px;
		font-weight: 500;
	}
</style>
