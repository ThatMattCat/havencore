<script>
	import Card from '$lib/components/Card.svelte';
	import { sttTranscribe } from '$lib/api';

	let file = $state(null);
	let language = $state('');
	let transcript = $state('');
	let uploading = $state(false);
	let error = $state('');
	let latencyMs = $state(0);

	async function transcribe() {
		if (!file) return;
		uploading = true;
		error = '';
		transcript = '';
		const started = performance.now();
		try {
			const result = await sttTranscribe(file, language ? { language } : undefined);
			transcript = result.text || JSON.stringify(result);
			latencyMs = Math.round(performance.now() - started);
		} catch (e) {
			error = e.message || String(e);
		} finally {
			uploading = false;
		}
	}

	function onFile(e) {
		file = e.target.files?.[0] || null;
	}

	// Live mic: record full clip, upload on stop
	let recording = $state(false);
	let processing = $state(false);
	let liveTranscript = $state('');
	let liveError = $state('');
	let recordingSeconds = $state(0);
	let recordingUrl = $state('');
	let liveLatencyMs = $state(0);
	let mediaRecorder = null;
	let audioStream = null;
	let recordedChunks = [];
	let recordedMime = 'audio/webm';
	let timerInterval = null;
	let startedAt = 0;

	function pickMimeType() {
		const candidates = [
			'audio/webm;codecs=opus',
			'audio/webm',
			'audio/ogg;codecs=opus',
			'audio/mp4',
		];
		for (const t of candidates) {
			if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported?.(t)) {
				return t;
			}
		}
		return '';
	}

	async function startRecording() {
		liveError = '';
		liveTranscript = '';
		liveLatencyMs = 0;
		recordedChunks = [];
		if (recordingUrl) {
			URL.revokeObjectURL(recordingUrl);
			recordingUrl = '';
		}

		try {
			audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
		} catch (e) {
			liveError = `Microphone access denied: ${e.message}`;
			return;
		}

		const mimeType = pickMimeType();
		try {
			mediaRecorder = mimeType
				? new MediaRecorder(audioStream, { mimeType })
				: new MediaRecorder(audioStream);
		} catch (e) {
			liveError = `Recorder init failed: ${e.message}`;
			audioStream.getTracks().forEach((t) => t.stop());
			audioStream = null;
			return;
		}
		recordedMime = mediaRecorder.mimeType || mimeType || 'audio/webm';

		mediaRecorder.ondataavailable = (event) => {
			if (event.data && event.data.size > 0) recordedChunks.push(event.data);
		};
		mediaRecorder.onstop = onRecordingStopped;

		mediaRecorder.start();
		recording = true;
		startedAt = performance.now();
		recordingSeconds = 0;
		timerInterval = setInterval(() => {
			recordingSeconds = Math.floor((performance.now() - startedAt) / 1000);
		}, 250);
	}

	function stopRecording() {
		if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
		try {
			mediaRecorder.stop();
		} catch (e) {
			liveError = `Stop failed: ${e.message}`;
		}
		recording = false;
		if (timerInterval) {
			clearInterval(timerInterval);
			timerInterval = null;
		}
	}

	async function onRecordingStopped() {
		try {
			audioStream?.getTracks().forEach((t) => t.stop());
		} catch {}
		audioStream = null;
		mediaRecorder = null;

		if (recordedChunks.length === 0) {
			liveError = 'No audio was captured.';
			return;
		}

		const blob = new Blob(recordedChunks, { type: recordedMime });
		recordingUrl = URL.createObjectURL(blob);

		const ext = recordedMime.includes('mp4')
			? 'm4a'
			: recordedMime.includes('ogg')
				? 'ogg'
				: 'webm';
		const audioFile = new File([blob], `live-recording.${ext}`, { type: recordedMime });

		processing = true;
		const started = performance.now();
		try {
			const result = await sttTranscribe(audioFile, language ? { language } : undefined);
			liveTranscript = result.text || JSON.stringify(result);
			liveLatencyMs = Math.round(performance.now() - started);
		} catch (e) {
			liveError = e.message || String(e);
		} finally {
			processing = false;
		}
	}
</script>

<div class="page">
	<div class="header">
		<a href="/playgrounds" class="back">← Playgrounds</a>
		<h1>Speech to Text</h1>
	</div>

	<div class="layout">
		<Card title="File upload">
			<label class="field">
				<span>Audio file</span>
				<input type="file" accept="audio/*" onchange={onFile} />
			</label>
			<label class="field">
				<span>Language (optional, e.g. &quot;en&quot;)</span>
				<input type="text" bind:value={language} placeholder="auto-detect" />
			</label>
			<button onclick={transcribe} disabled={uploading || !file}>
				{uploading ? 'Transcribing…' : 'Transcribe'}
			</button>
			{#if error}<div class="error">{error}</div>{/if}
			{#if transcript}
				<div class="transcript">
					<div class="transcript-label">Transcript · {latencyMs} ms</div>
					<div class="transcript-text">{transcript}</div>
				</div>
			{/if}
		</Card>

		<Card title="Microphone recording">
			<p class="hint">Record a clip from your mic; it will be transcribed after you press Stop.</p>
			{#if liveError}<div class="error">{liveError}</div>{/if}
			<div class="recorder-row">
				{#if !recording}
					<button onclick={startRecording} disabled={processing}>
						{processing ? 'Transcribing…' : 'Start recording'}
					</button>
				{:else}
					<button class="stop" onclick={stopRecording}>Stop</button>
					<span class="rec-indicator">● recording {recordingSeconds}s</span>
				{/if}
			</div>

			{#if recordingUrl && !processing}
				<audio class="playback" controls src={recordingUrl}></audio>
			{/if}

			{#if processing}
				<p class="muted">Uploading and transcribing…</p>
			{/if}

			{#if liveTranscript}
				<div class="transcript">
					<div class="transcript-label">Transcript · {liveLatencyMs} ms</div>
					<div class="transcript-text">{liveTranscript}</div>
				</div>
			{/if}
		</Card>
	</div>
</div>

<style>
	.header {
		display: flex;
		align-items: baseline;
		gap: 16px;
		margin-bottom: 20px;
	}
	.back { color: #a78bfa; font-size: 13px; }
	h1 { font-size: 24px; font-weight: 600; color: #f0f0f0; }
	.layout {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
		gap: 16px;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		margin-bottom: 12px;
		font-size: 13px;
		color: #c9cdd5;
	}
	input[type="text"], input[type="file"] {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 14px;
	}
	button {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border: none;
		border-radius: 8px;
		padding: 10px 20px;
		font-size: 14px;
		font-weight: 500;
		cursor: pointer;
	}
	button.stop {
		background: #dc2626;
	}
	button:disabled { opacity: 0.5; cursor: not-allowed; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
		margin-top: 10px;
	}
	.transcript { margin-top: 14px; }
	.transcript-label { color: #6b7280; font-size: 12px; margin-bottom: 4px; }
	.transcript-text {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 10px;
		font-size: 14px;
		white-space: pre-wrap;
	}
	.hint {
		color: #9ca3af;
		font-size: 12px;
		margin-bottom: 10px;
	}
	.recorder-row {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.rec-indicator {
		font-size: 13px;
		color: #f87171;
		font-family: ui-monospace, monospace;
		animation: rec-pulse 1.2s ease-in-out infinite;
	}
	@keyframes rec-pulse {
		0%, 100% { opacity: 0.5; }
		50% { opacity: 1; }
	}
	.playback {
		display: block;
		width: 100%;
		margin-top: 12px;
	}
	.muted { color: #6b7280; font-size: 13px; }
</style>
