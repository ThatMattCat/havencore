<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import SpeakToDevice from '$lib/components/SpeakToDevice.svelte';
	import {
		getTtsVoices,
		ttsSpeak,
		setTtsDefaultVoice,
		uploadTtsVoice,
		deleteTtsVoice,
	} from '$lib/api';

	let text = $state('Hello, I am Selene. It is good to hear my own voice.');
	let voice = $state('');
	let format = $state('mp3');
	let speed = $state(1.0);
	let voices = $state([]);
	let formats = $state(['mp3', 'wav', 'opus', 'aac', 'flac', 'pcm']);
	let defaultVoice = $state('');
	let defaultOverride = $state(null);
	let audioUrl = $state('');
	let loading = $state(false);
	let error = $state('');
	let latencyMs = $state(0);

	// --- voice cloning / management state ---
	let uploadName = $state('');
	let uploadFile = $state(null);
	let uploadBusy = $state(false);
	let uploadMsg = $state('');
	let uploadErr = $state('');
	let defaultBusy = $state('');
	let deletingName = $state('');

	async function refreshVoices() {
		try {
			const res = await getTtsVoices();
			voices = res.voices;
			formats = res.formats;
			defaultVoice = res.default || '';
			defaultOverride = res.default_override || null;
			// Keep the user's current selection if it still exists; otherwise
			// fall back to the default.
			if (!voices.find(v => v.id === voice)) {
				voice = defaultVoice;
			}
		} catch (e) {
			// keep defaults
		}
	}

	onMount(refreshVoices);

	async function speak() {
		if (!text.trim()) return;
		loading = true;
		error = '';
		if (audioUrl) {
			URL.revokeObjectURL(audioUrl);
			audioUrl = '';
		}
		const started = performance.now();
		try {
			// The playground is the one surface that bypasses the runtime
		// default — the user explicitly picked a voice to test, even if
		// they pick something other than the configured default.
		const blob = await ttsSpeak({ text, voice, format, speed, force_voice: true });
			audioUrl = URL.createObjectURL(blob);
			latencyMs = Math.round(performance.now() - started);
		} catch (e) {
			error = e.message || String(e);
		} finally {
			loading = false;
		}
	}

	async function makeDefault(name) {
		defaultBusy = name;
		try {
			await setTtsDefaultVoice(name);
			await refreshVoices();
		} catch (e) {
			error = e.message || String(e);
		} finally {
			defaultBusy = '';
		}
	}

	async function clearDefault() {
		defaultBusy = '__clear__';
		try {
			await setTtsDefaultVoice(null);
			await refreshVoices();
		} catch (e) {
			error = e.message || String(e);
		} finally {
			defaultBusy = '';
		}
	}

	async function deleteOne(name) {
		if (!confirm(`Delete the cloned voice "${name}"? This can't be undone.`)) return;
		deletingName = name;
		try {
			await deleteTtsVoice(name);
			await refreshVoices();
		} catch (e) {
			error = e.message || String(e);
		} finally {
			deletingName = '';
		}
	}

	function onFilePicked(ev) {
		const f = ev.target.files?.[0] || null;
		uploadFile = f;
		uploadErr = '';
		uploadMsg = '';
		// Auto-suggest a name from the filename (without extension).
		if (f && !uploadName) {
			const base = f.name.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]/g, '_');
			uploadName = base.slice(0, 40);
		}
	}

	async function doUpload() {
		uploadErr = '';
		uploadMsg = '';
		if (!uploadFile) {
			uploadErr = 'Pick a recording first.';
			return;
		}
		if (!/^[A-Za-z0-9_-]{1,40}$/.test(uploadName)) {
			uploadErr = 'Name must be 1-40 chars: letters, numbers, underscore, hyphen.';
			return;
		}
		uploadBusy = true;
		try {
			const res = await uploadTtsVoice(uploadName, uploadFile);
			uploadMsg = `Saved ${res.name} (${res.duration_sec}s).`;
			await refreshVoices();
			// Select the new voice + offer to make it default.
			voice = res.name;
			if (confirm(`Use "${res.name}" as the default voice for all TTS (chat, autonomy, dashboard)?`)) {
				await makeDefault(res.name);
			}
			uploadName = '';
			uploadFile = null;
			// Reset the file input visually.
			const input = document.getElementById('voice-upload-input');
			if (input) input.value = '';
		} catch (e) {
			uploadErr = e.message || String(e);
		} finally {
			uploadBusy = false;
		}
	}
</script>

<div class="page">
	<div class="header">
		<a href="/playgrounds" class="back">← Playgrounds</a>
		<h1>Text to Speech</h1>
	</div>

	<div class="layout">
		<Card title="Input">
			<label class="field">
				<span>Text</span>
				<textarea bind:value={text} rows="6" placeholder="Type something to speak..."></textarea>
			</label>

			<div class="row">
				<label class="field">
					<span>Voice</span>
					<select bind:value={voice}>
						{#each voices as v}
							<option value={v.id}>{v.label}{v.id === defaultVoice ? ' · default' : ''}</option>
						{/each}
					</select>
				</label>
				<label class="field">
					<span>Format</span>
					<select bind:value={format}>
						{#each formats as f}
							<option value={f}>{f}</option>
						{/each}
					</select>
				</label>
				<label class="field">
					<span>Speed</span>
					<input type="number" min="0.5" max="2" step="0.1" bind:value={speed} />
				</label>
			</div>

			<button onclick={speak} disabled={loading || !text.trim()}>
				{loading ? 'Synthesizing…' : 'Speak'}
			</button>
		</Card>

		<Card title="Output">
			{#if error}
				<div class="error">{error}</div>
			{:else if audioUrl}
				<audio controls src={audioUrl} autoplay></audio>
				<div class="meta">Generated in {latencyMs} ms · {format}</div>
				<a class="download" href={audioUrl} download={`tts.${format}`}>Download</a>
			{:else}
				<p class="muted">Submit text to hear synthesized speech.</p>
			{/if}
		</Card>
	</div>

	<div class="section">
		<Card title="Voices">
			<div class="default-row">
				<div>
					<div class="label">Default voice</div>
					<div class="value">
						{defaultVoice || '—'}
						{#if defaultOverride}
							<span class="badge">overridden</span>
						{/if}
					</div>
					<div class="hint">
						Used by the chat dashboard, autonomy speak channel, and any client
						that doesn't send an explicit <code>voice</code>. The companion app
						and ESP32 satellites all pick this up automatically.
					</div>
				</div>
				{#if defaultOverride}
					<button
						class="ghost"
						onclick={clearDefault}
						disabled={defaultBusy === '__clear__'}
					>
						{defaultBusy === '__clear__' ? 'Clearing…' : 'Reset to engine default'}
					</button>
				{/if}
			</div>

			<div class="upload">
				<div class="upload-head">
					<div class="upload-title">Clone a voice</div>
					<div class="upload-help">
						Upload 10–30 seconds of clean speech from a single speaker, minimal
						background noise. WAV / FLAC / OGG accepted (convert MP3 first).
						Chatterbox-Turbo clones the timbre on every request — no training step.
					</div>
				</div>
				<div class="upload-form">
					<label class="field">
						<span>Name</span>
						<input
							type="text"
							placeholder="e.g. Selene"
							maxlength="40"
							bind:value={uploadName}
						/>
					</label>
					<label class="field">
						<span>Recording</span>
						<input
							id="voice-upload-input"
							type="file"
							accept="audio/wav,audio/flac,audio/ogg,audio/x-flac,audio/vorbis,.wav,.flac,.ogg"
							onchange={onFilePicked}
						/>
					</label>
					<button onclick={doUpload} disabled={uploadBusy || !uploadFile || !uploadName}>
						{uploadBusy ? 'Uploading…' : 'Upload & clone'}
					</button>
				</div>
				{#if uploadMsg}<div class="ok">{uploadMsg}</div>{/if}
				{#if uploadErr}<div class="error">{uploadErr}</div>{/if}
			</div>

			<div class="voices-list">
				<div class="list-head">All voices</div>
				{#each voices.filter(v => v.kind) as v (v.id)}
					<div class="voice-row" class:is-default={v.id === defaultVoice}>
						<div class="voice-main">
							<div class="voice-name">
								{v.id}
								<span class="kind">{v.kind === 'user' ? 'cloned' : 'bundled'}</span>
								{#if v.id === defaultVoice}
									<span class="badge">default</span>
								{/if}
							</div>
						</div>
						<div class="voice-actions">
							{#if v.id !== defaultVoice}
								<button
									class="ghost"
									onclick={() => makeDefault(v.id)}
									disabled={defaultBusy === v.id}
								>
									{defaultBusy === v.id ? '…' : 'Set as default'}
								</button>
							{/if}
							{#if v.deletable}
								<button
									class="danger"
									onclick={() => deleteOne(v.id)}
									disabled={deletingName === v.id}
								>
									{deletingName === v.id ? '…' : 'Delete'}
								</button>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</Card>
	</div>

	<div class="section">
		<SpeakToDevice />
	</div>
</div>

<style>
	.header {
		display: flex;
		align-items: baseline;
		gap: 16px;
		margin-bottom: 20px;
	}
	.back {
		color: #a78bfa;
		font-size: 13px;
	}
	h1 {
		font-size: 24px;
		font-weight: 600;
		color: #f0f0f0;
	}
	.layout {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 16px;
	}
	.section { margin-top: 16px; }
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		margin-bottom: 12px;
		font-size: 13px;
		color: #c9cdd5;
	}
	textarea, select, input[type="number"], input[type="text"] {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 14px;
		font-family: inherit;
	}
	input[type="file"] {
		color: #e1e4e8;
		font-size: 13px;
	}
	textarea { resize: vertical; }
	.row { display: flex; gap: 10px; flex-wrap: wrap; }
	.row .field { flex: 1; min-width: 100px; }
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
	button:disabled { opacity: 0.5; cursor: not-allowed; }
	button.ghost {
		background: transparent;
		border: 1px solid #4b5170;
		color: #d4d4d8;
		padding: 6px 12px;
		font-size: 12px;
	}
	button.danger {
		background: rgba(239, 68, 68, 0.12);
		border: 1px solid rgba(239, 68, 68, 0.4);
		color: #fca5a5;
		padding: 6px 12px;
		font-size: 12px;
	}
	audio { width: 100%; }
	.meta { margin-top: 10px; color: #6b7280; font-size: 12px; }
	.muted { color: #6b7280; font-size: 13px; }
	.error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
		margin-top: 8px;
	}
	.ok {
		background: rgba(34, 197, 94, 0.1);
		color: #86efac;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
		margin-top: 8px;
	}
	.download {
		display: inline-block;
		margin-top: 10px;
		color: #a78bfa;
		font-size: 13px;
	}

	.default-row {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 16px;
		padding-bottom: 16px;
		border-bottom: 1px solid #2d3148;
		margin-bottom: 16px;
	}
	.default-row .label {
		font-size: 12px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #6b7280;
		margin-bottom: 4px;
	}
	.default-row .value {
		font-size: 18px;
		font-weight: 600;
		color: #f0f0f0;
	}
	.default-row .hint {
		margin-top: 6px;
		font-size: 12px;
		color: #6b7280;
		max-width: 540px;
		line-height: 1.45;
	}
	.default-row code {
		background: #0f1117;
		padding: 1px 5px;
		border-radius: 3px;
		font-size: 11px;
	}
	.badge {
		display: inline-block;
		background: rgba(167, 139, 250, 0.18);
		color: #c4b5fd;
		font-size: 10px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 1px 6px;
		border-radius: 4px;
		margin-left: 6px;
		vertical-align: middle;
	}

	.upload {
		background: #0f1117;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 14px;
		margin-bottom: 16px;
	}
	.upload-title {
		font-size: 14px;
		font-weight: 600;
		color: #f0f0f0;
		margin-bottom: 4px;
	}
	.upload-help {
		font-size: 12px;
		color: #6b7280;
		line-height: 1.45;
		margin-bottom: 12px;
	}
	.upload-form {
		display: grid;
		grid-template-columns: 1fr 1fr auto;
		gap: 10px;
		align-items: end;
	}
	.upload-form button { padding: 8px 14px; font-size: 13px; height: 36px; }

	.list-head {
		font-size: 12px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #6b7280;
		margin-bottom: 8px;
	}
	.voice-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		padding: 8px 10px;
		border-radius: 6px;
	}
	.voice-row:nth-child(even) { background: rgba(255, 255, 255, 0.02); }
	.voice-row.is-default { background: rgba(167, 139, 250, 0.08); }
	.voice-name {
		font-size: 14px;
		color: #e1e4e8;
		font-weight: 500;
	}
	.kind {
		display: inline-block;
		font-size: 10px;
		color: #6b7280;
		margin-left: 8px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.voice-actions { display: flex; gap: 6px; }
</style>
