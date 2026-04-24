<script>
	import { onMount, onDestroy, tick } from 'svelte';
	import { messages, isConnected, isProcessing, connectionState, currentSessionId, currentDeviceName, connect, sendMessage, disconnect, clearMessages, retryNow, startNewChat } from '$lib/stores/chat';
	import ToolCallCard from '$lib/components/ToolCallCard.svelte';
	import { sttTranscribe, ttsSpeak } from '$lib/api';
	import { marked } from 'marked';

	let inputText = $state('');
	let messagesContainer = $state(null);

	// Configure marked for safe rendering
	marked.setOptions({ breaks: true, gfm: true });

	onMount(() => {
		connect();
		try {
			autoSpeak = localStorage.getItem('chat.autoSpeak') === '1';
		} catch {}
	});

	onDestroy(() => {
		disconnect();
		stopPlayback();
		if (recording) stopRecording();
	});

	function handleSend() {
		if (!inputText.trim() || $isProcessing) return;
		stopPlayback();
		sendMessage(inputText);
		inputText = '';
		scrollToBottom();
	}

	function handleKeyDown(e) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	async function scrollToBottom() {
		await tick();
		if (messagesContainer) {
			messagesContainer.scrollTop = messagesContainer.scrollHeight;
		}
	}

	// Auto-scroll on new messages
	$effect(() => {
		if ($messages.length > 0) {
			scrollToBottom();
		}
	});

	function renderMarkdown(text) {
		if (!text) return '';
		return marked.parse(text);
	}

	function fmtMs(n) {
		if (n == null) return '';
		if (n < 1000) return `${Math.round(n)}ms`;
		return `${(n / 1000).toFixed(2)}s`;
	}

	function getErrorEvent(events) {
		if (!events) return null;
		return events.find((e) => e.type === 'error') || null;
	}

	function shortSessionId(sid) {
		if (!sid) return '—';
		return sid.slice(-8);
	}

	function handleNewChat() {
		stopPlayback();
		if (recording) stopRecording();
		startNewChat();
	}

	const bannerLabels = {
		connecting: 'Connecting to agent…',
		reconnecting: 'Connection lost. Reconnecting…',
		disconnected: 'Disconnected from agent.',
	};

	// --- Push-to-talk (mic → STT → auto-send) ---
	let recording = $state(false);
	let transcribing = $state(false);
	let micError = $state('');
	let mediaRecorder = null;
	let audioStream = null;
	let recordedChunks = [];
	let recordedMime = 'audio/webm';

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
		if (!$isConnected || $isProcessing || transcribing) return;
		micError = '';
		recordedChunks = [];
		stopPlayback();
		try {
			audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
		} catch (e) {
			micError = `Microphone access denied: ${e.message || e}`;
			return;
		}
		const mimeType = pickMimeType();
		try {
			mediaRecorder = mimeType
				? new MediaRecorder(audioStream, { mimeType })
				: new MediaRecorder(audioStream);
		} catch (e) {
			micError = `Recorder init failed: ${e.message || e}`;
			audioStream.getTracks().forEach((t) => t.stop());
			audioStream = null;
			return;
		}
		recordedMime = mediaRecorder.mimeType || mimeType || 'audio/webm';
		mediaRecorder.ondataavailable = (ev) => {
			if (ev.data && ev.data.size > 0) recordedChunks.push(ev.data);
		};
		mediaRecorder.onstop = onRecordingStopped;
		mediaRecorder.start();
		recording = true;
	}

	function stopRecording() {
		if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
		try {
			mediaRecorder.stop();
		} catch (e) {
			micError = `Stop failed: ${e.message || e}`;
		}
		recording = false;
	}

	async function onRecordingStopped() {
		try {
			audioStream?.getTracks().forEach((t) => t.stop());
		} catch {}
		audioStream = null;
		mediaRecorder = null;

		if (recordedChunks.length === 0) {
			micError = 'No audio captured.';
			return;
		}

		const blob = new Blob(recordedChunks, { type: recordedMime });
		const ext = recordedMime.includes('mp4')
			? 'm4a'
			: recordedMime.includes('ogg')
				? 'ogg'
				: 'webm';
		const audioFile = new File([blob], `chat-mic.${ext}`, { type: recordedMime });

		transcribing = true;
		try {
			const result = await sttTranscribe(audioFile);
			const text = (result.text || '').trim();
			if (text) {
				stopPlayback();
				sendMessage(text);
				scrollToBottom();
			} else {
				micError = 'No speech detected.';
			}
		} catch (e) {
			micError = e.message || String(e);
		} finally {
			transcribing = false;
		}
	}

	function toggleMic() {
		if (recording) stopRecording();
		else startRecording();
	}

	// --- Auto-speak (TTS on assistant done) ---
	let autoSpeak = $state(false);
	let currentAudio = $state(null);
	let speaking = $state(false);
	let lastSpokenIndex = -1;
	let speakInitialized = false;

	function persistAutoSpeak() {
		try {
			localStorage.setItem('chat.autoSpeak', autoSpeak ? '1' : '0');
		} catch {}
	}

	function toggleAutoSpeak() {
		autoSpeak = !autoSpeak;
		persistAutoSpeak();
		if (!autoSpeak) stopPlayback();
	}

	function stopPlayback() {
		if (currentAudio) {
			try {
				currentAudio.pause();
			} catch {}
			try {
				URL.revokeObjectURL(currentAudio.src);
			} catch {}
			currentAudio = null;
		}
		speaking = false;
	}

	async function speak(text) {
		if (!text || !text.trim()) return;
		stopPlayback();
		speaking = true;
		try {
			const blob = await ttsSpeak({ text, voice: 'af_heart', format: 'mp3', speed: 1.0 });
			const url = URL.createObjectURL(blob);
			const audio = new Audio(url);
			currentAudio = audio;
			audio.onended = () => {
				speaking = false;
				try { URL.revokeObjectURL(url); } catch {}
				if (currentAudio === audio) currentAudio = null;
			};
			audio.onerror = () => {
				speaking = false;
				try { URL.revokeObjectURL(url); } catch {}
				if (currentAudio === audio) currentAudio = null;
			};
			await audio.play();
		} catch (e) {
			speaking = false;
			micError = `TTS failed: ${e.message || e}`;
		}
	}

	// Watch for newly completed assistant messages and speak them.
	// On first run (fresh mount, possibly with existing messages from a prior
	// visit to this page) we snapshot the current length so we never replay
	// turns that finished before the user navigated to /chat this time.
	$effect(() => {
		const msgs = $messages;
		if (!speakInitialized) {
			lastSpokenIndex = msgs.length - 1;
			speakInitialized = true;
			return;
		}
		if (lastSpokenIndex >= msgs.length) {
			lastSpokenIndex = msgs.length - 1;
		}
		if (!autoSpeak) {
			lastSpokenIndex = msgs.length - 1;
			return;
		}
		for (let i = lastSpokenIndex + 1; i < msgs.length; i++) {
			const m = msgs[i];
			if (m.role === 'assistant' && m.content && m.metric) {
				lastSpokenIndex = i;
				speak(m.content);
			} else if (m.role === 'user') {
				lastSpokenIndex = i;
			}
		}
	});
</script>

<div class="chat-page">
	<div class="chat-header">
		<h1 class="page-title">Chat with Selene</h1>
		<div class="header-actions">
			<span class="connection-status" class:connected={$isConnected}>
				<span class="dot"></span>
				{$isConnected ? 'Connected' : 'Disconnected'}
			</span>
			<span class="session-badge" title={$currentSessionId ?? 'No session yet'}>
				{#if $currentDeviceName}
					<span class="device-label">{$currentDeviceName}</span>
					<span class="badge-sep">·</span>
				{/if}
				session · <span class="session-id">{shortSessionId($currentSessionId)}</span>
			</span>
			<button
				class="icon-btn"
				class:active={autoSpeak}
				onclick={toggleAutoSpeak}
				title={autoSpeak ? 'Auto-speak on (click to disable)' : 'Auto-speak off (click to enable)'}
				aria-label="Toggle auto-speak"
			>
				{#if autoSpeak}
					<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
				{:else}
					<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
				{/if}
				{#if speaking}<span class="speaking-dot"></span>{/if}
			</button>
			<button class="clear-btn" onclick={handleNewChat} title="Start a fresh session">New Chat</button>
			<button class="clear-btn" onclick={clearMessages} title="Clear visible messages (same session)">Clear</button>
		</div>
	</div>

	{#if $connectionState !== 'connected'}
		<div class="conn-banner" class:disconnected={$connectionState === 'disconnected'}>
			<span class="conn-banner-dot"></span>
			<span class="conn-banner-text">{bannerLabels[$connectionState] ?? 'Connecting…'}</span>
			<button class="conn-banner-btn" onclick={retryNow}>Retry now</button>
		</div>
	{/if}

	<div class="messages" bind:this={messagesContainer}>
		{#if $messages.length === 0}
			<div class="empty-state">
				<div class="empty-icon">
					<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
				</div>
				<p>Start a conversation with Selene</p>
				<p class="muted">Ask about your home, the weather, or anything else</p>
			</div>
		{/if}

		{#each $messages as msg, i}
			{#if msg.role === 'summary'}
				<details class="summary-card">
					<summary class="summary-card-header">
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
						<span class="summary-card-label">Conversation summarized</span>
						<span class="summary-card-hint">click to view</span>
					</summary>
					<div class="summary-card-body">
						{msg.content || '(summary unavailable)'}
					</div>
				</details>
			{:else}
				<div class="message" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'}>
					<div class="message-avatar">
						{#if msg.role === 'user'}
							<span class="avatar user-avatar">U</span>
						{:else}
							<span class="avatar assistant-avatar">S</span>
						{/if}
					</div>
					<div class="message-body">
						{#if msg.role === 'assistant'}
							<!-- Show tool events -->
							{#each msg.events.filter(e => e.type === 'tool_call' || e.type === 'tool_result') as event}
								<ToolCallCard {event} />
							{/each}

							<!-- Show reasoning (dashboard-only chain-of-thought) -->
							{#each msg.events.filter(e => e.type === 'reasoning') as event, i (i)}
								<details class="reasoning-card">
									<summary class="reasoning-card-header">
										<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
										<span class="reasoning-card-label">Reasoning</span>
										<span class="reasoning-card-hint">click to view</span>
									</summary>
									<div class="reasoning-card-body">{event.content || ''}</div>
								</details>
							{/each}

							<!-- Show thinking indicator -->
							{#if !msg.content && msg.events.length > 0 && !msg.events.some(e => e.type === 'done' || e.type === 'error')}
								<div class="thinking">
									<span class="thinking-dot"></span>
									<span class="thinking-dot"></span>
									<span class="thinking-dot"></span>
								</div>
							{/if}
						{/if}

						{#if msg.role === 'assistant' && getErrorEvent(msg.events)}
							<div class="error-card">
								<div class="error-card-header">
									<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
									<span>Turn failed</span>
								</div>
								<div class="error-card-body">{getErrorEvent(msg.events).error || msg.content || 'An error occurred.'}</div>
							</div>
						{:else if msg.content}
							<div class="message-content">
								{#if msg.role === 'assistant'}
									{@html renderMarkdown(msg.content)}
								{:else}
									{msg.content}
								{/if}
							</div>
						{/if}

						{#if msg.role === 'assistant' && msg.metric}
							<div class="metric-badges" title="Turn timings">
								<span class="badge">LLM {fmtMs(msg.metric.llm_ms)}</span>
								<span class="badge">Tools {fmtMs(msg.metric.tool_ms_total)}</span>
								<span class="badge">Total {fmtMs(msg.metric.total_ms)}</span>
								<span class="badge">{msg.metric.iterations} iter</span>
							</div>
						{/if}
					</div>
				</div>
			{/if}
		{/each}
	</div>

	{#if micError}
		<div class="mic-error">{micError}</div>
	{/if}

	<div class="input-bar">
		<textarea
			class="chat-input"
			bind:value={inputText}
			onkeydown={handleKeyDown}
			placeholder={$isConnected
				? recording
					? 'Recording… click mic to stop'
					: transcribing
						? 'Transcribing…'
						: 'Type a message...'
				: 'Connecting...'}
			disabled={!$isConnected || recording || transcribing}
			rows="1"
		></textarea>
		<button
			class="mic-btn"
			class:recording
			onclick={toggleMic}
			disabled={!$isConnected || $isProcessing || transcribing}
			title={recording ? 'Stop recording' : 'Record voice message'}
			aria-label={recording ? 'Stop recording' : 'Record voice message'}
		>
			{#if recording}
				<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
			{:else if transcribing}
				<svg class="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
			{:else}
				<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
			{/if}
		</button>
		<button
			class="send-btn"
			onclick={handleSend}
			disabled={!$isConnected || $isProcessing || !inputText.trim()}
		>
			<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
		</button>
	</div>
</div>

<style>
	.chat-page {
		display: flex;
		flex-direction: column;
		height: calc(100vh - 48px);
		max-width: 900px;
		margin: 0 auto;
	}

	.chat-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding-bottom: 16px;
		border-bottom: 1px solid #2d3148;
		margin-bottom: 0;
		flex-shrink: 0;
	}

	.page-title {
		font-size: 20px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.connection-status {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: #f87171;
	}

	.connection-status.connected {
		color: #4ade80;
	}

	.connection-status .dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #f87171;
	}

	.connection-status.connected .dot {
		background: #4ade80;
	}

	.session-badge {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 3px 8px;
		background: #1a1d2e;
		border: 1px solid #2d3148;
		border-radius: 10px;
		color: #9ca3af;
		font-size: 11px;
	}

	.session-badge .session-id {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		color: #c9cdd5;
	}

	.session-badge .device-label {
		color: #a5b4fc;
		font-weight: 500;
	}

	.session-badge .badge-sep {
		color: #4b5563;
	}

	.clear-btn {
		padding: 6px 12px;
		background: #252a3e;
		border: 1px solid #2d3148;
		border-radius: 6px;
		color: #9ca3af;
		font-size: 12px;
		cursor: pointer;
	}

	.clear-btn:hover {
		background: #2d3148;
		color: #e1e4e8;
	}

	.messages {
		flex: 1;
		overflow-y: auto;
		padding: 20px 0;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		gap: 8px;
		color: #6b7280;
		font-size: 14px;
	}

	.empty-state .muted {
		font-size: 13px;
		color: #4b5563;
	}

	.message {
		display: flex;
		gap: 12px;
		padding: 12px 0;
	}

	.avatar {
		width: 32px;
		height: 32px;
		border-radius: 8px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 14px;
		font-weight: 600;
		flex-shrink: 0;
	}

	.user-avatar {
		background: #374151;
		color: #d1d5db;
	}

	.assistant-avatar {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
	}

	.message-body {
		flex: 1;
		min-width: 0;
	}

	.message-content {
		font-size: 14px;
		line-height: 1.6;
		color: #e1e4e8;
	}

	.message.user .message-content {
		color: #c9cdd5;
	}

	.message-content :global(p) {
		margin-bottom: 8px;
	}

	.message-content :global(p:last-child) {
		margin-bottom: 0;
	}

	.message-content :global(code) {
		background: #1a1d2e;
		padding: 2px 6px;
		border-radius: 4px;
		font-size: 13px;
	}

	.message-content :global(pre) {
		background: #0f1117;
		padding: 12px;
		border-radius: 8px;
		overflow-x: auto;
		margin: 8px 0;
	}

	.message-content :global(pre code) {
		background: none;
		padding: 0;
	}

	.metric-badges {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 6px;
	}

	.metric-badges .badge {
		background: #1a1d2e;
		color: #9ca3af;
		border: 1px solid #2d3148;
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 11px;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}

	.thinking {
		display: flex;
		gap: 4px;
		padding: 8px 0;
	}

	.thinking-dot {
		width: 8px;
		height: 8px;
		background: #6b7280;
		border-radius: 50%;
		animation: pulse 1.4s ease-in-out infinite;
	}

	.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
	.thinking-dot:nth-child(3) { animation-delay: 0.4s; }

	@keyframes pulse {
		0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
		40% { opacity: 1; transform: scale(1); }
	}

	.input-bar {
		display: flex;
		gap: 10px;
		padding: 16px 0 0;
		border-top: 1px solid #2d3148;
		flex-shrink: 0;
	}

	.chat-input {
		flex: 1;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 10px;
		padding: 12px 16px;
		color: #e1e4e8;
		font-size: 14px;
		font-family: inherit;
		resize: none;
		outline: none;
		transition: border-color 0.15s;
	}

	.chat-input:focus {
		border-color: #6366f1;
	}

	.chat-input::placeholder {
		color: #4b5563;
	}

	.chat-input:disabled {
		opacity: 0.5;
	}

	.send-btn {
		width: 44px;
		height: 44px;
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		border: none;
		border-radius: 10px;
		color: white;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: opacity 0.15s;
		flex-shrink: 0;
	}

	.send-btn:hover:not(:disabled) {
		opacity: 0.9;
	}

	.send-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.icon-btn {
		position: relative;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 28px;
		padding: 0;
		background: #252a3e;
		border: 1px solid #2d3148;
		border-radius: 6px;
		color: #9ca3af;
		cursor: pointer;
		transition: background 0.15s, color 0.15s, border-color 0.15s;
	}

	.icon-btn:hover {
		background: #2d3148;
		color: #e1e4e8;
	}

	.icon-btn.active {
		background: rgba(99, 102, 241, 0.15);
		border-color: #6366f1;
		color: #a5b4fc;
	}

	.speaking-dot {
		position: absolute;
		top: -2px;
		right: -2px;
		width: 8px;
		height: 8px;
		background: #4ade80;
		border-radius: 50%;
		box-shadow: 0 0 0 2px #161822;
		animation: pulse 1.4s ease-in-out infinite;
	}

	.mic-btn {
		width: 44px;
		height: 44px;
		background: #252a3e;
		border: 1px solid #2d3148;
		border-radius: 10px;
		color: #c9cdd5;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		transition: background 0.15s, color 0.15s, border-color 0.15s;
	}

	.mic-btn:hover:not(:disabled) {
		background: #2d3148;
		color: #e1e4e8;
	}

	.mic-btn.recording {
		background: #dc2626;
		border-color: #dc2626;
		color: white;
		animation: rec-pulse 1.2s ease-in-out infinite;
	}

	.mic-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.mic-btn .spin {
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	@keyframes rec-pulse {
		0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.5); }
		50% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
	}

	.mic-error {
		background: rgba(239, 68, 68, 0.1);
		color: #f87171;
		padding: 8px 12px;
		border-radius: 8px;
		font-size: 12px;
		margin-bottom: 8px;
	}

	.conn-banner {
		display: flex;
		align-items: center;
		gap: 10px;
		background: rgba(234, 179, 8, 0.08);
		border: 1px solid rgba(234, 179, 8, 0.3);
		color: #fbbf24;
		padding: 8px 12px;
		border-radius: 8px;
		font-size: 12px;
		margin-top: 12px;
		flex-shrink: 0;
	}

	.conn-banner.disconnected {
		background: rgba(239, 68, 68, 0.1);
		border-color: rgba(239, 68, 68, 0.3);
		color: #f87171;
	}

	.conn-banner-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: currentColor;
		animation: pulse 1.4s ease-in-out infinite;
	}

	.conn-banner-text {
		flex: 1;
	}

	.conn-banner-btn {
		background: transparent;
		border: 1px solid currentColor;
		color: inherit;
		padding: 3px 10px;
		border-radius: 6px;
		font-size: 11px;
		cursor: pointer;
	}

	.conn-banner-btn:hover {
		background: rgba(255, 255, 255, 0.05);
	}

	.error-card {
		background: rgba(239, 68, 68, 0.08);
		border: 1px solid rgba(239, 68, 68, 0.3);
		border-radius: 8px;
		padding: 10px 12px;
		margin-top: 4px;
	}

	.error-card-header {
		display: flex;
		align-items: center;
		gap: 6px;
		color: #f87171;
		font-size: 12px;
		font-weight: 600;
		margin-bottom: 4px;
	}

	.error-card-body {
		color: #fca5a5;
		font-size: 13px;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.summary-card {
		background: #1a1d2e;
		border: 1px solid #2d3148;
		border-left: 3px solid #a5b4fc;
		border-radius: 8px;
		padding: 8px 12px;
		margin: 6px 0;
		font-size: 13px;
	}

	.summary-card-header {
		display: flex;
		align-items: center;
		gap: 8px;
		cursor: pointer;
		color: #a5b4fc;
		font-weight: 600;
		list-style: none;
		user-select: none;
	}

	.summary-card-header::-webkit-details-marker {
		display: none;
	}

	.summary-card-label {
		letter-spacing: 0.04em;
		text-transform: uppercase;
		font-size: 11px;
	}

	.summary-card-hint {
		margin-left: auto;
		color: #6b7280;
		font-weight: 400;
		font-size: 11px;
		font-style: italic;
	}

	.summary-card[open] .summary-card-hint {
		display: none;
	}

	.summary-card-body {
		color: #c9cdd5;
		font-style: italic;
		line-height: 1.5;
		margin-top: 8px;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.reasoning-card {
		background: #1a1d2e;
		border: 1px solid #2d3148;
		border-left: 3px solid #c4b5fd;
		border-radius: 8px;
		padding: 8px 12px;
		margin: 6px 0;
		font-size: 13px;
	}

	.reasoning-card-header {
		display: flex;
		align-items: center;
		gap: 8px;
		cursor: pointer;
		color: #c4b5fd;
		font-weight: 600;
		list-style: none;
		user-select: none;
	}

	.reasoning-card-header::-webkit-details-marker {
		display: none;
	}

	.reasoning-card-label {
		letter-spacing: 0.04em;
		text-transform: uppercase;
		font-size: 11px;
	}

	.reasoning-card-hint {
		margin-left: auto;
		color: #6b7280;
		font-weight: 400;
		font-size: 11px;
		font-style: italic;
	}

	.reasoning-card[open] .reasoning-card-hint {
		display: none;
	}

	.reasoning-card-body {
		color: #c9cdd5;
		line-height: 1.5;
		margin-top: 8px;
		white-space: pre-wrap;
		word-break: break-word;
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		font-size: 12px;
	}
</style>
