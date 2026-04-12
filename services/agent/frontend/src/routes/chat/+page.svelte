<script>
	import { onMount, onDestroy, tick } from 'svelte';
	import { messages, isConnected, isProcessing, connect, sendMessage, disconnect, clearMessages } from '$lib/stores/chat';
	import ToolCallCard from '$lib/components/ToolCallCard.svelte';
	import { marked } from 'marked';

	let inputText = $state('');
	let messagesContainer = $state(null);

	// Configure marked for safe rendering
	marked.setOptions({ breaks: true, gfm: true });

	onMount(() => {
		connect();
	});

	onDestroy(() => {
		disconnect();
	});

	function handleSend() {
		if (!inputText.trim() || $isProcessing) return;
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
</script>

<div class="chat-page">
	<div class="chat-header">
		<h1 class="page-title">Chat with Selene</h1>
		<div class="header-actions">
			<span class="connection-status" class:connected={$isConnected}>
				<span class="dot"></span>
				{$isConnected ? 'Connected' : 'Disconnected'}
			</span>
			<button class="clear-btn" onclick={clearMessages}>Clear</button>
		</div>
	</div>

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

						<!-- Show thinking indicator -->
						{#if !msg.content && msg.events.length > 0 && !msg.events.some(e => e.type === 'done' || e.type === 'error')}
							<div class="thinking">
								<span class="thinking-dot"></span>
								<span class="thinking-dot"></span>
								<span class="thinking-dot"></span>
							</div>
						{/if}
					{/if}

					{#if msg.content}
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
		{/each}
	</div>

	<div class="input-bar">
		<textarea
			class="chat-input"
			bind:value={inputText}
			onkeydown={handleKeyDown}
			placeholder={$isConnected ? 'Type a message...' : 'Connecting...'}
			disabled={!$isConnected}
			rows="1"
		></textarea>
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
</style>
