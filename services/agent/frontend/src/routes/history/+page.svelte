<script>
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import Card from '$lib/components/Card.svelte';
	import { listConversations, getConversation, resumeConversation } from '$lib/api';
	import { setSessionId } from '$lib/stores/chat';

	let conversations = $state([]);
	let selectedConv = $state(null);
	let selectedMessages = $state(null);
	let loading = $state(true);
	let loadingDetail = $state(false);
	let error = $state('');
	let offset = $state(0);
	let atEnd = $state(false);
	const limit = 20;

	onMount(async () => {
		await loadConversations();
	});

	async function loadConversations() {
		loading = true;
		try {
			const data = await listConversations(limit, offset);
			conversations = data.conversations;
			atEnd = conversations.length < limit;
		} catch (e) {
			error = e.message;
		}
		loading = false;
	}

	async function selectConversation(conv) {
		selectedConv = conv;
		selectedMessages = null;
		loadingDetail = true;
		try {
			const data = await getConversation(conv.session_id);
			selectedMessages = data.conversation;
		} catch (e) {
			error = e.message;
		}
		loadingDetail = false;
	}

	function formatTime(iso) {
		if (!iso) return '';
		return new Date(iso).toLocaleString();
	}

	async function nextPage() {
		if (atEnd) return;
		const prevOffset = offset;
		offset += limit;
		await loadConversations();
		if (conversations.length === 0) {
			// Rolled past the end — restore prior page and mark as end.
			offset = prevOffset;
			atEnd = true;
			await loadConversations();
			atEnd = true;
		}
	}

	function prevPage() {
		offset = Math.max(0, offset - limit);
		loadConversations();
	}

	let resumingSid = $state(null);

	async function handleResume(conv, event) {
		// Prevent the row's selectConversation click.
		if (event) event.stopPropagation();
		resumingSid = conv.session_id;
		try {
			const result = await resumeConversation(conv.session_id);
			setSessionId(result.session_id);
			await goto('/chat');
		} catch (e) {
			error = `Resume failed: ${e.message || e}`;
		} finally {
			resumingSid = null;
		}
	}

	function getMessagePreview(messages) {
		if (!messages || messages.length === 0) return 'Empty conversation';
		const userMsg = messages.find(m => m.role === 'user');
		if (userMsg) {
			const content = typeof userMsg.content === 'string' ? userMsg.content : '';
			// Strip the system context wrapper
			const match = content.match(/### User Message\n([\s\S]*)/);
			const text = match ? match[1].trim() : content.trim();
			return text.length > 80 ? text.slice(0, 80) + '...' : text;
		}
		return 'No user message';
	}
</script>

<div class="history-page">
	<h1 class="page-title">Conversation History</h1>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<div class="history-layout">
		<div class="conversation-list-panel">
			{#if loading}
				<p class="muted">Loading...</p>
			{:else if conversations.length === 0}
				<p class="muted">No conversations stored yet</p>
			{:else}
				{#each conversations as conv}
					<div class="conv-row" class:selected={selectedConv?.session_id === conv.session_id}>
						<button
							class="conv-item"
							onclick={() => selectConversation(conv)}
						>
							<div class="conv-time">{formatTime(conv.created_at)}</div>
							<div class="conv-info">{conv.message_count} messages</div>
						</button>
						<button
							class="resume-btn"
							onclick={(e) => handleResume(conv, e)}
							disabled={resumingSid === conv.session_id}
							title="Resume this conversation in /chat"
						>
							{resumingSid === conv.session_id ? '…' : 'Resume'}
						</button>
					</div>
				{/each}

				<div class="pagination">
					<button onclick={prevPage} disabled={offset === 0}>Previous</button>
					<span class="page-info">
						Page {Math.floor(offset / limit) + 1}
						{#if atEnd}<span class="end-hint">· end</span>{/if}
					</span>
					<button onclick={nextPage} disabled={atEnd}>Next</button>
				</div>
			{/if}
		</div>

		<div class="conversation-detail-panel">
			{#if !selectedConv}
				<div class="empty-detail">
					<p class="muted">Select a conversation to view details</p>
				</div>
			{:else if loadingDetail}
				<p class="muted">Loading conversation...</p>
			{:else if selectedMessages}
				<div class="detail-header">
					<h2>{formatTime(selectedConv.created_at)}</h2>
					<span class="muted">{selectedConv.message_count} messages</span>
				</div>
				{#each selectedMessages as history}
					<div class="message-list">
						{#each history.messages as msg}
							{#if msg.role !== 'system'}
								<div class="msg" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'} class:tool={msg.role === 'tool'}>
									<span class="msg-role">{msg.role}</span>
									<div class="msg-content">
										{#if typeof msg.content === 'string'}
											{msg.content}
										{:else}
											<pre>{JSON.stringify(msg.content, null, 2)}</pre>
										{/if}
									</div>
								</div>
							{/if}
						{/each}
					</div>
				{/each}
			{/if}
		</div>
	</div>
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

	.history-layout {
		display: flex;
		gap: 16px;
		height: calc(100vh - 120px);
	}

	.conversation-list-panel {
		width: 300px;
		flex-shrink: 0;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.conv-row {
		display: flex;
		gap: 6px;
		align-items: stretch;
	}

	.conv-row .conv-item {
		flex: 1;
	}

	.conv-row.selected .conv-item {
		background: #252a3e;
		border-color: #6366f1;
	}

	.conv-item {
		display: block;
		text-align: left;
		padding: 12px;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 8px;
		color: #c9cdd5;
		cursor: pointer;
		transition: all 0.15s;
	}

	.conv-item:hover {
		background: #1e2235;
	}

	.resume-btn {
		padding: 0 12px;
		background: #1e2235;
		border: 1px solid #2d3148;
		border-radius: 8px;
		color: #a5b4fc;
		font-size: 12px;
		cursor: pointer;
		transition: background 0.15s, border-color 0.15s;
	}

	.resume-btn:hover:not(:disabled) {
		background: #252a3e;
		border-color: #6366f1;
	}

	.resume-btn:disabled {
		opacity: 0.5;
		cursor: wait;
	}

	.conv-time {
		font-size: 13px;
		color: #e1e4e8;
		margin-bottom: 4px;
	}

	.conv-info {
		font-size: 12px;
		color: #6b7280;
	}

	.pagination {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 12px 0;
		gap: 8px;
	}

	.pagination button {
		padding: 6px 12px;
		background: #252a3e;
		border: 1px solid #2d3148;
		border-radius: 6px;
		color: #9ca3af;
		font-size: 12px;
		cursor: pointer;
	}

	.pagination button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.page-info {
		font-size: 12px;
		color: #6b7280;
	}

	.end-hint {
		color: #4b5563;
		margin-left: 4px;
	}

	.conversation-detail-panel {
		flex: 1;
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		padding: 16px;
		overflow-y: auto;
	}

	.empty-detail {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
	}

	.detail-header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 16px;
		padding-bottom: 12px;
		border-bottom: 1px solid #2d3148;
	}

	.detail-header h2 {
		font-size: 16px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.message-list {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.msg {
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 13px;
	}

	.msg.user {
		background: #1e2235;
	}

	.msg.assistant {
		background: #1a1d2e;
	}

	.msg.tool {
		background: #1a2218;
		border-left: 3px solid #4ade80;
	}

	.msg-role {
		display: block;
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		color: #6b7280;
		margin-bottom: 4px;
	}

	.msg.user .msg-role { color: #818cf8; }
	.msg.assistant .msg-role { color: #a78bfa; }
	.msg.tool .msg-role { color: #4ade80; }

	.msg-content {
		color: #c9cdd5;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.msg-content pre {
		background: #0f1117;
		padding: 8px;
		border-radius: 6px;
		font-size: 11px;
		overflow-x: auto;
	}
</style>
