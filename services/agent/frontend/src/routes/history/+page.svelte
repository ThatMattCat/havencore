<script>
	import { onMount } from 'svelte';
	import Card from '$lib/components/Card.svelte';
	import { listConversations, getConversation } from '$lib/api';

	let conversations = $state([]);
	let selectedConv = $state(null);
	let selectedMessages = $state(null);
	let loading = $state(true);
	let loadingDetail = $state(false);
	let error = $state('');
	let offset = $state(0);
	const limit = 20;

	onMount(async () => {
		await loadConversations();
	});

	async function loadConversations() {
		loading = true;
		try {
			const data = await listConversations(limit, offset);
			conversations = data.conversations;
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

	function nextPage() {
		offset += limit;
		loadConversations();
	}

	function prevPage() {
		offset = Math.max(0, offset - limit);
		loadConversations();
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
					<button
						class="conv-item"
						class:selected={selectedConv?.session_id === conv.session_id}
						onclick={() => selectConversation(conv)}
					>
						<div class="conv-time">{formatTime(conv.created_at)}</div>
						<div class="conv-info">{conv.message_count} messages</div>
					</button>
				{/each}

				<div class="pagination">
					<button onclick={prevPage} disabled={offset === 0}>Previous</button>
					<span class="page-info">Page {Math.floor(offset / limit) + 1}</span>
					<button onclick={nextPage} disabled={conversations.length < limit}>Next</button>
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

	.conv-item {
		display: block;
		width: 100%;
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

	.conv-item.selected {
		background: #252a3e;
		border-color: #6366f1;
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
