<script>
	let { children } = $props();

	const navItems = [
		{ href: '/', label: 'Dashboard', icon: 'grid' },
		{ href: '/chat', label: 'Chat', icon: 'message' },
		{ href: '/playgrounds', label: 'Playgrounds', icon: 'beaker' },
		{ href: '/metrics', label: 'Metrics', icon: 'activity' },
		{ href: '/devices', label: 'Devices', icon: 'cpu' },
		{ href: '/people', label: 'People', icon: 'users' },
		{ href: '/cameras', label: 'Cameras', icon: 'camera' },
		{ href: '/memory', label: 'Memory', icon: 'brain' },
		{ href: '/autonomy', label: 'Autonomy', icon: 'zap' },
		{ href: '/history', label: 'History', icon: 'clock' },
		{ href: '/system', label: 'System', icon: 'settings' },
	];

	import { page } from '$app/state';
</script>

<div class="app">
	<nav class="sidebar">
		<div class="logo">
			<span class="logo-mark">H</span>
			<span class="logo-text">HavenCore</span>
		</div>
		<ul class="nav-list">
			{#each navItems as item}
				<li>
					<a
						href={item.href}
						class="nav-link"
						class:active={page.url.pathname === item.href ||
							(item.href !== '/' && page.url.pathname.startsWith(item.href))}
					>
						<span class="nav-icon">{@html getIcon(item.icon)}</span>
						<span>{item.label}</span>
					</a>
				</li>
			{/each}
		</ul>
		<div class="sidebar-footer">
			<span class="version">Selene Agent v1.0</span>
		</div>
	</nav>
	<main class="content">
		{@render children()}
	</main>
</div>

<script module>
	function getIcon(name) {
		const icons = {
			grid: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
			message: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>',
			cpu: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
			clock: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
			settings: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
			beaker: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3v6l-5 9a2 2 0 001.75 3h12.5A2 2 0 0020 18l-5-9V3"/><line x1="8" y1="3" x2="16" y2="3"/><line x1="7" y1="14" x2="17" y2="14"/></svg>',
			activity: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
			brain: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>',
			zap: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
			users: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
			camera: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>',
		};
		return icons[name] || '';
	}
</script>

<style>
	:global(*) {
		margin: 0;
		padding: 0;
		box-sizing: border-box;
	}

	:global(body) {
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
		background: #0f1117;
		color: #e1e4e8;
		overflow: hidden;
		height: 100vh;
	}

	:global(a) {
		color: inherit;
		text-decoration: none;
	}

	.app {
		display: flex;
		height: 100vh;
	}

	.sidebar {
		width: 220px;
		background: #161822;
		border-right: 1px solid #2d3148;
		display: flex;
		flex-direction: column;
		flex-shrink: 0;
	}

	.logo {
		padding: 20px 16px;
		display: flex;
		align-items: center;
		gap: 10px;
		border-bottom: 1px solid #2d3148;
	}

	.logo-mark {
		width: 32px;
		height: 32px;
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		border-radius: 8px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-weight: 700;
		font-size: 18px;
		color: white;
	}

	.logo-text {
		font-size: 16px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.nav-list {
		list-style: none;
		padding: 12px 8px;
		flex: 1;
	}

	.nav-link {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 12px;
		border-radius: 8px;
		font-size: 14px;
		color: #9ca3af;
		transition: all 0.15s ease;
	}

	.nav-link:hover {
		background: #1e2235;
		color: #e1e4e8;
	}

	.nav-link.active {
		background: #252a3e;
		color: #a78bfa;
	}

	.nav-icon {
		display: flex;
		align-items: center;
	}

	.sidebar-footer {
		padding: 12px 16px;
		border-top: 1px solid #2d3148;
	}

	.version {
		font-size: 11px;
		color: #555;
	}

	.content {
		flex: 1;
		overflow-y: auto;
		padding: 24px;
	}
</style>
