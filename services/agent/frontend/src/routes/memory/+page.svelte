<script lang="ts">
  import { onMount } from 'svelte';

  type L4Entry = {
    id: string;
    text: string;
    importance: number;
    importance_effective: number;
    timestamp: string;
    tags: string[];
    pending_l4_approval: boolean;
    proposed_at?: string;
    proposal_rationale?: string;
    source_ids: string[];
  };

  type Stats = {
    l2_count: number; l3_count: number; l4_count: number;
    pending_proposals: number; l4_est_tokens: number;
  };

  let stats: Stats | null = null;
  let l4: L4Entry[] = [];
  let proposals: L4Entry[] = [];
  let runs: any[] = [];
  let loading = true;
  let error = '';
  let lastRunTime: string | null = null;
  let triggering = false;
  let newL4Text = '';
  let newL4Importance = 5;

  async function refresh() {
    loading = true;
    error = '';
    try {
      const [s, l, p, r] = await Promise.all([
        fetch('/api/memory/stats').then(r => r.json()),
        fetch('/api/memory/l4').then(r => r.json()),
        fetch('/api/memory/l4/proposals').then(r => r.json()),
        fetch('/api/memory/runs?limit=20').then(r => r.json()),
      ]);
      stats = s;
      l4 = l.entries ?? [];
      proposals = p.proposals ?? [];
      runs = r.runs ?? [];
      lastRunTime = runs.length ? runs[0].triggered_at : null;
    } catch (e: any) {
      error = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  async function approve(id: string) {
    await fetch(`/api/memory/l4/proposals/${id}/approve`, { method: 'POST' });
    await refresh();
  }
  async function reject(id: string) {
    await fetch(`/api/memory/l4/proposals/${id}/reject`, { method: 'POST' });
    await refresh();
  }
  async function deleteL4(id: string) {
    if (!confirm('Remove this L4 entry? It will be demoted to L3, not deleted.')) return;
    await fetch(`/api/memory/l4/${id}`, { method: 'DELETE' });
    await refresh();
  }
  async function addL4() {
    if (!newL4Text.trim()) return;
    await fetch('/api/memory/l4', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: newL4Text, importance: newL4Importance, tags: [] }),
    });
    newL4Text = '';
    await refresh();
  }
  async function runNow() {
    triggering = true;
    try {
      await fetch('/api/memory/runs/trigger', { method: 'POST' });
      // Poll for completion.
      for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 3000));
        await refresh();
        if (runs[0]?.triggered_at !== lastRunTime) break;
      }
    } finally {
      triggering = false;
    }
  }

  onMount(refresh);
</script>

<svelte:head><title>Memory — Selene</title></svelte:head>

<div class="p-6 max-w-5xl mx-auto">
  <h1 class="text-2xl font-bold mb-4">Memory</h1>

  {#if error}
    <div class="mb-4 p-3 bg-red-900/30 border border-red-700 rounded">{error}</div>
  {/if}

  <!-- Header stats -->
  <section class="mb-6 grid grid-cols-5 gap-4 text-sm">
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L2 episodic</div>
      <div class="text-xl">{stats?.l2_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L3 consolidated</div>
      <div class="text-xl">{stats?.l3_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L4 persistent</div>
      <div class="text-xl">{stats?.l4_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">Pending proposals</div>
      <div class="text-xl {proposals.length ? 'text-amber-400' : ''}">{stats?.pending_proposals ?? 0}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L4 ~tokens</div>
      <div class="text-xl {stats && stats.l4_est_tokens > 1500 ? 'text-red-400' : ''}">
        {stats?.l4_est_tokens ?? 0}
      </div>
    </div>
  </section>

  <div class="mb-6 flex items-center gap-3">
    <button
      class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50"
      on:click={runNow}
      disabled={triggering}
    >{triggering ? 'Running…' : 'Run consolidation now'}</button>
    <span class="text-gray-400 text-sm">
      Last run: {lastRunTime ?? 'never'}
    </span>
  </div>

  <!-- L4 section -->
  <section class="mb-8">
    <h2 class="text-xl font-semibold mb-3">L4 — Persistent context</h2>

    <div class="mb-4 p-3 bg-gray-800 rounded flex gap-2 items-start">
      <textarea
        class="flex-1 bg-gray-900 rounded p-2"
        rows="2"
        placeholder="Add an L4 entry (injected into every system prompt)"
        bind:value={newL4Text}
      ></textarea>
      <div class="flex flex-col gap-2">
        <label class="text-xs text-gray-400">Importance
          <input type="number" min="1" max="5" bind:value={newL4Importance}
                 class="w-16 bg-gray-900 rounded px-2 py-1" />
        </label>
        <button class="px-3 py-1 bg-green-600 hover:bg-green-500 rounded"
                on:click={addL4}>Add</button>
      </div>
    </div>

    {#if l4.length === 0}
      <div class="text-gray-400 italic">No L4 entries yet.</div>
    {:else}
      <table class="w-full text-sm">
        <thead class="text-left text-gray-400 border-b border-gray-700">
          <tr><th class="py-2">Text</th><th>Importance</th><th>Age</th><th></th></tr>
        </thead>
        <tbody>
          {#each l4 as e}
            <tr class="border-b border-gray-800">
              <td class="py-2">{e.text}</td>
              <td>{e.importance}</td>
              <td>{e.timestamp?.slice(0, 10)}</td>
              <td class="text-right">
                <button class="text-red-400 hover:text-red-200"
                        on:click={() => deleteL4(e.id)}>Remove</button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    <h3 class="text-lg font-semibold mt-6 mb-2">Pending proposals</h3>
    {#if proposals.length === 0}
      <div class="text-gray-400 italic">No pending proposals.</div>
    {:else}
      <div class="space-y-2">
        {#each proposals as p}
          <div class="p-3 bg-gray-800 rounded">
            <div class="font-medium">{p.text}</div>
            {#if p.proposal_rationale}
              <div class="text-sm text-gray-400 italic mt-1">{p.proposal_rationale}</div>
            {/if}
            <div class="text-xs text-gray-500 mt-1">
              sources: {p.source_ids.length} · importance: {p.importance}
            </div>
            <div class="mt-2 flex gap-2">
              <button class="px-3 py-1 bg-green-600 hover:bg-green-500 rounded"
                      on:click={() => approve(p.id)}>Approve</button>
              <button class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded"
                      on:click={() => reject(p.id)}>Reject</button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </section>
</div>
