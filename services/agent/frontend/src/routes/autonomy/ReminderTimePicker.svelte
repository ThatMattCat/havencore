<script>
	import cronstrue from 'cronstrue';

	let {
		cron = $bindable(''),
		oneShot = $bindable(false),
		timezone = 'UTC',
	} = $props();

	const PRESETS = [
		{ label: 'Every 15 minutes', cron: '*/15 * * * *' },
		{ label: 'Every hour', cron: '0 * * * *' },
		{ label: 'Daily at 8:00 AM', cron: '0 8 * * *' },
		{ label: 'Daily at 8:00 PM', cron: '0 20 * * *' },
		{ label: 'Weekdays at 9:00 AM', cron: '0 9 * * 1-5' },
		{ label: 'Weekends at 10:00 AM', cron: '0 10 * * 0,6' },
		{ label: 'First of month at midnight', cron: '0 0 1 * *' },
		{ label: 'Nightly at 3:00 AM', cron: '0 3 * * *' },
	];

	// --- Mode + tab state ---
	let mode = $state('easy'); // 'easy' | 'cron'
	let tab = $state('in'); // 'in' | 'at' | 'every'

	// --- "In…" tab ---
	let inAmount = $state(30);
	let inUnit = $state('minutes'); // 'minutes' | 'hours' | 'days'

	// --- "At…" tab ---
	function _defaultAtDate() {
		const d = new Date(Date.now() + 60 * 60 * 1000); // +1h
		return d.toISOString().slice(0, 10);
	}
	function _defaultAtTime() {
		const d = new Date(Date.now() + 60 * 60 * 1000);
		const hh = String(d.getHours()).padStart(2, '0');
		const mm = String(d.getMinutes()).padStart(2, '0');
		return `${hh}:${mm}`;
	}
	let atDate = $state(_defaultAtDate());
	let atTime = $state(_defaultAtTime());

	// --- "Every…" tab ---
	let everyFreq = $state('daily'); // 'daily' | 'weekdays' | 'weekends' | 'weekly' | 'custom'
	let everyTime = $state('08:00');
	// For weekly: single day index (0=Sun..6=Sat)
	let everyWeeklyDay = $state(1); // Mon
	// For custom: array of day indices
	let everyCustomDays = $state([1, 3, 5]); // Mon/Wed/Fri default
	const DAYS = [
		{ idx: 0, short: 'Sun' },
		{ idx: 1, short: 'Mon' },
		{ idx: 2, short: 'Tue' },
		{ idx: 3, short: 'Wed' },
		{ idx: 4, short: 'Thu' },
		{ idx: 5, short: 'Fri' },
		{ idx: 6, short: 'Sat' },
	];

	// --- Cron mode (advanced passthrough) ---
	let cronPreset = $state('');
	let cronManual = $state(cron);
	let cronManualOneShot = $state(oneShot);

	// --- Cron synthesis helpers ---

	/**
	 * Build a one-shot 5-field cron expression from a JS Date (local time).
	 * Uses minute/hour/day/month wildcards on day-of-week so the cron only
	 * matches the exact moment. Pair with one_shot=true so the engine deletes
	 * after the fire.
	 */
	function oneShotCronFromDate(d) {
		return `${d.getMinutes()} ${d.getHours()} ${d.getDate()} ${d.getMonth() + 1} *`;
	}

	function inOffsetMs() {
		const n = Math.max(1, Number(inAmount) || 0);
		if (inUnit === 'minutes') return n * 60_000;
		if (inUnit === 'hours') return n * 3_600_000;
		return n * 86_400_000;
	}

	function inFireAt() {
		const target = new Date(Date.now() + inOffsetMs());
		// Round up to next minute boundary so the cron matches
		// at-or-after the requested moment.
		if (target.getSeconds() || target.getMilliseconds()) {
			target.setSeconds(0, 0);
			target.setMinutes(target.getMinutes() + 1);
		}
		return target;
	}

	function atFireAt() {
		// "YYYY-MM-DD" + "HH:MM" interpreted as local time.
		if (!atDate || !atTime) return null;
		const [hh, mm] = atTime.split(':').map((s) => Number(s));
		const [y, mo, da] = atDate.split('-').map((s) => Number(s));
		if ([hh, mm, y, mo, da].some((n) => Number.isNaN(n))) return null;
		return new Date(y, mo - 1, da, hh, mm, 0, 0);
	}

	function everyToCron() {
		if (!everyTime) return '';
		const [hh, mm] = everyTime.split(':').map((s) => Number(s));
		if (Number.isNaN(hh) || Number.isNaN(mm)) return '';
		if (everyFreq === 'daily') return `${mm} ${hh} * * *`;
		if (everyFreq === 'weekdays') return `${mm} ${hh} * * 1-5`;
		if (everyFreq === 'weekends') return `${mm} ${hh} * * 0,6`;
		if (everyFreq === 'weekly') return `${mm} ${hh} * * ${everyWeeklyDay}`;
		if (everyFreq === 'custom') {
			if (everyCustomDays.length === 0) return '';
			const sorted = [...new Set(everyCustomDays)].sort((a, b) => a - b);
			return `${mm} ${hh} * * ${sorted.join(',')}`;
		}
		return '';
	}

	function toggleCustomDay(idx) {
		const set = new Set(everyCustomDays);
		if (set.has(idx)) set.delete(idx);
		else set.add(idx);
		everyCustomDays = [...set].sort((a, b) => a - b);
	}

	// --- Compute outputs based on active tab/mode ---
	let synthesized = $derived.by(() => {
		if (mode === 'cron') {
			return { cron: cronManual.trim(), oneShot: !!cronManualOneShot };
		}
		if (tab === 'in') {
			const fire = inFireAt();
			return { cron: oneShotCronFromDate(fire), oneShot: true, fireAt: fire };
		}
		if (tab === 'at') {
			const fire = atFireAt();
			if (!fire) return { cron: '', oneShot: true };
			if (fire.getTime() <= Date.now()) {
				return { cron: '', oneShot: true, error: 'Time must be in the future' };
			}
			return { cron: oneShotCronFromDate(fire), oneShot: true, fireAt: fire };
		}
		if (tab === 'every') {
			return { cron: everyToCron(), oneShot: false };
		}
		return { cron: '', oneShot: false };
	});

	// Push synthesized values back to bound parent props.
	$effect(() => {
		cron = synthesized.cron;
		oneShot = synthesized.oneShot;
	});

	// Keep cronManual in sync if the parent updates `cron` from outside (e.g. on edit hydration).
	$effect(() => {
		if (mode === 'cron' && cronManual !== cron) {
			cronManual = cron;
			cronManualOneShot = oneShot;
		}
	});

	function applyPreset() {
		if (cronPreset) {
			cronManual = cronPreset;
			cronManualOneShot = false;
			cronPreset = '';
		}
	}

	// --- Preview text ---
	let preview = $derived.by(() => {
		if (synthesized.error) return { text: synthesized.error, ok: false };
		if (!synthesized.cron) return { text: '', ok: false };
		if (mode === 'easy' && (tab === 'in' || tab === 'at') && synthesized.fireAt) {
			return {
				text: `Will fire ${synthesized.fireAt.toLocaleString()} (one-shot)`,
				ok: true,
			};
		}
		try {
			const human = cronstrue.toString(synthesized.cron, { throwExceptionOnParseError: true });
			return { text: human, ok: true };
		} catch (e) {
			return { text: `⚠ ${e?.message ?? 'invalid cron'}`, ok: false };
		}
	});
</script>

<div class="picker">
	<div class="mode-toggle">
		<button
			type="button"
			class="mode-btn"
			class:active={mode === 'easy'}
			onclick={() => (mode = 'easy')}
		>
			Easy
		</button>
		<button
			type="button"
			class="mode-btn"
			class:active={mode === 'cron'}
			onclick={() => (mode = 'cron')}
		>
			Cron (advanced)
		</button>
	</div>

	{#if mode === 'easy'}
		<div class="tabs">
			<button
				type="button"
				class="tab"
				class:active={tab === 'in'}
				onclick={() => (tab = 'in')}
			>
				In…
			</button>
			<button
				type="button"
				class="tab"
				class:active={tab === 'at'}
				onclick={() => (tab = 'at')}
			>
				At…
			</button>
			<button
				type="button"
				class="tab"
				class:active={tab === 'every'}
				onclick={() => (tab = 'every')}
			>
				Every…
			</button>
		</div>

		{#if tab === 'in'}
			<div class="row">
				<input
					class="input num"
					type="number"
					min="1"
					bind:value={inAmount}
				/>
				<select class="input" bind:value={inUnit}>
					<option value="minutes">minutes</option>
					<option value="hours">hours</option>
					<option value="days">days</option>
				</select>
				<small class="muted">from now</small>
			</div>
		{:else if tab === 'at'}
			<div class="row">
				<input class="input" type="date" bind:value={atDate} />
				<input class="input" type="time" bind:value={atTime} />
			</div>
		{:else if tab === 'every'}
			<div class="row wrap">
				<select class="input" bind:value={everyFreq}>
					<option value="daily">Every day</option>
					<option value="weekdays">Every weekday</option>
					<option value="weekends">Every weekend</option>
					<option value="weekly">Once a week</option>
					<option value="custom">Custom days</option>
				</select>

				{#if everyFreq === 'weekly'}
					<select class="input" bind:value={everyWeeklyDay}>
						{#each DAYS as d}
							<option value={d.idx}>{d.short}</option>
						{/each}
					</select>
				{/if}

				<input class="input" type="time" bind:value={everyTime} />
			</div>

			{#if everyFreq === 'custom'}
				<div class="day-chips">
					{#each DAYS as d}
						<button
							type="button"
							class="chip"
							class:on={everyCustomDays.includes(d.idx)}
							onclick={() => toggleCustomDay(d.idx)}
						>
							{d.short}
						</button>
					{/each}
				</div>
			{/if}
		{/if}
	{:else}
		<div class="cron-row">
			<input
				class="input mono cron-input"
				type="text"
				bind:value={cronManual}
				placeholder="0 8 * * *"
			/>
			<select
				class="input preset-select"
				bind:value={cronPreset}
				onchange={applyPreset}
				aria-label="Insert cron preset"
			>
				<option value="">Presets…</option>
				{#each PRESETS as p}
					<option value={p.cron}>{p.label}</option>
				{/each}
			</select>
		</div>
		<label class="checkbox-field">
			<input type="checkbox" bind:checked={cronManualOneShot} />
			<span>One-shot (delete after fire)</span>
		</label>
	{/if}

	{#if preview.text}
		<small class="muted preview" class:valid={preview.ok} class:invalid={!preview.ok}>
			{preview.ok ? '✓' : ''}
			{preview.text}
		</small>
	{/if}
	<small class="muted">
		Times interpreted in <code>{timezone}</code> (from <code>CURRENT_TIMEZONE</code>).
	</small>
</div>

<style>
	.picker {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.mode-toggle {
		display: inline-flex;
		gap: 0;
		border: 1px solid #2d3148;
		border-radius: 6px;
		overflow: hidden;
		width: fit-content;
	}
	.mode-btn {
		background: #161822;
		color: #c9cdd5;
		border: none;
		padding: 5px 12px;
		font-size: 12px;
		cursor: pointer;
	}
	.mode-btn.active {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
	}
	.tabs {
		display: flex;
		gap: 4px;
		margin-top: 4px;
	}
	.tab {
		background: #161822;
		color: #c9cdd5;
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 5px 12px;
		font-size: 12px;
		cursor: pointer;
	}
	.tab.active {
		background: #252a3e;
		border-color: #6366f1;
		color: #e1e4e8;
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: nowrap;
	}
	.row.wrap {
		flex-wrap: wrap;
	}
	.input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 6px;
		padding: 6px 10px;
		font-size: 13px;
	}
	.input.num {
		width: 80px;
	}
	.input.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.cron-row {
		display: flex;
		gap: 8px;
	}
	.cron-input {
		flex: 1;
	}
	.preset-select {
		flex-shrink: 0;
		width: auto;
	}
	.day-chips {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	.chip {
		background: #161822;
		color: #6b7280;
		border: 1px solid #2d3148;
		border-radius: 999px;
		padding: 4px 12px;
		font-size: 12px;
		cursor: pointer;
	}
	.chip.on {
		background: rgba(99, 102, 241, 0.15);
		color: #a78bfa;
		border-color: #6366f1;
	}
	.checkbox-field {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 13px;
		color: #c9cdd5;
		margin-top: 4px;
	}
	.muted {
		color: #6b7280;
		font-size: 11px;
	}
	.preview.valid {
		color: #34d399;
	}
	.preview.invalid {
		color: #f87171;
	}
	code {
		background: #161822;
		padding: 1px 4px;
		border-radius: 3px;
		font-size: 11px;
	}
</style>
