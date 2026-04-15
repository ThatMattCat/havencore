<script>
	import cronstrue from 'cronstrue';

	let { item = null, timezone = 'UTC', onclose, onsaved } = $props();

	const isEdit = !!item;

	// --- Reference data (surfaced as help content) ---
	const AUTONOMY_LEVELS = [
		{
			value: 'observe',
			label: 'observe',
			desc: 'Read-only. Memory search, web, weather, Wikipedia, Wolfram, HA state queries. No notifications, no device control. Safe for pure analysis.',
		},
		{
			value: 'notify',
			label: 'notify',
			desc: 'Everything in observe + push / Signal notifications. Default tier — fits reminders, briefings, watches, and anomaly reports.',
		},
		{
			value: 'speak',
			label: 'speak',
			desc: 'Same tools as notify, but delivery is TTS (Kokoro) routed to a Music Assistant speaker instead of a push message.',
		},
		{
			value: 'act',
			label: 'act',
			desc: 'Everything in notify + bounded HA device control (lights, switches, scenes, scripts, climate). Required for kind=act. Gated per-item by action_allow_list and AUTONOMY_ACT_ENABLED=true.',
		},
	];

	const KINDS = [
		{
			value: 'reminder',
			label: 'reminder',
			triggerHint: 'cron',
			desc: 'Fires a plain notification at a scheduled time. No LLM. Use for fixed reminders ("take meds at 8am").',
		},
		{
			value: 'watch',
			label: 'watch',
			triggerHint: 'mqtt or webhook',
			desc: 'Listens for an event (MQTT / webhook) and sends a template-rendered notification. No LLM. Placeholders: {_topic} {_name} {_source} {_ts} plus any payload key.',
		},
		{
			value: 'watch_llm',
			label: 'watch_llm',
			triggerHint: 'mqtt or webhook',
			desc: 'Reactive trigger → gathers HA entity state + memories → LLM decides if this is worth surfacing and at what severity. Has signature-based cooldown to prevent spam.',
		},
		{
			value: 'routine',
			label: 'routine',
			triggerHint: 'cron or event',
			desc: 'LLM writes the output. Good for briefings, daily summaries, contextual announcements. Optional tools_override must be a subset of the autonomy tier.',
		},
		{
			value: 'act',
			label: 'act',
			triggerHint: 'cron or event',
			desc: 'Two-phase device control: LLM plans JSON actions → engine validates against allow-list → optional user confirmation → execute. Requires autonomy_level=act and AUTONOMY_ACT_ENABLED=true.',
		},
		{
			value: 'briefing',
			label: 'briefing',
			triggerHint: 'cron',
			desc: 'Built-in morning summary (calendar, weather, history). Usually system-seeded via AUTONOMY_BRIEFING_CRON — rarely created manually.',
		},
		{
			value: 'anomaly_sweep',
			label: 'anomaly_sweep',
			triggerHint: 'cron',
			desc: 'Built-in: snapshots HA state and asks the LLM if anything looks abnormal. Per-signature cooldowns. Usually system-seeded.',
		},
		{
			value: 'memory_review',
			label: 'memory_review',
			triggerHint: 'cron',
			desc: 'Built-in nightly L1–L4 memory consolidation job. Importance decay, tier promotion. System-seeded.',
		},
	];

	const CRON_PRESETS = [
		{ label: 'Every 15 minutes', cron: '*/15 * * * *' },
		{ label: 'Every hour', cron: '0 * * * *' },
		{ label: 'Daily at 8:00 AM', cron: '0 8 * * *' },
		{ label: 'Daily at 8:00 PM', cron: '0 20 * * *' },
		{ label: 'Weekdays at 9:00 AM', cron: '0 9 * * 1-5' },
		{ label: 'Weekends at 10:00 AM', cron: '0 10 * * 0,6' },
		{ label: 'First of month at midnight', cron: '0 0 1 * *' },
		{ label: 'Nightly at 3:00 AM', cron: '0 3 * * *' },
	];

	// --- Common fields ---
	let kind = $state(item?.kind ?? 'reminder');
	let name = $state(item?.name ?? '');
	let autonomyLevel = $state(item?.autonomy_level ?? 'notify');
	let enabled = $state(item?.enabled ?? true);

	// --- Help panel toggles ---
	let showAutonomyHelp = $state(false);
	let showKindHelp = $state(false);

	// --- Trigger ---
	let triggerMode = $state(
		item?.trigger_spec?.source === 'mqtt'
			? 'mqtt'
			: item?.trigger_spec?.source === 'webhook'
				? 'webhook'
				: 'cron',
	);
	let cron = $state(item?.schedule_cron ?? '');
	let cronPreset = $state('');
	let mqttTopic = $state(item?.trigger_spec?.match?.topic ?? '');
	let mqttPayloadJson = $state(
		JSON.stringify(item?.trigger_spec?.match?.payload ?? {}, null, 2),
	);
	let webhookName = $state(item?.trigger_spec?.match?.name ?? '');

	// --- Reminder config ---
	let reminderTitle = $state(item?.config?.title ?? '');
	let reminderBody = $state(item?.config?.body ?? '');
	let reminderChannel = $state(item?.config?.channel ?? 'ha_push');
	let reminderTo = $state(item?.config?.to ?? '');
	let reminderOneShot = $state(item?.config?.one_shot ?? false);

	// --- Watch config ---
	let watchBodyTemplate = $state(item?.config?.body_template ?? '');
	let watchChannel = $state(item?.config?.channel ?? 'ha_push');
	let watchTo = $state(item?.config?.to ?? '');
	let watchSeverity = $state(item?.config?.severity ?? 'info');
	let watchConditionEntity = $state(item?.config?.condition?.entity_id ?? '');
	let watchConditionState = $state(item?.config?.condition?.state ?? '');
	let watchConditionMinDuration = $state(item?.config?.condition?.min_duration_sec ?? '');

	// --- Routine config ---
	let routinePrompt = $state(item?.config?.prompt ?? '');
	let routineToolsOverride = $state((item?.config?.tools_override ?? []).join(', '));
	let routineDeliverChannel = $state(item?.config?.deliver?.channel ?? 'ha_push');
	let routineDeliverTo = $state(item?.config?.deliver?.to ?? '');
	let routineDeliverDevice = $state(item?.config?.deliver?.device ?? '');
	let routineDeliverVoice = $state(item?.config?.deliver?.voice ?? '');
	let routineDeliverVolume = $state(item?.config?.deliver?.volume ?? '');

	// --- watch_llm config ---
	let watchLlmSubject = $state(item?.config?.subject ?? '');
	let watchLlmEntities = $state((item?.config?.gather?.entities ?? []).join(', '));
	let watchLlmMemoriesK = $state(item?.config?.gather?.memories_k ?? 3);
	let watchLlmSeverityFloor = $state(item?.config?.severity_floor ?? 'low');
	let watchLlmCooldown = $state(item?.config?.cooldown_min ?? 15);
	let watchLlmNotifyChannel = $state(item?.config?.notify?.channel ?? 'signal');
	let watchLlmNotifyTo = $state(item?.config?.notify?.to ?? '');
	let watchLlmNotifyDevice = $state(item?.config?.notify?.device ?? '');
	let watchLlmNotifyVoice = $state(item?.config?.notify?.voice ?? '');
	let watchLlmNotifyVolume = $state(item?.config?.notify?.volume ?? '');

	// --- act config ---
	let actPrompt = $state(item?.config?.prompt ?? '');
	let actAllowList = $state((item?.config?.action_allow_list ?? []).join(', '));
	let actRequireConfirmation = $state(item?.config?.require_confirmation ?? true);
	let actConfirmTimeout = $state(item?.config?.confirmation_timeout_sec ?? 300);
	let actStrictExecute = $state(item?.config?.strict_execute ?? false);
	let actDeliverChannel = $state(item?.config?.deliver?.channel ?? 'signal');
	let actDeliverTo = $state(item?.config?.deliver?.to ?? '');

	// --- Quiet hours ---
	let quietStart = $state(item?.config?.quiet_hours?.start ?? '');
	let quietEnd = $state(item?.config?.quiet_hours?.end ?? '');
	let quietPolicy = $state(item?.config?.quiet_hours?.policy ?? 'defer');

	// --- Event rate limit ---
	let eventRateLimit = $state(item?.config?.event_rate_limit ?? '');

	let saving = $state(false);
	let error = $state('');

	// --- Derived helpers ---
	let selectedKind = $derived(KINDS.find((k) => k.value === kind) ?? KINDS[0]);
	let selectedLevel = $derived(
		AUTONOMY_LEVELS.find((l) => l.value === autonomyLevel) ?? AUTONOMY_LEVELS[1],
	);

	let cronHuman = $derived.by(() => {
		if (!cron || !cron.trim()) return '';
		try {
			return cronstrue.toString(cron.trim(), { throwExceptionOnParseError: true });
		} catch (e) {
			return `⚠ ${e?.message ?? 'invalid cron expression'}`;
		}
	});
	let cronValid = $derived(!!cron && !cronHuman.startsWith('⚠'));

	// Warn when kind=act but level!=act (backend will reject).
	let levelKindMismatch = $derived(kind === 'act' && autonomyLevel !== 'act');

	function applyPreset() {
		if (cronPreset) {
			cron = cronPreset;
			cronPreset = '';
		}
	}

	function parseJSONSafe(s) {
		if (!s || !s.trim()) return {};
		try {
			return JSON.parse(s);
		} catch {
			return null;
		}
	}

	let triggerSpec = $derived.by(() => {
		if (triggerMode === 'cron') return null;
		if (triggerMode === 'mqtt') {
			const match = { topic: mqttTopic };
			const payload = parseJSONSafe(mqttPayloadJson);
			if (payload && Object.keys(payload).length > 0) match.payload = payload;
			return { source: 'mqtt', match };
		}
		if (triggerMode === 'webhook') {
			return { source: 'webhook', match: { name: webhookName } };
		}
		return null;
	});

	let configObj = $derived.by(() => {
		const cfg = {};
		if (kind === 'reminder') {
			if (reminderTitle) cfg.title = reminderTitle;
			if (reminderBody) cfg.body = reminderBody;
			if (reminderChannel) cfg.channel = reminderChannel;
			if (reminderTo) cfg.to = reminderTo;
			if (reminderOneShot) cfg.one_shot = true;
		} else if (kind === 'watch') {
			if (watchBodyTemplate) cfg.body_template = watchBodyTemplate;
			if (watchChannel) cfg.channel = watchChannel;
			if (watchTo) cfg.to = watchTo;
			if (watchSeverity) cfg.severity = watchSeverity;
			if (watchConditionEntity) {
				cfg.condition = { entity_id: watchConditionEntity };
				if (watchConditionState) cfg.condition.state = watchConditionState;
				if (watchConditionMinDuration)
					cfg.condition.min_duration_sec = Number(watchConditionMinDuration);
			}
		} else if (kind === 'routine') {
			if (routinePrompt) cfg.prompt = routinePrompt;
			const toolsList = routineToolsOverride
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
			if (toolsList.length > 0) cfg.tools_override = toolsList;
			cfg.deliver = {};
			if (routineDeliverChannel) cfg.deliver.channel = routineDeliverChannel;
			if (routineDeliverTo) cfg.deliver.to = routineDeliverTo;
			if (routineDeliverChannel === 'speaker') {
				if (routineDeliverDevice) cfg.deliver.device = routineDeliverDevice;
				if (routineDeliverVoice) cfg.deliver.voice = routineDeliverVoice;
				if (routineDeliverVolume !== '' && routineDeliverVolume != null)
					cfg.deliver.volume = Number(routineDeliverVolume);
			}
			if (Object.keys(cfg.deliver).length === 0) delete cfg.deliver;
		} else if (kind === 'watch_llm') {
			if (watchLlmSubject) cfg.subject = watchLlmSubject;
			const entList = watchLlmEntities
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
			const gather = {};
			if (entList.length > 0) gather.entities = entList;
			if (watchLlmMemoriesK !== '' && watchLlmMemoriesK != null)
				gather.memories_k = Number(watchLlmMemoriesK);
			if (Object.keys(gather).length > 0) cfg.gather = gather;
			if (watchLlmSeverityFloor) cfg.severity_floor = watchLlmSeverityFloor;
			if (watchLlmCooldown !== '' && watchLlmCooldown != null)
				cfg.cooldown_min = Number(watchLlmCooldown);
			cfg.notify = {};
			if (watchLlmNotifyChannel) cfg.notify.channel = watchLlmNotifyChannel;
			if (watchLlmNotifyTo) cfg.notify.to = watchLlmNotifyTo;
			if (watchLlmNotifyChannel === 'speaker') {
				if (watchLlmNotifyDevice) cfg.notify.device = watchLlmNotifyDevice;
				if (watchLlmNotifyVoice) cfg.notify.voice = watchLlmNotifyVoice;
				if (watchLlmNotifyVolume !== '' && watchLlmNotifyVolume != null)
					cfg.notify.volume = Number(watchLlmNotifyVolume);
			}
			if (Object.keys(cfg.notify).length === 0) delete cfg.notify;
		} else if (kind === 'act') {
			if (actPrompt) cfg.prompt = actPrompt;
			const allow = actAllowList
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
			if (allow.length > 0) cfg.action_allow_list = allow;
			cfg.require_confirmation = !!actRequireConfirmation;
			if (actConfirmTimeout !== '' && actConfirmTimeout != null)
				cfg.confirmation_timeout_sec = Number(actConfirmTimeout);
			cfg.strict_execute = !!actStrictExecute;
			cfg.deliver = {};
			if (actDeliverChannel) cfg.deliver.channel = actDeliverChannel;
			if (actDeliverTo) cfg.deliver.to = actDeliverTo;
			if (Object.keys(cfg.deliver).length === 0) delete cfg.deliver;
		}
		if (quietStart || quietEnd) {
			cfg.quiet_hours = { policy: quietPolicy };
			if (quietStart) cfg.quiet_hours.start = quietStart;
			if (quietEnd) cfg.quiet_hours.end = quietEnd;
		}
		if (eventRateLimit) cfg.event_rate_limit = eventRateLimit;
		return cfg;
	});

	let payload = $derived({
		kind,
		name: name || null,
		schedule_cron: triggerMode === 'cron' ? cron || null : null,
		trigger_spec: triggerSpec,
		config: configObj,
		autonomy_level: autonomyLevel,
		enabled,
	});

	let payloadPreview = $derived(JSON.stringify(payload, null, 2));

	async function save() {
		error = '';
		saving = true;
		try {
			const body = { ...payload };
			if (body.schedule_cron === null) delete body.schedule_cron;
			if (body.trigger_spec === null) delete body.trigger_spec;
			if (body.name === null) delete body.name;

			const url = isEdit ? `/api/autonomy/items/${item.id}` : '/api/autonomy/items';
			const method = isEdit ? 'PATCH' : 'POST';
			const r = await fetch(url, {
				method,
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
			});
			if (!r.ok) {
				const text = await r.text();
				throw new Error(`${r.status}: ${text}`);
			}
			onsaved?.();
		} catch (e) {
			error = e.message;
		} finally {
			saving = false;
		}
	}
</script>

<div class="modal-backdrop" onclick={onclose} role="presentation">
	<div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
		<div class="modal-header">
			<h2>{isEdit ? 'Edit' : 'Create'} agenda item</h2>
			<button class="close-btn" onclick={onclose}>×</button>
		</div>

		<div class="modal-body">
			{#if error}
				<div class="error">{error}</div>
			{/if}

			<div class="form-grid">
				<label class="field">
					<span>Name</span>
					<input class="input" type="text" bind:value={name} placeholder="optional label" />
				</label>

				<div class="field">
					<span class="label-row">
						Autonomy level
						<button
							type="button"
							class="info-btn"
							aria-label="Explain autonomy levels"
							onclick={() => (showAutonomyHelp = !showAutonomyHelp)}
						>
							{showAutonomyHelp ? '×' : '?'}
						</button>
					</span>
					<select class="input" bind:value={autonomyLevel}>
						{#each AUTONOMY_LEVELS as lvl}
							<option value={lvl.value}>{lvl.label}</option>
						{/each}
					</select>
					<small class="muted hint">{selectedLevel.desc}</small>
				</div>
			</div>

			{#if showAutonomyHelp}
				<div class="help-panel">
					<div class="help-title">Autonomy levels — what each one unlocks</div>
					<p class="help-lede">
						Autonomy level sets the <strong>tool ceiling</strong> available to the LLM when this
						item fires. The form UI stays the same across levels — the difference is enforced
						server-side at tool-call time (see <code>autonomy/tool_gating.py</code>). Lower levels
						just refuse calls to tools outside their tier.
					</p>
					<ul class="help-list">
						{#each AUTONOMY_LEVELS as lvl}
							<li>
								<code>{lvl.value}</code> — {lvl.desc}
							</li>
						{/each}
					</ul>
					<p class="help-foot">
						Note: <code>kind=act</code> requires <code>autonomy_level=act</code>. Other kinds accept
						any level, but a lower level restricts what the LLM can actually do.
					</p>
				</div>
			{/if}

			{#if levelKindMismatch}
				<div class="warning">
					<code>kind=act</code> requires <code>autonomy_level=act</code> — the backend will reject
					this.
				</div>
			{/if}

			<!-- Kind selector (disabled when editing, since backend doesn't allow changing kind) -->
			<div class="field">
				<span class="label-row">
					Kind
					<button
						type="button"
						class="info-btn"
						aria-label="Explain all kinds"
						onclick={() => (showKindHelp = !showKindHelp)}
					>
						{showKindHelp ? '×' : '?'}
					</button>
				</span>
				<div class="radio-row">
					{#each KINDS as k}
						<label class="radio">
							<input type="radio" bind:group={kind} value={k.value} disabled={isEdit} />
							<span>{k.label}</span>
						</label>
					{/each}
				</div>
				<small class="muted hint">
					<strong>{selectedKind.label}</strong> — {selectedKind.desc}
					<span class="tag">trigger: {selectedKind.triggerHint}</span>
				</small>
			</div>

			{#if showKindHelp}
				<div class="help-panel">
					<div class="help-title">Kinds — what each one does</div>
					<ul class="help-list">
						{#each KINDS as k}
							<li>
								<code>{k.value}</code>
								<span class="tag">{k.triggerHint}</span>
								— {k.desc}
							</li>
						{/each}
					</ul>
				</div>
			{/if}

			<!-- Trigger selector -->
			<div class="field">
				<span>Trigger</span>
				<div class="radio-row">
					<label class="radio">
						<input type="radio" bind:group={triggerMode} value="cron" />
						<span>cron</span>
					</label>
					<label class="radio">
						<input type="radio" bind:group={triggerMode} value="mqtt" />
						<span>mqtt</span>
					</label>
					<label class="radio">
						<input type="radio" bind:group={triggerMode} value="webhook" />
						<span>webhook</span>
					</label>
				</div>
			</div>

			{#if triggerMode === 'cron'}
				<div class="field">
					<span>Cron expression</span>
					<div class="cron-row">
						<input
							class="input mono cron-input"
							type="text"
							bind:value={cron}
							placeholder="0 8 * * *"
						/>
						<select
							class="input preset-select"
							bind:value={cronPreset}
							onchange={applyPreset}
							aria-label="Insert cron preset"
						>
							<option value="">Presets…</option>
							{#each CRON_PRESETS as p}
								<option value={p.cron}>{p.label}</option>
							{/each}
						</select>
					</div>
					{#if cron}
						<small class="muted" class:valid={cronValid} class:invalid={!cronValid}>
							{cronValid ? '✓' : ''}
							{cronHuman}
						</small>
					{/if}
					<small class="muted">
						5-field cron. Interpreted in <code>{timezone}</code> (from
						<code>CURRENT_TIMEZONE</code> env).
					</small>
				</div>
			{:else if triggerMode === 'mqtt'}
				<label class="field">
					<span>Topic (supports + wildcard)</span>
					<input
						class="input mono"
						type="text"
						bind:value={mqttTopic}
						placeholder="home/sensor/+/state"
					/>
				</label>
				<label class="field">
					<span>Payload match (JSON, subset)</span>
					<textarea
						class="input mono"
						rows="4"
						bind:value={mqttPayloadJson}
						placeholder={'{}'}
					></textarea>
					<small class="muted">Keys must all match incoming payload values. {'{}'} matches any.</small>
				</label>
			{:else if triggerMode === 'webhook'}
				<label class="field">
					<span>Webhook name (URL path segment)</span>
					<input
						class="input mono"
						type="text"
						bind:value={webhookName}
						placeholder="front-door-open"
					/>
					<small class="muted">
						POST JSON to <code>/api/autonomy/webhook/&lt;name&gt;</code>
					</small>
				</label>
			{/if}

			<!-- Kind-scoped config -->
			{#if kind === 'reminder'}
				<fieldset class="fieldset">
					<legend>Reminder config</legend>
					<label class="field">
						<span>Title</span>
						<input class="input" type="text" bind:value={reminderTitle} />
					</label>
					<label class="field">
						<span>Body</span>
						<textarea class="input" rows="3" bind:value={reminderBody}></textarea>
					</label>
					<div class="form-grid">
						<label class="field">
							<span>Channel</span>
							<select class="input" bind:value={reminderChannel}>
								<option value="ha_push">ha_push</option>
								<option value="signal">signal</option>
							</select>
						</label>
						<label class="field">
							<span>To</span>
							<input class="input" type="text" bind:value={reminderTo} placeholder="optional override" />
						</label>
					</div>
					<label class="checkbox-field">
						<input type="checkbox" bind:checked={reminderOneShot} />
						<span>One-shot (disable after fire)</span>
					</label>
				</fieldset>
			{:else if kind === 'watch'}
				<fieldset class="fieldset">
					<legend>Watch config</legend>
					<label class="field">
						<span>Body template</span>
						<textarea
							class="input mono"
							rows="3"
							bind:value={watchBodyTemplate}
							placeholder={'Front door {state} at {_ts}'}
						></textarea>
						<small class="muted">
							{`{_topic} {_name} {_source} {_ts}`} + any payload key.
						</small>
					</label>
					<div class="form-grid">
						<label class="field">
							<span>Channel</span>
							<select class="input" bind:value={watchChannel}>
								<option value="ha_push">ha_push</option>
								<option value="signal">signal</option>
							</select>
						</label>
						<label class="field">
							<span>To</span>
							<input class="input" type="text" bind:value={watchTo} placeholder="optional override" />
						</label>
						<label class="field">
							<span>Severity</span>
							<select class="input" bind:value={watchSeverity}>
								<option value="info">info</option>
								<option value="warn">warn</option>
								<option value="critical">critical</option>
							</select>
						</label>
					</div>
					<div class="form-grid">
						<label class="field">
							<span>Condition entity (optional)</span>
							<input class="input mono" type="text" bind:value={watchConditionEntity} placeholder="binary_sensor.front_door" />
						</label>
						<label class="field">
							<span>Required state</span>
							<input class="input mono" type="text" bind:value={watchConditionState} placeholder="on" />
						</label>
						<label class="field">
							<span>Min duration (sec)</span>
							<input class="input" type="number" bind:value={watchConditionMinDuration} placeholder="300" />
						</label>
					</div>
				</fieldset>
			{:else if kind === 'routine'}
				<fieldset class="fieldset">
					<legend>Routine config</legend>
					<label class="field">
						<span>Prompt (what the LLM should produce)</span>
						<textarea class="input" rows="3" bind:value={routinePrompt}></textarea>
					</label>
					<label class="field">
						<span>Tools override (comma-separated, must subset tier)</span>
						<input class="input mono" type="text" bind:value={routineToolsOverride} placeholder="search_memories,ha_notify" />
					</label>
					<div class="form-grid">
						<label class="field">
							<span>Deliver channel</span>
							<select class="input" bind:value={routineDeliverChannel}>
								<option value="ha_push">ha_push</option>
								<option value="signal">signal</option>
								<option value="speaker">speaker</option>
							</select>
						</label>
						<label class="field">
							<span>Deliver to</span>
							<input class="input" type="text" bind:value={routineDeliverTo} placeholder="optional override" />
						</label>
					</div>
					{#if routineDeliverChannel === 'speaker'}
						<div class="form-grid">
							<label class="field">
								<span>MA player</span>
								<input class="input" type="text" bind:value={routineDeliverDevice} placeholder="Living Room" />
							</label>
							<label class="field">
								<span>Voice</span>
								<input class="input" type="text" bind:value={routineDeliverVoice} placeholder="af_heart" />
							</label>
							<label class="field">
								<span>Volume (0–1)</span>
								<input class="input" type="number" step="0.05" min="0" max="1" bind:value={routineDeliverVolume} placeholder="0.5" />
							</label>
						</div>
					{/if}
				</fieldset>
			{:else if kind === 'watch_llm'}
				<fieldset class="fieldset">
					<legend>watch_llm config</legend>
					<label class="field">
						<span>Subject (short label used in LLM prompt + signature)</span>
						<input class="input" type="text" bind:value={watchLlmSubject} placeholder="front-door-after-midnight" />
					</label>
					<label class="field">
						<span>Gather entities (comma-separated)</span>
						<input class="input mono" type="text" bind:value={watchLlmEntities} placeholder="binary_sensor.front_door, person.matt" />
					</label>
					<div class="form-grid">
						<label class="field">
							<span>Memories k</span>
							<input class="input" type="number" min="0" bind:value={watchLlmMemoriesK} />
						</label>
						<label class="field">
							<span>Severity floor</span>
							<select class="input" bind:value={watchLlmSeverityFloor}>
								<option value="low">low</option>
								<option value="med">med</option>
								<option value="high">high</option>
							</select>
						</label>
						<label class="field">
							<span>Cooldown (min)</span>
							<input class="input" type="number" min="0" bind:value={watchLlmCooldown} />
						</label>
					</div>
					<div class="form-grid">
						<label class="field">
							<span>Notify channel</span>
							<select class="input" bind:value={watchLlmNotifyChannel}>
								<option value="signal">signal</option>
								<option value="ha_push">ha_push</option>
								<option value="speaker">speaker</option>
							</select>
						</label>
						<label class="field">
							<span>Notify to</span>
							<input class="input" type="text" bind:value={watchLlmNotifyTo} placeholder="optional override" />
						</label>
					</div>
					{#if watchLlmNotifyChannel === 'speaker'}
						<div class="form-grid">
							<label class="field">
								<span>MA player</span>
								<input class="input" type="text" bind:value={watchLlmNotifyDevice} placeholder="Living Room" />
							</label>
							<label class="field">
								<span>Voice</span>
								<input class="input" type="text" bind:value={watchLlmNotifyVoice} placeholder="af_heart" />
							</label>
							<label class="field">
								<span>Volume (0–1)</span>
								<input class="input" type="number" step="0.05" min="0" max="1" bind:value={watchLlmNotifyVolume} placeholder="0.5" />
							</label>
						</div>
					{/if}
				</fieldset>
			{:else if kind === 'act'}
				<fieldset class="fieldset">
					<legend>act config — requires <code>AUTONOMY_ACT_ENABLED=true</code></legend>
					<label class="field">
						<span>Prompt (goal for the planner)</span>
						<textarea class="input" rows="3" bind:value={actPrompt} placeholder="Turn on the living room lamp."></textarea>
					</label>
					<label class="field">
						<span>Action allow-list (comma-separated tool names)</span>
						<input class="input mono" type="text" bind:value={actAllowList} placeholder="ha_control_light, ha_activate_scene" />
						<small class="muted">Only tools in this list will be executed. Empty = item rejected.</small>
					</label>
					<div class="form-grid">
						<label class="checkbox-field">
							<input type="checkbox" bind:checked={actRequireConfirmation} />
							<span>Require confirmation</span>
						</label>
						<label class="field">
							<span>Confirmation timeout (sec)</span>
							<input class="input" type="number" min="10" bind:value={actConfirmTimeout} />
						</label>
						<label class="checkbox-field">
							<input type="checkbox" bind:checked={actStrictExecute} />
							<span>Strict execute (any error → status=error)</span>
						</label>
					</div>
					<div class="form-grid">
						<label class="field">
							<span>Confirmation channel</span>
							<select class="input" bind:value={actDeliverChannel}>
								<option value="signal">signal</option>
								<option value="ha_push">ha_push</option>
							</select>
						</label>
						<label class="field">
							<span>Deliver to</span>
							<input class="input" type="text" bind:value={actDeliverTo} placeholder="optional override" />
						</label>
					</div>
				</fieldset>
			{/if}

			<fieldset class="fieldset">
				<legend>Quiet hours (optional)</legend>
				<div class="form-grid">
					<label class="field">
						<span>Start (HH:MM)</span>
						<input class="input mono" type="text" bind:value={quietStart} placeholder="22:00" />
					</label>
					<label class="field">
						<span>End (HH:MM)</span>
						<input class="input mono" type="text" bind:value={quietEnd} placeholder="07:00" />
					</label>
					<label class="field">
						<span>Policy</span>
						<select class="input" bind:value={quietPolicy}>
							<option value="defer">defer</option>
							<option value="drop">drop</option>
						</select>
					</label>
				</div>
			</fieldset>

			<label class="field">
				<span>Event rate limit (optional)</span>
				<input class="input mono" type="text" bind:value={eventRateLimit} placeholder="10/min" />
				<small class="muted">For reactive triggers. Formats: N/sec, N/min, N/hr.</small>
			</label>

			<label class="checkbox-field">
				<input type="checkbox" bind:checked={enabled} />
				<span>Enabled</span>
			</label>

			<!-- Live JSON preview -->
			<details class="json-preview">
				<summary>JSON preview</summary>
				<pre class="mono">{payloadPreview}</pre>
			</details>
		</div>

		<div class="modal-footer">
			<button class="btn ghost" onclick={onclose} disabled={saving}>Cancel</button>
			<button class="btn primary" onclick={save} disabled={saving}>
				{saving ? 'Saving…' : isEdit ? 'Save changes' : 'Create'}
			</button>
		</div>
	</div>
</div>

<style>
	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.6);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 100;
		padding: 20px;
	}

	.modal {
		background: #161822;
		border: 1px solid #2d3148;
		border-radius: 12px;
		width: 100%;
		max-width: 720px;
		max-height: 90vh;
		display: flex;
		flex-direction: column;
	}

	.modal-header {
		padding: 16px 20px;
		display: flex;
		justify-content: space-between;
		align-items: center;
		border-bottom: 1px solid #2d3148;
	}

	.modal-header h2 {
		font-size: 17px;
		font-weight: 600;
		color: #f0f0f0;
	}

	.close-btn {
		background: none;
		border: none;
		color: #9ca3af;
		font-size: 22px;
		cursor: pointer;
	}

	.modal-body {
		padding: 20px;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.modal-footer {
		padding: 14px 20px;
		border-top: 1px solid #2d3148;
		display: flex;
		justify-content: flex-end;
		gap: 10px;
	}

	.form-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 10px;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 13px;
	}

	.field > span {
		color: #c9cdd5;
		font-size: 12px;
		font-weight: 500;
	}

	.label-row {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}

	.info-btn {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: #252a3e;
		border: 1px solid #2d3148;
		color: #a78bfa;
		font-size: 11px;
		font-weight: 700;
		line-height: 1;
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0;
	}

	.info-btn:hover {
		background: #2d3148;
		border-color: #6366f1;
	}

	.checkbox-field {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 13px;
		color: #c9cdd5;
		cursor: pointer;
	}

	.input {
		background: #0f1117;
		border: 1px solid #2d3148;
		color: #e1e4e8;
		border-radius: 6px;
		padding: 7px 10px;
		font-size: 13px;
		font-family: inherit;
		width: 100%;
	}

	.input:focus {
		outline: none;
		border-color: #6366f1;
	}

	.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 12px;
	}

	.muted {
		color: #6b7280;
		font-size: 11px;
	}

	.hint {
		color: #9ca3af;
		line-height: 1.4;
	}

	.valid {
		color: #34d399;
	}

	.invalid {
		color: #f87171;
	}

	.tag {
		display: inline-block;
		margin-left: 6px;
		padding: 1px 6px;
		background: #252a3e;
		border-radius: 4px;
		color: #a78bfa;
		font-size: 10px;
		font-weight: 500;
	}

	.fieldset {
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 12px 14px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.fieldset legend {
		padding: 0 6px;
		color: #a78bfa;
		font-size: 12px;
		font-weight: 600;
	}

	.radio-row {
		display: flex;
		gap: 14px;
		flex-wrap: wrap;
	}

	.radio {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: 13px;
		color: #c9cdd5;
		cursor: pointer;
	}

	.help-panel {
		background: #0b0d14;
		border: 1px solid #3a2f5a;
		border-radius: 8px;
		padding: 12px 14px;
		font-size: 12px;
		color: #c9cdd5;
	}

	.help-title {
		color: #a78bfa;
		font-weight: 600;
		margin-bottom: 6px;
	}

	.help-lede {
		margin: 0 0 8px 0;
		line-height: 1.5;
		color: #9ca3af;
	}

	.help-list {
		margin: 0;
		padding-left: 18px;
		display: flex;
		flex-direction: column;
		gap: 5px;
		line-height: 1.45;
	}

	.help-list code {
		color: #fbbf24;
	}

	.help-foot {
		margin: 10px 0 0 0;
		padding-top: 8px;
		border-top: 1px solid #2d3148;
		color: #9ca3af;
		line-height: 1.45;
	}

	.warning {
		background: rgba(251, 191, 36, 0.08);
		border: 1px solid #78350f;
		color: #fbbf24;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 12px;
	}

	.cron-row {
		display: flex;
		gap: 8px;
	}

	.cron-input {
		flex: 1;
	}

	.preset-select {
		flex: 0 0 auto;
		width: auto;
		min-width: 140px;
	}

	.json-preview {
		background: #0b0d14;
		border: 1px solid #2d3148;
		border-radius: 8px;
		padding: 8px 12px;
	}

	.json-preview summary {
		cursor: pointer;
		color: #a78bfa;
		font-size: 12px;
		font-weight: 500;
		user-select: none;
	}

	.json-preview pre {
		margin-top: 8px;
		color: #c9cdd5;
		font-size: 11px;
		max-height: 240px;
		overflow: auto;
		white-space: pre;
	}

	.error {
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid #7f1d1d;
		color: #f87171;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 13px;
	}

	.btn {
		border: 1px solid #2d3148;
		border-radius: 6px;
		padding: 7px 14px;
		font-size: 13px;
		cursor: pointer;
		font-weight: 500;
	}

	.btn.primary {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		color: white;
		border-color: transparent;
	}

	.btn.primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn.ghost {
		background: #252a3e;
		color: #c9cdd5;
	}

	.btn.ghost:hover:not(:disabled) {
		background: #2d3148;
	}
</style>
