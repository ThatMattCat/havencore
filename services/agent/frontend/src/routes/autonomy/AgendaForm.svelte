<script>
	let { item = null, onclose, onsaved } = $props();

	const isEdit = !!item;

	// --- Common fields ---
	let kind = $state(item?.kind ?? 'reminder');
	let name = $state(item?.name ?? '');
	let autonomyLevel = $state(item?.autonomy_level ?? 'notify');
	let enabled = $state(item?.enabled ?? true);

	// --- Trigger ---
	// triggerMode = 'cron' | 'mqtt' | 'webhook'
	let triggerMode = $state(
		item?.trigger_spec?.source === 'mqtt'
			? 'mqtt'
			: item?.trigger_spec?.source === 'webhook'
				? 'webhook'
				: 'cron',
	);
	let cron = $state(item?.schedule_cron ?? '');
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

	// --- Quiet hours ---
	let quietStart = $state(item?.config?.quiet_hours?.start ?? '');
	let quietEnd = $state(item?.config?.quiet_hours?.end ?? '');
	let quietPolicy = $state(item?.config?.quiet_hours?.policy ?? 'defer');

	// --- Event rate limit ---
	let eventRateLimit = $state(item?.config?.event_rate_limit ?? '');

	let saving = $state(false);
	let error = $state('');

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
			if (routineDeliverChannel || routineDeliverTo) {
				cfg.deliver = {};
				if (routineDeliverChannel) cfg.deliver.channel = routineDeliverChannel;
				if (routineDeliverTo) cfg.deliver.to = routineDeliverTo;
			}
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
			// strip nulls so the backend defaults kick in cleanly
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

				<label class="field">
					<span>Autonomy level</span>
					<select class="input" bind:value={autonomyLevel}>
						<option value="observe">observe</option>
						<option value="notify">notify</option>
					</select>
				</label>
			</div>

			<!-- Kind selector (disabled when editing, since backend doesn't allow changing kind) -->
			<div class="field">
				<span>Kind</span>
				<div class="radio-row">
					{#each ['reminder', 'watch', 'routine', 'briefing', 'anomaly_sweep', 'memory_review'] as k}
						<label class="radio">
							<input type="radio" bind:group={kind} value={k} disabled={isEdit} />
							<span>{k}</span>
						</label>
					{/each}
				</div>
			</div>

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
				<label class="field">
					<span>Cron expression</span>
					<input
						class="input mono"
						type="text"
						bind:value={cron}
						placeholder="0 8 * * *"
					/>
					<small class="muted">5-field cron. Interpreted in {`"America/Chicago"`} (config.CURRENT_TIMEZONE).</small>
				</label>
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
							</select>
						</label>
						<label class="field">
							<span>Deliver to</span>
							<input class="input" type="text" bind:value={routineDeliverTo} placeholder="optional override" />
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
			<details class="json-preview" open>
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
