"""Tests for autonomy.reminder_personalize.personalize_reminder()."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_client(content: str | None) -> AsyncMock:
    """Build a mock matching openai-style `client.chat.completions.create`."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    resp = SimpleNamespace(choices=[choice])
    create = AsyncMock(return_value=resp)
    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


@pytest.mark.asyncio
async def test_personalize_returns_rewritten_body_and_image_for_signal():
    from selene_agent.autonomy import reminder_personalize

    client = _make_client(
        '{"body": "Hey, time to take the trash out — bins go to the curb.", '
        '"image_prompt": "stylized trash bin at curb at sunset"}'
    )
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
    )

    assert out["body"].startswith("Hey")
    assert out["image_prompt"] == "stylized trash bin at curb at sunset"


@pytest.mark.asyncio
async def test_personalize_strips_image_prompt_for_non_signal_channels():
    from selene_agent.autonomy import reminder_personalize

    # Even if the model misbehaves and includes an image_prompt for ha_push,
    # the helper strips it — non-signal channels can't render attachments.
    client = _make_client(
        '{"body": "Don\'t forget the trash.", "image_prompt": "trash bin"}'
    )
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="ha_push",
    )

    assert out["body"] == "Don't forget the trash."
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_handles_explicit_null_image_prompt():
    from selene_agent.autonomy import reminder_personalize

    client = _make_client('{"body": "Take the trash out tonight.", "image_prompt": null}')
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
    )

    assert out["body"] == "Take the trash out tonight."
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_extracts_json_from_noisy_output():
    from selene_agent.autonomy import reminder_personalize

    # Some models prepend prose before the JSON despite instructions.
    client = _make_client(
        'Sure! Here is the rewrite:\n'
        '{"body": "Time to switch the laundry.", "image_prompt": null}\n'
        'Let me know if you want a different tone.'
    )
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Laundry",
        body="Switch laundry to dryer",
        channel="ha_push",
    )

    assert out["body"] == "Time to switch the laundry."


@pytest.mark.asyncio
async def test_personalize_falls_back_on_unparseable_output():
    from selene_agent.autonomy import reminder_personalize

    client = _make_client("totally not json")
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
    )

    assert out["body"] == "Take the trash out"  # original
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_falls_back_on_empty_body_in_response():
    from selene_agent.autonomy import reminder_personalize

    client = _make_client('{"body": "   ", "image_prompt": null}')
    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
    )

    assert out["body"] == "Take the trash out"
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_falls_back_on_llm_exception():
    from selene_agent.autonomy import reminder_personalize

    create = AsyncMock(side_effect=RuntimeError("model unreachable"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
    )

    assert out["body"] == "Take the trash out"
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_falls_back_on_timeout():
    from selene_agent.autonomy import reminder_personalize

    async def _slow(*a, **kw):
        await asyncio.sleep(5)
        raise RuntimeError("should not reach")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_slow)))

    out = await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="Trash",
        body="Take the trash out",
        channel="signal",
        timeout_sec=0.1,
    )

    assert out["body"] == "Take the trash out"
    assert out["image_prompt"] is None


@pytest.mark.asyncio
async def test_personalize_system_prompt_mentions_channel():
    """The system prompt must encode the channel so the LLM knows whether
    image_prompt is allowed. We assert by inspecting the call args."""
    from selene_agent.autonomy import reminder_personalize

    client = _make_client('{"body": "ok", "image_prompt": null}')
    await reminder_personalize.personalize_reminder(
        client=client,
        model_name="gpt-3.5-turbo",
        title="x",
        body="y",
        channel="signal",
    )
    create_call = client.chat.completions.create.await_args
    messages = create_call.kwargs["messages"]
    system_text = messages[0]["content"]
    assert "signal" in system_text.lower() or "image_prompt" in system_text

    # And for ha_push: the prompt must instruct null image_prompt.
    client2 = _make_client('{"body": "ok", "image_prompt": null}')
    await reminder_personalize.personalize_reminder(
        client=client2,
        model_name="gpt-3.5-turbo",
        title="x",
        body="y",
        channel="ha_push",
    )
    sys2 = client2.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "null" in sys2.lower() or "not delivered" in sys2.lower()
