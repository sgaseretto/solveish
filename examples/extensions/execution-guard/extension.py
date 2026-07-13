import asyncio

from fasthtml.common import Span

from dialeng.services.settings_service import SettingDefinition


async def setup(api, ctx):
    api.register_setting(SettingDefinition(
        "blocked_marker", "str", "# BLOCK", label="Blocked source marker",
        description="Code containing this marker never reaches the kernel.",
        group="Execution Guard", order=10,
    ))
    api.register_setting(SettingDefinition(
        "transform_enabled", "bool", False, label="Enable source transform",
        description="Prepend a harmless reference comment before execution.",
        group="Execution Guard", order=20,
    ))

    def guard(payload):
        marker = ctx.setting("blocked_marker", "# BLOCK")
        if marker and marker in str(payload.get("source", "")):
            return {"block": True, "reason": f"Execution Guard blocked marker: {marker}"}

    def transform(payload):
        if not ctx.setting("transform_enabled", False):
            return payload
        return {**payload, "source": "# transformed by execution-guard\n" + payload["source"]}

    def output_observer(_payload):
        ctx.state["outputs"] = int(ctx.state.get("outputs", 0)) + 1

    def saved(_payload):
        ctx.state["saves"] = int(ctx.state.get("saves", 0)) + 1

    api.on("execution.before_start", guard, name="block-marker", mode="guard", order=10)
    api.on("execution.before_start", transform, name="prepend-comment", mode="transform", order=20)
    api.on("execution.output", output_observer, name="count-output", order=50)
    api.on("notebook.saved", saved, name="count-save", order=50)

    async def heartbeat():
        while True:
            ctx.state["heartbeats"] = int(ctx.state.get("heartbeats", 0)) + 1
            await asyncio.sleep(1)

    ctx.create_task(heartbeat(), name="heartbeat")
    api.register_status_item(
        "metrics",
        lambda _context: Span(
            f"Guard outputs {int(ctx.state.get('outputs', 0))}",
            cls="execution-guard-status",
        ),
    )
