import asyncio

from fasthtml.common import Span

from dialeng.services.settings_service import SettingDefinition


def setup(api, ctx):
    api.register_setting(SettingDefinition(
        "greeting", "str", "Hello", label="Greeting",
        description="Greeting shown by the reference command.",
        group="Hello Workspace", order=10,
    ))
    api.register_setting(SettingDefinition(
        "show_status", "bool", True, label="Show status",
        description="Show the compact invocation counter.",
        group="Hello Workspace", order=20,
    ))

    async def say_hello(command_context):
        count = int(ctx.state.get("invocations", 0)) + 1
        ctx.state["invocations"] = count
        greeting = ctx.setting("greeting", "Hello")
        return {"message": f"{greeting} from Hello Workspace ({count})"}

    async def notebook_greeting(nb_id: str, **kwargs):
        return {
            "notebook_id": nb_id,
            "greeting": ctx.setting("greeting", "Hello"),
            "invocations": int(ctx.state.get("invocations", 0)),
        }

    api.register_command(
        "say-hello", title="Say hello", description="Show the configured greeting.",
        handler=say_hello,
    )
    api.register_command(
        "kernel-only", title="Kernel only",
        description="Demonstrate a command that requires a running kernel.",
        availability="kernel.required",
        handler=lambda _context: {"message": "The kernel-required command ran."},
    )

    async def slow_greeting(command_context):
        for index in range(100):
            if command_context["cancel_event"].is_set():
                return {"message": "Slow greeting cancelled."}
            if index % 10 == 0:
                await command_context["report_progress"]("Preparing greeting", current=index, total=100)
            await asyncio.sleep(0.05)
        return {"message": "Slow greeting completed."}

    api.register_command(
        "slow-greeting", title="Slow greeting",
        description="Demonstrate progress and cancellation.",
        handler=slow_greeting,
    )
    api.register_action(
        "greeting", notebook_greeting,
        response_schema={
            "type": "object",
            "required": ["notebook_id", "greeting", "invocations"],
            "properties": {
                "notebook_id": {"type": "string"},
                "greeting": {"type": "string"},
                "invocations": {"type": "integer"},
            },
        },
    )
    api.register_toolbar_item(
        "hello", command="say-hello", title="Hello", icon="hand", order=10,
    )
    api.register_shortcut(
        "hello-shortcut", "ctrl+alt+h", "say-hello",
        description="Run the Hello Workspace greeting.",
    )
    api.register_status_item(
        "invocations",
        lambda _context: Span(
            f"Hello x{int(ctx.state.get('invocations', 0))}",
            cls="hello-workspace-status",
        ) if ctx.setting("show_status", True) else Span(),
        order=10,
    )
