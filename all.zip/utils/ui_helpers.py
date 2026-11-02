from markdown_it import MarkdownIt

# Raw HTML is escaped (html=False), images are disabled (no remote-URL
# exfiltration via injected markdown), and markdown-it already blocks
# javascript: links.
_md = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table").disable("image")


def render_markdown_safely(text: str) -> str:
    return _md.render(text or "")


def chat_bubble_html(role: str, content: str) -> str:
    is_user = role == "user"
    avatar = "👤 You" if is_user else "🤖 Assistant"
    avatar_cls = "user-avatar" if is_user else "assistant-avatar"
    bubble_cls = "user" if is_user else "assistant"
    body = render_markdown_safely(content)
    return (
        f'<div class="chat-bubble {bubble_cls}">'
        f'<div class="chat-avatar {avatar_cls}">{avatar}</div>'
        f"<div>{body}</div></div>"
    )
