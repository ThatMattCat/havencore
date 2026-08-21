/**
 * Markdown rendering for untrusted content.
 *
 * Assistant replies are LLM output that routinely embeds text the agent has no
 * control over: web-search snippets, GitHub issue bodies, Signal messages,
 * camera/vision descriptions, memory documents. `marked` deliberately passes
 * raw inline HTML straight through (it has had no `sanitize` option since v7),
 * so `{@html marked.parse(...)}` would execute anything the model echoes back —
 * in the dashboard origin, which has unauthenticated access to every agent API.
 *
 * Every `{@html}` site that shows model- or user-produced content MUST use
 * `renderMarkdown()` from this module. Do not call `marked.parse()` directly.
 */

import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });

/** Tags ordinary markdown output needs. Anything else is dropped. */
const ALLOWED_TAGS = [
	'a',
	'b',
	'blockquote',
	'br',
	'code',
	'del',
	'em',
	'h1',
	'h2',
	'h3',
	'h4',
	'h5',
	'h6',
	'hr',
	'i',
	'img',
	'ins',
	'li',
	'ol',
	'p',
	'pre',
	's',
	'span',
	'strong',
	'sub',
	'sup',
	'table',
	'tbody',
	'td',
	'tfoot',
	'th',
	'thead',
	'tr',
	'ul'
];

/**
 * Presentational attributes only. No `on*` handlers, no `srcdoc`, no `style`.
 * DOMPurify additionally restricts `href`/`src` to safe URI schemes, which is
 * what neutralises `javascript:` links.
 */
const ALLOWED_ATTR = ['align', 'alt', 'class', 'colspan', 'href', 'rowspan', 'src', 'start', 'title'];

/**
 * Belt-and-braces denylist. The allowlists above already exclude these; listing
 * them explicitly means a future widening of ALLOWED_TAGS/ALLOWED_ATTR cannot
 * silently reintroduce script execution.
 */
const FORBID_TAGS = [
	'button',
	'embed',
	'form',
	'iframe',
	'input',
	'link',
	'math',
	'meta',
	'object',
	'option',
	'script',
	'select',
	'style',
	'svg',
	'textarea'
];
const FORBID_ATTR = ['formaction', 'srcdoc', 'style', 'xlink:href'];

const PURIFY_CONFIG = {
	ALLOWED_TAGS,
	ALLOWED_ATTR,
	FORBID_TAGS,
	FORBID_ATTR,
	ALLOW_DATA_ATTR: false,
	ALLOW_ARIA_ATTR: false,
	ALLOW_UNKNOWN_PROTOCOLS: false
	// Deliberately no USE_PROFILES: it *replaces* ALLOWED_TAGS/ALLOWED_ATTR with
	// DOMPurify's much broader built-in HTML profile instead of intersecting
	// with them, which would quietly widen the allowlist above.
};

function escapeHtml(text) {
	return String(text)
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

/**
 * Parse markdown and strip anything executable from the result.
 *
 * @param {string} text raw markdown, assumed untrusted
 * @returns {string} HTML safe to hand to `{@html}`
 */
export function renderMarkdown(text) {
	if (!text) return '';
	const html = marked.parse(text);
	// DOMPurify needs a DOM. Under SSR/prerender there isn't one and its
	// `sanitize()` degrades to a pass-through, so fall back to escaping
	// instead — degraded rendering is preferable to unsanitised HTML.
	if (!DOMPurify.isSupported) return escapeHtml(text);
	return DOMPurify.sanitize(html, PURIFY_CONFIG);
}

export default renderMarkdown;
