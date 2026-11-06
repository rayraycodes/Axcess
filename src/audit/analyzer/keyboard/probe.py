"""Playwright-driven keyboard probe (SC 2.1.2 traps + SC 2.1.1 modal checks).

What it does
============

The probe runs two conservative checks against an already-loaded Playwright page:

1. **Modal-dialog keyboard checks (SC 2.1.1)** — for each modal dialog
   already open at page load: record whether initial focus moved into
   the dialog, Tab-walk from inside an ``aria-modal`` dialog to see
   whether focus reaches background content while the dialog stays
   open, and press Escape to see whether the dialog offers any keyboard
   dismiss affordance. All three are review leads requiring manual
   confirmation, never automatic conformance failures.

2. **Bidirectional Tab-walk (SC 2.1.2)** — press Tab through the page and
   record ``document.activeElement`` after each press. A review lead is
   emitted only when the same observable element blocks several consecutive
   Tab attempts *and* several consecutive Shift+Tab attempts.

The probe deliberately does not infer an SC 2.1.2 *trap* from a two-element
focus cycle, Escape behavior, or an iframe remaining
``document.activeElement``. Those observations all have common conforming
explanations. Missing iframe titles are covered by the DOM engines and
belong to accessible-name testing, not SC 2.1.2.

What it does NOT detect
=======================

* Traps or modal dialogs that only appear after user interaction
  (clicking to mount a modal we never see — phase 2 would need the
  interaction probe's click machinery).
* Traps inside closed shadow DOM or embedded browsing contexts.
* Pages behind login.
* Traps that only manifest with screen-reader virtual cursors.
* Components that use a documented non-standard keyboard command to exit.

These limits are surfaced in the Issues view via the existing
scope-honesty banner pattern.

Safety
======

The check is wrapped in a try/except that logs and returns ``[]``.
A bad page must never crash the crawl. Work is bounded to roughly
``max_focusable * 2`` forward Tab presses plus a small, fixed reverse
confirmation. A page with 800 focusable controls therefore cannot make
the production probe run without a cap.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from audit.analyzer.keyboard.base import (
    RULE_IFRAME,
    RULE_MODAL_FOCUS_NOT_CONTAINED,
    RULE_MODAL_INITIAL_FOCUS_OUTSIDE,
    RULE_MODAL_NO_KEYBOARD_DISMISS,
    RULE_NO_ESCAPE,
    RULE_STUCK,
    KeyboardFinding,
    KeyboardTrap,
    ModalKeyboardFinding,
)

if TYPE_CHECKING:
    from playwright.async_api import Page

log = logging.getLogger(__name__)


# --- Tunables. Exposed as constants so tests can monkey-patch and the
# orchestrator can override via construction. ---

# How many Tab presses count as "stuck on this element" before flagging.
# K=4 means: four attempts in each direction failed to move away from
# the same element. One direction alone is not enough: a component may
# intentionally use Tab internally while Shift+Tab provides an exit.
DEFAULT_STUCK_THRESHOLD = 4

# Hard cap on Tab presses per page. focusable_count * 2 + slack covers
# any well-behaved page; the extra slack accounts for elements that
# move focus into iframes / sub-documents we walk through.
DEFAULT_MAX_FOCUSABLE = 50

# Cap on how many open modal dialogs the SC 2.1.1 checks examine per page.
# (Originally introduced for the retired SC 2.1.2 Escape observation.)
DEFAULT_MAX_MODALS = 5

# Cap on Tab presses per dialog during the containment walk. A dialog with
# more focusable controls than this is skipped rather than half-walked.
_MAX_CONTAINMENT_TABS = 25

# Accessible-name fragments that mark a control as a dismiss affordance.
# Lowercase; matched as substrings of the control's visible text,
# aria-label, or title. Kept deliberately short — a miss only downgrades
# the observation, it never upgrades one.
_DISMISS_NAME_FRAGMENTS = (
    "close",
    "dismiss",
    "cancel",
    "exit",
    "\u00d7",  # multiplication sign, the classic close glyph
    "\u2715",  # multiplication x
    "\u2716",  # heavy multiplication x
)

# How many chars of the element's outerHTML we keep on the finding for
# rendering / debugging.
_SNIPPET_CHARS = 240


# --- In-page JS for the modal checks. Kept at module level so each
# snippet is readable as one unit and testable by eye. All of them pierce
# open shadow roots the same way _active_element_signature does, and walk
# ancestry through shadow hosts (getRootNode().host) so a focused element
# inside a component within the dialog still counts as "inside". ---

_ENUMERATE_MODALS_JS = """
(maxModals) => {
  const sel = (
    'dialog[open], ' +
    '[role="dialog"][aria-modal="true"], ' +
    '[aria-modal="true"][role="alertdialog"]'
  );
  const focusableSel = (
    'a[href], button:not([disabled]), input:not([disabled]), ' +
    'select:not([disabled]), textarea:not([disabled]), ' +
    '[tabindex]:not([tabindex="-1"])'
  );
  let ae = document.activeElement;
  while (ae && ae.shadowRoot && ae.shadowRoot.activeElement) {
    ae = ae.shadowRoot.activeElement;
  }
  const insideOf = (node, el) => {
    let cur = node;
    while (cur) {
      if (cur === el) return true;
      cur = cur.parentElement || (cur.getRootNode && cur.getRootNode().host) || null;
    }
    return false;
  };
  const out = [];
  const list = document.querySelectorAll(sel);
  for (let i = 0; i < list.length; i++) {
    const el = list[i];
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const inside = el.querySelectorAll(focusableSel);
    if (!inside.length) continue;
    const mark = 'kbprobe-modal-' + i;
    el.setAttribute('data-kbprobe-mark', mark);
    let ariaModal = el.getAttribute('aria-modal') === 'true';
    if (!ariaModal) {
      // A native dialog opened via showModal() is modal without the
      // attribute; :modal is the only runtime signal.
      try { ariaModal = el.matches(':modal'); } catch (e) {}
    }
    const id = el.id ? '#' + el.id : '';
    let selOut = el.tagName.toLowerCase() + id;
    if (!id && el.className && typeof el.className === 'string') {
      const cls = el.className.trim().split(/\\s+/).slice(0, 2).join('.');
      if (cls) selOut = el.tagName.toLowerCase() + '.' + cls;
    }
    out.push({
      mark: mark,
      selector: selOut,
      snippet: el.outerHTML.slice(0, 240),
      ariaModal: ariaModal,
      containsFocus: !!(ae && ae !== document.body && insideOf(ae, el)),
    });
    if (out.length >= maxModals) break;
  }
  return out;
}
"""

_FOCUS_FIRST_IN_MODAL_JS = """
(mark) => {
  const el = document.querySelector('[data-kbprobe-mark="' + mark + '"]');
  if (!el) return false;
  const first = el.querySelector(
    'a[href], button:not([disabled]), input:not([disabled]), ' +
    'select:not([disabled]), textarea:not([disabled]), ' +
    '[tabindex]:not([tabindex="-1"])'
  );
  if (!first) return false;
  try { first.focus(); } catch (e) { return false; }
  let ae = document.activeElement;
  while (ae && ae.shadowRoot && ae.shadowRoot.activeElement) {
    ae = ae.shadowRoot.activeElement;
  }
  return ae === first;
}
"""

_MODAL_FOCUS_POSITION_JS = """
(mark) => {
  const el = document.querySelector('[data-kbprobe-mark="' + mark + '"]');
  let dialogOpen = false;
  if (el) {
    const cs = getComputedStyle(el);
    dialogOpen = (
      cs.display !== 'none' && cs.visibility !== 'hidden' &&
      (el.tagName.toLowerCase() !== 'dialog' || el.open)
    );
  }
  let ae = document.activeElement;
  while (ae && ae.shadowRoot && ae.shadowRoot.activeElement) {
    ae = ae.shadowRoot.activeElement;
  }
  const isBody = !ae || ae === document.body || ae === document.documentElement;
  let inside = false;
  if (el && ae && !isBody) {
    let cur = ae;
    while (cur) {
      if (cur === el) { inside = true; break; }
      cur = cur.parentElement || (cur.getRootNode && cur.getRootNode().host) || null;
    }
  }
  let outsideSelector = '';
  if (!inside && !isBody && ae) {
    const tag = ae.tagName.toLowerCase();
    const id = ae.id ? '#' + ae.id : '';
    outsideSelector = tag + id;
    if (!id && ae.className && typeof ae.className === 'string') {
      const cls = ae.className.trim().split(/\\s+/).slice(0, 2).join('.');
      if (cls) outsideSelector = tag + '.' + cls;
    }
  }
  return { dialogOpen: dialogOpen, inside: inside, isBody: isBody,
           outsideSelector: outsideSelector };
}
"""

_MODAL_DISMISS_STATE_JS = """
({ mark, frags }) => {
  const el = document.querySelector('[data-kbprobe-mark="' + mark + '"]');
  let open = false;
  if (el) {
    const cs = getComputedStyle(el);
    open = (
      cs.display !== 'none' && cs.visibility !== 'hidden' &&
      (el.tagName.toLowerCase() !== 'dialog' || el.open)
    );
  }
  let ae = document.activeElement;
  while (ae && ae.shadowRoot && ae.shadowRoot.activeElement) {
    ae = ae.shadowRoot.activeElement;
  }
  let inside = false;
  if (el && ae && ae !== document.body) {
    let cur = ae;
    while (cur) {
      if (cur === el) { inside = true; break; }
      cur = cur.parentElement || (cur.getRootNode && cur.getRootNode().host) || null;
    }
  }
  let hasDismiss = false;
  if (el) {
    const dialogFormSel =
      'form[method="dialog"] button, form[method="dialog"] input[type="submit"]';
    if (el.querySelector(dialogFormSel)) {
      hasDismiss = true;
    }
    if (!hasDismiss) {
      const controls = el.querySelectorAll(
        'button, a[href], input[type="button"], input[type="submit"], [role="button"]'
      );
      for (const c of controls) {
        const name = (
          (c.textContent || '') + ' ' +
          (c.getAttribute('aria-label') || '') + ' ' +
          (c.getAttribute('title') || '') + ' ' +
          (c.value || '')
        ).toLowerCase();
        if (frags.some(f => name.includes(f))) { hasDismiss = true; break; }
      }
    }
  }
  return { open: open, inside: inside, hasDismiss: hasDismiss };
}
"""


@dataclass
class KeyboardProbe:
    """Reusable probe. Construct once per crawl, call :meth:`run` per page."""

    max_focusable: int = DEFAULT_MAX_FOCUSABLE
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD
    max_modals: int = DEFAULT_MAX_MODALS
    # Browser errors can include an authenticated URL or target text. The
    # protected companion opts into terse, non-evidence diagnostics instead.
    suppress_diagnostics: bool = False

    async def run(self, page: Page) -> list[KeyboardFinding]:
        """Probe ``page`` for keyboard findings. Returns a (possibly empty) list.

        Safe to call from inside ``JsFetcher.fetch`` — operates on the
        live page before the context closes. Never raises; any failure
        logs at WARNING and returns the findings collected so far.

        The modal checks run FIRST: they must observe where the page put
        initial focus before the tab walk moves it.
        """
        findings: list[KeyboardFinding] = []
        try:
            findings.extend(await self._probe_modal_keyboard(page))
        except Exception as exc:
            self._log_failure("modal_keyboard", exc)
        try:
            findings.extend(await self._probe_tab_walk(page))
        except Exception as exc:
            self._log_failure("tab_walk", exc)
        return findings

    # -----------------------------------------------------------------
    # Modal-dialog keyboard checks (SC 2.1.1).
    # -----------------------------------------------------------------

    async def _probe_modal_keyboard(self, page: Page) -> list[ModalKeyboardFinding]:
        """Check keyboard behavior of modal dialogs open at page load.

        Three observations per dialog, all SC 2.1.1 review leads:

        * ``modal-initial-focus-outside`` — the page loaded with the
          dialog open but focus never moved into it. Must be computed
          before anything else moves focus (hence this probe runs before
          the tab walk).
        * ``modal-focus-not-contained`` — starting inside an
          ``aria-modal`` dialog, Tab reached real background content
          while the dialog stayed open. Plain non-modal ``dialog[open]``
          elements are exempt: containment is only expected of modals.
        * ``modal-no-keyboard-dismiss`` — Escape neither closed the
          dialog nor moved focus out, and no focusable close/cancel
          control was found inside. Explicitly advisory: a page may
          provide another documented keyboard exit.

        Never raises; each step is wrapped and logged via
        :meth:`_log_failure`. Markers are cleaned up best-effort so they
        don't leak into saved HTML.
        """
        try:
            raw = await page.evaluate(_ENUMERATE_MODALS_JS, self.max_modals)
        except Exception as exc:
            self._log_failure("modal_enumerate", exc)
            return []
        modals = [m for m in (raw or []) if isinstance(m, dict) and m.get("mark")]
        if not modals:
            return []

        findings: list[ModalKeyboardFinding] = []
        try:
            # Observation 1 — initial focus. If any dialog contains the
            # initial focus, the page made a reasonable choice; only flag
            # when focus is outside every open modal dialog.
            if not any(m.get("containsFocus") for m in modals):
                for m in modals:
                    findings.append(
                        ModalKeyboardFinding(
                            rule_id=RULE_MODAL_INITIAL_FOCUS_OUTSIDE,
                            impact="minor",
                            target_selector=str(m.get("selector", "(unknown)")),
                            failure_summary=(
                                "Measured initial focus: the page loaded with this modal "
                                "dialog open while document.activeElement remained outside "
                                "the dialog. Keyboard users must Tab through intervening "
                                "content to reach it. Verify the intended focus behavior "
                                "manually — the dialog may receive focus by another path."
                            ),
                            html_snippet=str(m.get("snippet", ""))[:_SNIPPET_CHARS],
                            help=(
                                "When a modal dialog opens, move keyboard focus to an "
                                "element inside it."
                            ),
                        )
                    )

            for m in modals:
                mark = str(m["mark"])
                # Observation 2 — focus containment (aria-modal only).
                if m.get("ariaModal"):
                    try:
                        finding = await self._check_modal_containment(page, mark, m)
                    except Exception as exc:
                        self._log_failure("modal_containment", exc)
                        finding = None
                    if finding is not None:
                        findings.append(finding)
                # Observation 3 — keyboard dismiss.
                try:
                    finding = await self._check_modal_dismiss(page, mark, m)
                except Exception as exc:
                    self._log_failure("modal_dismiss", exc)
                    finding = None
                if finding is not None:
                    findings.append(finding)
        finally:
            # Remove our markers so they don't leak into the page's saved
            # HTML or content hash. Best-effort, never blocking.
            with contextlib.suppress(Exception):
                await page.evaluate(
                    """
                    () => {
                      document.querySelectorAll('[data-kbprobe-mark]').forEach(el => {
                        el.removeAttribute('data-kbprobe-mark');
                      });
                    }
                    """
                )
        return findings

    async def _check_modal_containment(
        self, page: Page, mark: str, modal: dict[str, object]
    ) -> ModalKeyboardFinding | None:
        """Tab from inside the marked aria-modal dialog; flag focus leaks.

        Focus transiting through ``body`` (Chromium routes focus through
        the document while wrapping past browser chrome) is not a leak;
        only landing on a real focusable element outside the dialog
        subtree — while the dialog is still open — is.
        """
        if not await page.evaluate(_FOCUS_FIRST_IN_MODAL_JS, mark):
            # Couldn't put focus inside; the walk would measure the wrong
            # thing. Skip rather than guess.
            return None
        max_tabs = min(self.max_focusable, _MAX_CONTAINMENT_TABS)
        for presses in range(1, max_tabs + 1):
            await page.keyboard.press("Tab")
            pos = await page.evaluate(_MODAL_FOCUS_POSITION_JS, mark)
            if not isinstance(pos, dict) or not pos.get("dialogOpen"):
                # Dialog closed or vanished mid-walk — nothing to contain.
                return None
            if pos.get("inside") or pos.get("isBody"):
                continue
            outside = str(pos.get("outsideSelector") or "(unknown)")
            return ModalKeyboardFinding(
                rule_id=RULE_MODAL_FOCUS_NOT_CONTAINED,
                impact="serious",
                target_selector=str(modal.get("selector", "(unknown)")),
                failure_summary=(
                    "Measured keyboard containment: starting inside this open "
                    "aria-modal dialog, focus reached content outside the dialog "
                    f"({outside}) after {presses} Tab press(es) while the dialog "
                    "stayed open. Background content remains keyboard-operable "
                    "behind the modal. Verify manually whether a focus trap or "
                    "inert background is missing."
                ),
                html_snippet=str(modal.get("snippet", ""))[:_SNIPPET_CHARS],
                help=(
                    "Keyboard focus should stay within an open modal dialog "
                    "until the dialog is dismissed."
                ),
            )
        return None

    async def _check_modal_dismiss(
        self, page: Page, mark: str, modal: dict[str, object]
    ) -> ModalKeyboardFinding | None:
        """Press Escape inside the marked dialog; flag a missing keyboard exit.

        Escape counts as handled when the dialog closes/hides OR focus
        leaves the dialog subtree (close-and-return-focus implementations
        often only move focus). A visible close/cancel/dismiss control, or
        a ``form[method=dialog]`` submit, also counts as a keyboard exit.
        """
        if not await page.evaluate(_FOCUS_FIRST_IN_MODAL_JS, mark):
            return None
        await page.keyboard.press("Escape")
        state = await page.evaluate(
            _MODAL_DISMISS_STATE_JS, {"mark": mark, "frags": list(_DISMISS_NAME_FRAGMENTS)}
        )
        if not isinstance(state, dict):
            return None
        if state.get("open") and state.get("inside") and not state.get("hasDismiss"):
            return ModalKeyboardFinding(
                rule_id=RULE_MODAL_NO_KEYBOARD_DISMISS,
                impact="moderate",
                target_selector=str(modal.get("selector", "(unknown)")),
                failure_summary=(
                    "Measured dismiss behavior: pressing Escape left this modal "
                    "dialog open with focus still inside, and no focusable "
                    "close, cancel, or dismiss control was detected inside the "
                    "dialog. This is not automatically a WCAG failure — the page "
                    "may provide another documented keyboard exit. Verify "
                    "manually how a keyboard user is expected to leave."
                ),
                html_snippet=str(modal.get("snippet", ""))[:_SNIPPET_CHARS],
                help=(
                    "Modal dialogs should be dismissible with the keyboard, for "
                    "example via Escape or a focusable close control."
                ),
            )
        return None

    # -----------------------------------------------------------------
    # Check 1 — Tab walk.
    # -----------------------------------------------------------------

    async def _probe_tab_walk(self, page: Page) -> list[KeyboardTrap]:
        """Walk Tab from body; flag focus that cannot leave an element.

        Identity is exact, not heuristic: an in-page WeakMap hands every
        distinct focused element a stable integer id, so two different
        elements can never collide. (The first version of this probe
        keyed on an outerHTML prefix; dogfooding it against the tracker
        app produced false positives on nav links that share one long
        class string. Exact identity removes that whole failure class.)

        One high-precision review-lead shape is detected: the same
        observable element resists ``stuck_threshold`` exit attempts in
        both directions. Ordinary page wrapping and bounded composite-widget
        cycles are not treated as failures.
        """
        # Move focus to a deterministic starting point. Without this,
        # the walk starts wherever the page's autofocus landed, which
        # makes test results non-deterministic.
        # Some pages reject focus on body (e.g. designMode). The probe
        # still works — Playwright's keyboard.press routes to whatever
        # element is currently focused. Suppression is deliberate.
        with contextlib.suppress(Exception):
            await page.evaluate("() => { try { document.body.focus(); } catch (e) {} }")

        # Walk up to `max_focusable * 2 + 4` Tab presses; enough
        # headroom even for chrome that injects pseudo-focusable
        # elements (scrollbars, etc.).
        max_tabs = self.max_focusable * 2 + 4
        last_id: int | None = None
        stuck_count = 0
        for _ in range(max_tabs):
            try:
                await page.keyboard.press("Tab")
            except Exception as exc:
                self._log_failure("tab_press", exc)
                break
            sig = await self._active_element_signature(page)
            if sig is None:
                # The page detached / navigated away. Stop cleanly.
                break
            el_id = int(sig["el_id"])

            # Body focus (id 0) means "no element is trapped" — either
            # the page has no focusable controls or Tab wrapped through
            # the document. Reset all trap state.
            if el_id == 0:
                stuck_count = 0
                last_id = None
                continue

            # The parent document reports an iframe/object/embed (and a
            # closed-shadow custom element) as the active element while focus
            # may be moving normally inside it. We cannot observe that internal
            # movement, so repeated outer identity is not trap evidence.
            if sig.get("opaque") == "true":
                stuck_count = 0
                last_id = None
                continue

            # Shape 1: stuck on one element across consecutive presses.
            if last_id is not None and el_id == last_id:
                stuck_count += 1
                if stuck_count >= self.stuck_threshold:
                    reverse_blocked = await self._confirm_reverse_exit_blocked(page, el_id)
                    if reverse_blocked:
                        attempts = self.stuck_threshold
                        return [
                            KeyboardTrap(
                                rule_id=RULE_STUCK,
                                impact="critical",
                                target_selector=str(sig.get("selector", "(unknown)")),
                                failure_summary=(
                                    "Measured focus exit behavior: focus remained on this "
                                    f"same element after {attempts} Tab attempts and "
                                    f"{attempts} Shift+Tab attempts ({attempts * 2} failed "
                                    "exit attempts total). Axcess does not count ordinary "
                                    "focus wrapping, two-control cycles, modal containment, "
                                    "or opaque iframe focus as a trap. Confirm manually that "
                                    "the component has no documented keyboard exit command "
                                    "before recording a WCAG 2.1.2 failure."
                                ),
                                html_snippet=str(sig.get("html", ""))[:_SNIPPET_CHARS],
                            )
                        ]
                    # Shift+Tab moved away, so this is demonstrably escapable.
                    # Continue from the new focus position without carrying the
                    # forward-only observation into another candidate.
                    stuck_count = 0
                    last_id = None
                    continue
            else:
                stuck_count = 0
            last_id = el_id

        return []

    async def _confirm_reverse_exit_blocked(self, page: Page, candidate_id: int) -> bool:
        """Require the same focus identity after repeated reverse exits.

        This second direction is the precision gate. If Shift+Tab moves focus
        anywhere else, the component is keyboard-escapable and cannot support
        an automated SC 2.1.2 review lead from this observation.
        """
        for _ in range(self.stuck_threshold):
            try:
                await page.keyboard.press("Shift+Tab")
            except Exception as exc:
                self._log_failure("reverse_tab_press", exc)
                return False
            sig = await self._active_element_signature(page)
            if sig is None or sig.get("opaque") == "true":
                return False
            if int(sig["el_id"]) != candidate_id:
                return False
        return True

    # -----------------------------------------------------------------
    # Legacy observations — intentionally excluded from run().
    # -----------------------------------------------------------------

    # These helpers remain temporarily for direct downstream callers. Escape
    # behavior and missing iframe titles are useful manual-review signals, but
    # neither proves a keyboard trap. Axcess therefore does not persist their
    # output as SC 2.1.2 findings.

    async def _probe_modal_escape(self, page: Page) -> list[KeyboardTrap]:
        """For each open dialog, focus inside and press Esc — verify exit."""
        try:
            modals = await page.evaluate(
                # Arrow functions don't have `arguments`, so we
                # destructure the argument explicitly. Playwright
                # passes the second arg of `page.evaluate` as the
                # first (and only) argument to this function.
                """
                (maxModals) => {
                  const out = [];
                  const sel = (
                    'dialog[open], ' +
                    '[role="dialog"][aria-modal="true"], ' +
                    '[aria-modal="true"][role="alertdialog"]'
                  );
                  const list = document.querySelectorAll(sel);
                  for (let i = 0; i < list.length; i++) {
                    const el = list[i];
                    const inside = el.querySelectorAll(
                      'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
                    );
                    if (!inside.length) continue;
                    // Stamp a marker so we can find this element again
                    // post-Escape without keeping a JS handle alive.
                    const mark = 'kbprobe-modal-' + i;
                    el.setAttribute('data-kbprobe-mark', mark);
                    out.push({ mark: mark, snippet: el.outerHTML.slice(0, 240) });
                    if (out.length >= maxModals) break;
                  }
                  return out;
                }
                """,
                self.max_modals,
            )
        except Exception as exc:
            self._log_failure("modal_enumerate", exc)
            return []

        if not modals:
            return []

        findings: list[KeyboardTrap] = []
        for modal in modals:
            mark: str = modal["mark"]
            snippet: str = modal.get("snippet", "")
            try:
                # Move focus into the modal's first focusable element.
                # Playwright's locator API is the safe path — query
                # selectors via the live frame.
                focused = await page.evaluate(
                    f"""
                    () => {{
                      const el = document.querySelector('[data-kbprobe-mark="{mark}"]');
                      if (!el) return false;
                      const first = el.querySelector(
                        'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
                      );
                      if (!first) return false;
                      first.focus();
                      return document.activeElement === first;
                    }}
                    """
                )
                if not focused:
                    # Couldn't put focus inside; can't probe Escape
                    # reliably. Skip — not the same as a known trap.
                    continue

                await page.keyboard.press("Escape")

                # After Escape, where is focus? Walk up the DOM from
                # activeElement; if the marked modal is an ancestor,
                # focus is still trapped inside.
                still_inside = await page.evaluate(
                    f"""
                    () => {{
                      const ae = document.activeElement;
                      if (!ae) return false;
                      let cur = ae;
                      while (cur) {{
                        if (cur.getAttribute &&
                            cur.getAttribute('data-kbprobe-mark') === '{mark}') {{
                          return true;
                        }}
                        cur = cur.parentElement;
                      }}
                      return false;
                    }}
                    """
                )
                if still_inside:
                    findings.append(
                        KeyboardTrap(
                            rule_id=RULE_NO_ESCAPE,
                            impact="critical",
                            target_selector=f"[data-kbprobe-mark='{mark}']",
                            failure_summary=(
                                "Focus did not leave this modal after pressing Escape. "
                                "Modals must release focus on Escape so keyboard users can exit."
                            ),
                            html_snippet=snippet[:_SNIPPET_CHARS],
                        )
                    )
            except Exception as exc:
                self._log_failure("modal_escape_check", exc)
                continue

        # Clean up our markers so we don't leak DOM artifacts into the
        # caller's HTML hash / saved HTML. Suppression is deliberate;
        # marker cleanup is best-effort, never blocking.
        with contextlib.suppress(Exception):
            await page.evaluate(
                """
                () => {
                  document.querySelectorAll('[data-kbprobe-mark]').forEach(el => {
                    el.removeAttribute('data-kbprobe-mark');
                  });
                }
                """
            )

        return findings

    # -----------------------------------------------------------------
    # Check 3 — Iframe heuristic.
    # -----------------------------------------------------------------

    async def _probe_iframe_sanity(self, page: Page) -> list[KeyboardTrap]:
        """Flag iframes that are likely to trap focus (missing title + no tabindex)."""
        try:
            raw = await page.evaluate(
                """
                () => {
                  const out = [];
                  const list = document.querySelectorAll('iframe');
                  for (let i = 0; i < list.length && out.length < 8; i++) {
                    const f = list[i];
                    const title = (f.getAttribute('title') || '').trim();
                    const ti = f.getAttribute('tabindex');
                    const hasContent = !!(f.src || f.srcdoc);
                    // Only an iframe a keyboard user can actually Tab INTO can
                    // trap focus. Tracking / ad / pixel iframes are
                    // display:none, 0x0, hidden, or aria-hidden — never in the
                    // tab order — so flagging them as keyboard traps is a false
                    // positive (and they're on nearly every real page). Gate on
                    // real visibility.
                    // clientWidth/Height are the content-box size: 0 for a
                    // display:none or width=0/height=0 iframe (getBoundingClientRect
                    // would report ~4px for the latter because of the default 2px
                    // frame border, which let tracking pixels slip through).
                    const cs = getComputedStyle(f);
                    const reachable = (
                      cs.display !== 'none' &&
                      cs.visibility !== 'hidden' &&
                      !f.hasAttribute('hidden') &&
                      f.getAttribute('aria-hidden') !== 'true' &&
                      f.clientWidth > 2 && f.clientHeight > 2
                    );
                    // Heuristic: a visible, Tab-reachable iframe (tabindex !==
                    // '-1') with content and no title is the canonical "user
                    // gets stuck in the embedded doc" shape.
                    if (hasContent && reachable && title === '' && ti !== '-1') {
                      out.push({
                        selector: 'iframe' + (f.id ? '#' + f.id : ''),
                        snippet: f.outerHTML.slice(0, 240),
                      });
                    }
                  }
                  return out;
                }
                """
            )
        except Exception as exc:
            self._log_failure("iframe_scan", exc)
            return []

        return [
            KeyboardTrap(
                rule_id=RULE_IFRAME,
                # Heuristic, not a confirmed trap; impact serious rather
                # than critical so it doesn't outrank the confirmed cases.
                impact="serious",
                target_selector=item.get("selector", "iframe"),
                failure_summary=(
                    "Iframe is reachable by Tab but has no title. Keyboard users can land "
                    "inside but cannot tell what they entered, and if the embedded document "
                    'traps focus, they cannot escape. Add title="..." and ensure the '
                    "embedded document is itself keyboard-accessible."
                ),
                html_snippet=item.get("snippet", "")[:_SNIPPET_CHARS],
            )
            for item in (raw or [])
        ]

    # -----------------------------------------------------------------
    # Helpers.
    # -----------------------------------------------------------------

    async def _active_element_signature(self, page: Page) -> dict[str, str] | None:
        """Return an exact identity record for document.activeElement.

        Identity comes from an in-page WeakMap that hands every distinct
        focused element a stable integer id for the lifetime of the
        page. Two different elements can never share an id, no matter
        how similar their markup is. (The first version keyed on an
        outerHTML prefix and false-positived on nav links that share a
        long class string; dogfooding caught it.)

        Returns ``None`` if the page has detached (navigation away,
        crash). Returns a dict with:

          * ``el_id``: "0" for body or no focus, otherwise a stable
            positive integer unique to this element.
          * ``selector``: best-effort CSS selector for reporting.
          * ``html``: outerHTML prefix for the finding's snippet.
        """
        try:
            raw: object = await page.evaluate(
                """
                () => {
                  // Descend through (open) shadow roots to the innermost
                  // focused element. Without this, a web component's HOST
                  // shows up as document.activeElement for every Tab press
                  // while focus actually moves through its internal
                  // controls — which the tab-walk misreads as focus being
                  // "stuck" on the host (a false trap). Closed shadow roots
                  // expose no activeElement, so those stay opaque.
                  let ae = document.activeElement;
                  while (ae && ae.shadowRoot && ae.shadowRoot.activeElement) {
                    ae = ae.shadowRoot.activeElement;
                  }
                  if (!ae || ae === document.body) {
                    return { el_id: 0, selector: 'body', html: '', opaque: false };
                  }
                  if (!window.__kbprobe_ids) {
                    window.__kbprobe_ids = { map: new WeakMap(), next: 1 };
                  }
                  const reg = window.__kbprobe_ids;
                  if (!reg.map.has(ae)) {
                    reg.map.set(ae, reg.next);
                    reg.next += 1;
                  }
                  const tag = ae.tagName.toLowerCase();
                  // Focus movement inside these boundaries is not observable
                  // from the parent document. Repeated host identity must not
                  // be mistaken for repeated focus on the same inner control.
                  const opaque = (
                    ['iframe', 'object', 'embed'].includes(tag) ||
                    (tag.includes('-') && !ae.shadowRoot)
                  );
                  const id = ae.id ? '#' + ae.id : '';
                  // Try to build something selector-ish for reporting.
                  let sel = tag + id;
                  if (!id && ae.className && typeof ae.className === 'string') {
                    const cls = ae.className.trim().split(/\\s+/).slice(0, 2).join('.');
                    if (cls) sel = tag + '.' + cls;
                  }
                  const html = (ae.outerHTML || '').slice(0, 240);
                  return {
                    el_id: reg.map.get(ae), selector: sel, html: html, opaque: opaque
                  };
                }
                """
            )
        except Exception as exc:
            self._log_failure("active_element", exc, debug=True)
            return None
        # `page.evaluate` is typed as Any in Playwright's stubs; we
        # know our JS returns either a 3-key object or undefined.
        # Narrow it through a runtime shape check so the function's
        # declared return type is sound.
        if isinstance(raw, dict) and "el_id" in raw:
            return {
                "el_id": str(raw.get("el_id", 0)),
                "selector": str(raw.get("selector", "(unknown)")),
                "html": str(raw.get("html", "")),
                "opaque": "true" if bool(raw.get("opaque", False)) else "false",
            }
        return None

    def _log_failure(self, check: str, exc: Exception, *, debug: bool = False) -> None:
        """Log safely for public scans and without target data for protected ones."""

        logger = log.debug if debug else log.warning
        if self.suppress_diagnostics:
            logger("keyboard.%s_failed_in_protected_context", check)
        else:
            logger("keyboard.%s_failed: %s", check, exc)
