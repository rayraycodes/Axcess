"""Integration tests for the dynamic keyboard probe (SC 2.1.2 + SC 2.1.1).

The probe runs against real Playwright on hand-built fixture pages in
``tests/fixtures/site/sc_2_1_2/`` and ``sc_2_1_1/``. The SC 2.1.2 accuracy
contract is deliberately asymmetric: normal wrapping, modal containment,
and iframe focus must be suppressed; only a repeatable bidirectional exit
failure becomes a trap lead. Modal keyboard behavior (focus containment,
Escape, initial focus) is reported separately as SC 2.1.1 review leads
with the ``modal-*`` rule ids — never as a 2.1.2 trap.

  * Negative (for traps): ``clean.html``, both modal fixtures, and the
    iframe fixture.
  * Positive (traps): ``tab_loop.html`` → ``keyboard-trap-stuck`` only
    after Tab and Shift+Tab both fail repeatedly.
  * SC 2.1.1 modal checks: one fixture per rule id in ``sc_2_1_1/`` plus
    the fully conforming ``modal_focus_trapped.html`` negative control.

Skipped when Playwright + chromium aren't installed. The local dev
machine has them after ``make setup``; CI installs them in the
``pytest-playwright`` step.

We **don't** use a fixture HTTP server here — Playwright supports
``file://`` URLs, which avoids one whole class of "the integration
test failed because the fixture server bound the wrong port" flakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audit.analyzer.keyboard import KeyboardProbe
from audit.analyzer.keyboard.base import (
    RULE_MODAL_FOCUS_NOT_CONTAINED,
    RULE_MODAL_INITIAL_FOCUS_OUTSIDE,
    RULE_MODAL_NO_KEYBOARD_DISMISS,
    RULE_STUCK,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "site" / "sc_2_1_2"
FIXTURE_DIR_2_1_1 = Path(__file__).resolve().parents[1] / "fixtures" / "site" / "sc_2_1_1"

_MODAL_RULE_IDS = {
    RULE_MODAL_FOCUS_NOT_CONTAINED,
    RULE_MODAL_INITIAL_FOCUS_OUTSIDE,
    RULE_MODAL_NO_KEYBOARD_DISMISS,
}


def _file_url(name: str) -> str:
    """file:// URL for a fixture page."""
    return (FIXTURE_DIR / name).resolve().as_uri()


def _file_url_2_1_1(name: str) -> str:
    """file:// URL for an SC 2.1.1 modal fixture page."""
    return (FIXTURE_DIR_2_1_1 / name).resolve().as_uri()


# Skip the whole module if Playwright isn't importable / chromium not
# installed. Mirrors how the existing axe integration tests gate.
playwright = pytest.importorskip("playwright.async_api")


@pytest.fixture
async def browser():  # type: ignore[no-untyped-def]
    """One headless chromium per test session.

    A fresh context per test (in the page fixture) gives test isolation
    without paying the browser-launch cost five times.
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            await browser.close()
    finally:
        await pw.stop()


@pytest.fixture
async def page(browser):  # type: ignore[no-untyped-def]
    ctx = await browser.new_context()
    try:
        p = await ctx.new_page()
        yield p
    finally:
        await ctx.close()


# --------------------------------------------------------------------
# Negative cases — must produce zero findings.
# False positives here would be the worst failure mode: the probe
# tells operators their clean pages have traps. Keep these strict.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_page_has_no_findings(page) -> None:  # type: ignore[no-untyped-def]
    """A normal page with several focusable elements: zero findings."""
    await page.goto(_file_url("clean.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert findings == [], (
        f"Expected zero findings on the clean fixture, got {len(findings)}: "
        + ", ".join(f.rule_id for f in findings)
    )


@pytest.mark.asyncio
async def test_modal_with_proper_escape_handler_clean(page) -> None:  # type: ignore[no-untyped-def]
    """A modal that releases focus on Escape isn't flagged as no-escape.

    The SC 2.1.2 stance is unchanged: no ``keyboard-trap-*`` finding and no
    dismiss advisory (Escape works). The fixture does NOT trap focus or move
    initial focus into the dialog, so the SC 2.1.1 modal checks legitimately
    report those two observations — assert them explicitly rather than
    asserting an empty list.
    """
    await page.goto(_file_url("modal_clean.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert all(not f.rule_id.startswith("keyboard-trap-") for f in findings)
    assert all(f.criterion_sc != "2.1.2" for f in findings)
    rule_ids = {f.rule_id for f in findings}
    # Escape is handled → never a dismiss advisory.
    assert RULE_MODAL_NO_KEYBOARD_DISMISS not in rule_ids
    # aria-modal without a focus trap → containment lead (SC 2.1.1).
    assert RULE_MODAL_FOCUS_NOT_CONTAINED in rule_ids


# --------------------------------------------------------------------
# Regression: false-positive classes found by dogfooding the probe
# against real, trap-free pages. These build the page with set_content
# (no fixture file needed) because the markup is small and self-contained.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_dom_component_is_not_a_stuck_trap(page) -> None:  # type: ignore[no-untyped-def]
    """A web component with several internal focusable controls.

    document.activeElement reports the shadow HOST while focus actually
    moves through the inner controls. Without piercing the shadow root the
    tab-walk reads that as focus "stuck" on the host — a false trap. This
    is common on modern sites (design systems, embedded widgets).
    """
    await page.set_content(
        "<!doctype html><html lang=en><body><main>"
        "<my-toolbar></my-toolbar><a href='/after'>After</a></main>"
        "<script>customElements.define('my-toolbar', class extends HTMLElement{"
        "constructor(){super();this.attachShadow({mode:'open'}).innerHTML="
        "'<button>1</button><button>2</button><button>3</button>"
        "<button>4</button><button>5</button>';}});</script>"
        "</body></html>"
    )
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    stuck = [f for f in findings if f.rule_id == RULE_STUCK]
    assert stuck == [], (
        "A shadow-DOM component with internal controls must not be flagged as "
        f"a stuck keyboard trap; got: {[f.target_selector for f in stuck]}"
    )


@pytest.mark.asyncio
async def test_hidden_iframes_are_not_flagged(page) -> None:  # type: ignore[no-untyped-def]
    """Tracking / ad / pixel iframes aren't keyboard-reachable.

    display:none, 0x0, hidden, and aria-hidden iframes can't receive Tab
    focus, so the iframe heuristic must skip them — otherwise nearly every
    real page (analytics, ads) lights up with phantom keyboard traps.
    """
    await page.set_content(
        "<!doctype html><html lang=en><body>"
        "<iframe src='https://a.example/p' style='display:none'></iframe>"
        "<iframe src='https://b.example/px' width=0 height=0></iframe>"
        "<iframe src='https://c.example/x' style='visibility:hidden'></iframe>"
        "<iframe src='https://d.example' hidden></iframe>"
        "<iframe src='https://e.example' aria-hidden='true' "
        "style='position:absolute;left:-9999px'></iframe>"
        "<a href='/n'>Next</a></body></html>"
    )
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert findings == []


@pytest.mark.asyncio
async def test_visible_untitled_iframe_is_not_called_a_keyboard_trap(page) -> None:  # type: ignore[no-untyped-def]
    """A missing frame title does not establish that keyboard focus is trapped."""
    await page.set_content(
        "<!doctype html><html lang=en><body>"
        "<iframe src='https://maps.example/embed' width=600 height=400></iframe>"
        "</body></html>"
    )
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert findings == []


@pytest.mark.asyncio
async def test_two_control_page_wrap_is_not_a_trap(page) -> None:  # type: ignore[no-untyped-def]
    """A normal A→B→A tab sequence is page wrapping, not a focus trap."""
    await page.set_content(
        "<!doctype html><html lang=en><body>"
        "<a href='#a'>First</a><button type=button>Second</button>"
        "</body></html>"
    )
    findings = await KeyboardProbe(max_focusable=20, stuck_threshold=3).run(page)
    assert findings == []


@pytest.mark.asyncio
async def test_opaque_iframe_focus_is_not_misread_as_stuck(page) -> None:  # type: ignore[no-untyped-def]
    """The parent sees only the iframe while focus moves through inner controls."""
    await page.set_content(
        "<!doctype html><html lang=en><body><a href='#before'>Before</a>"
        "<iframe title='Editor' srcdoc=\"<button>One</button><button>Two</button>"
        '<button>Three</button><button>Four</button><button>Five</button>"></iframe>'
        "<a href='#after'>After</a></body></html>"
    )
    findings = await KeyboardProbe(max_focusable=20, stuck_threshold=3).run(page)
    assert findings == []


# --------------------------------------------------------------------
# Positive cases — each fixture has the named trap shape.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detects_stuck_focus_from_swallowed_tab(page) -> None:  # type: ignore[no-untyped-def]
    """A button whose keydown handler eats Tab → keyboard-trap-stuck."""
    await page.goto(_file_url("tab_loop.html"))
    # Move focus to the trap button via JS so the probe walks into it
    # immediately. This sidesteps the "what if the page autofocuses
    # somewhere else" non-determinism.
    await page.evaluate("document.getElementById('trap').focus()")
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=3)
    findings = await probe.run(page)
    stuck = [f for f in findings if f.rule_id == RULE_STUCK]
    assert stuck, "Expected at least one keyboard-trap-stuck finding; got rules: " + ", ".join(
        f.rule_id for f in findings
    )
    # The reported selector should point at SOME button-shaped element
    # (we don't pin the exact selector — the probe's selector
    # generation is best-effort, not contractual).
    assert any("button" in f.target_selector.lower() for f in stuck), (
        f"Stuck finding's selector should mention 'button': {[f.target_selector for f in stuck]}"
    )
    assert all("3 Tab attempts" in f.failure_summary for f in stuck)
    assert all("3 Shift+Tab attempts" in f.failure_summary for f in stuck)
    assert all("6 failed exit attempts total" in f.failure_summary for f in stuck)


@pytest.mark.asyncio
async def test_forward_only_block_is_suppressed_when_reverse_exits(page) -> None:  # type: ignore[no-untyped-def]
    """A forward observation alone cannot support a no-keyboard-trap lead."""
    await page.set_content(
        "<!doctype html><html lang=en><body><a href='#before'>Before</a>"
        "<button id=widget>Widget</button><a href='#after'>After</a>"
        "<script>widget.addEventListener('keydown', e => {"
        "if (e.key === 'Tab' && !e.shiftKey) e.preventDefault();"
        "});</script></body></html>"
    )
    await page.evaluate("document.getElementById('widget').focus()")
    findings = await KeyboardProbe(max_focusable=20, stuck_threshold=3).run(page)
    assert findings == []


@pytest.mark.asyncio
async def test_modal_without_escape_is_not_automatically_a_trap(page) -> None:  # type: ignore[no-untyped-def]
    """Escape is not the only conforming keyboard method for leaving a component.

    The SC 2.1.2 stance is unchanged: a modal without an Escape handler is
    never reported as a ``keyboard-trap-*`` finding. The fixture's dialog has
    a focusable "Cancel" button, so even the SC 2.1.1 dismiss advisory stays
    quiet; what it legitimately gets are the containment and initial-focus
    observations (no trap, no autofocus).
    """
    await page.goto(_file_url("modal_no_escape.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert all(not f.rule_id.startswith("keyboard-trap-") for f in findings)
    assert all(f.criterion_sc != "2.1.2" for f in findings)
    rule_ids = {f.rule_id for f in findings}
    # A focusable Cancel control counts as a keyboard dismiss affordance.
    assert RULE_MODAL_NO_KEYBOARD_DISMISS not in rule_ids
    assert RULE_MODAL_FOCUS_NOT_CONTAINED in rule_ids


# --------------------------------------------------------------------
# SC 2.1.1 modal keyboard checks — one fixture per rule id, plus the
# fully conforming negative control.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conforming_modal_produces_no_modal_findings(page) -> None:  # type: ignore[no-untyped-def]
    """Initial focus inside + focus trap + Escape close → zero modal findings."""
    await page.goto(_file_url_2_1_1("modal_focus_trapped.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    modal = [f for f in findings if f.rule_id in _MODAL_RULE_IDS]
    assert modal == [], (
        "Conforming modal fixture must produce no modal findings; got: "
        + ", ".join(f"{f.rule_id} ({f.target_selector})" for f in modal)
    )


@pytest.mark.asyncio
async def test_modal_focus_leak_is_detected(page) -> None:  # type: ignore[no-untyped-def]
    """Tab escaping an open aria-modal dialog → modal-focus-not-contained only."""
    await page.goto(_file_url_2_1_1("modal_focus_leaks.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    modal = [f for f in findings if f.rule_id in _MODAL_RULE_IDS]
    assert {f.rule_id for f in modal} == {RULE_MODAL_FOCUS_NOT_CONTAINED}, (
        f"Expected only the containment lead; got: {[f.rule_id for f in modal]}"
    )
    (finding,) = modal
    assert finding.criterion_sc == "2.1.1"
    assert finding.wcag_level == "A"
    assert finding.impact == "serious"
    assert "dlg" in finding.target_selector
    assert finding.failure_summary.startswith("Measured")
    # Must never collide with the keyboard-trap-stuck currency phrase.
    assert "Shift+Tab attempts" not in finding.failure_summary


@pytest.mark.asyncio
async def test_modal_without_any_keyboard_dismiss_is_advisory(page) -> None:  # type: ignore[no-untyped-def]
    """No Escape handling and no close-named control → dismiss advisory only."""
    await page.goto(_file_url_2_1_1("modal_no_dismiss.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    modal = [f for f in findings if f.rule_id in _MODAL_RULE_IDS]
    assert {f.rule_id for f in modal} == {RULE_MODAL_NO_KEYBOARD_DISMISS}, (
        f"Expected only the dismiss advisory; got: {[f.rule_id for f in modal]}"
    )
    (finding,) = modal
    assert finding.criterion_sc == "2.1.1"
    assert finding.impact == "moderate"
    # The summary must carry the advisory framing — this observation is
    # explicitly not an automatic WCAG failure.
    assert "not automatically a WCAG failure" in finding.failure_summary
    # A proper focus trap must never be misread as an SC 2.1.2 trap.
    assert all(not f.rule_id.startswith("keyboard-trap-") for f in findings)


@pytest.mark.asyncio
async def test_modal_open_at_load_without_focus_is_observed(page) -> None:  # type: ignore[no-untyped-def]
    """Dialog open at load with focus on body → initial-focus observation only."""
    await page.goto(_file_url_2_1_1("modal_initial_focus_body.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    modal = [f for f in findings if f.rule_id in _MODAL_RULE_IDS]
    assert {f.rule_id for f in modal} == {RULE_MODAL_INITIAL_FOCUS_OUTSIDE}, (
        f"Expected only the initial-focus observation; got: {[f.rule_id for f in modal]}"
    )
    (finding,) = modal
    assert finding.criterion_sc == "2.1.1"
    assert finding.impact == "minor"


@pytest.mark.asyncio
async def test_modal_probe_never_raises_when_dialog_vanishes(page) -> None:  # type: ignore[no-untyped-def]
    """A dialog that removes itself on first keydown must not crash the probe."""
    await page.set_content(
        "<!doctype html><html lang=en><body>"
        "<a href='#x'>Link</a>"
        "<div id=dlg role=dialog aria-modal=true>"
        "<input type=text><button type=button>OK</button></div>"
        "<script>document.addEventListener('keydown', () => {"
        "const d = document.getElementById('dlg'); if (d) d.remove();"
        "}, { once: true });</script>"
        "</body></html>"
    )
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    # Must return a list, not raise — the dialog disappearing mid-walk is
    # treated as "nothing to contain".
    findings = await probe.run(page)
    assert all(f.rule_id != RULE_MODAL_FOCUS_NOT_CONTAINED for f in findings)


@pytest.mark.asyncio
async def test_untitled_iframe_fixture_is_not_a_keyboard_trap(page) -> None:  # type: ignore[no-untyped-def]
    """Frame naming is left to the DOM engines and not mapped to SC 2.1.2."""
    await page.goto(_file_url("iframe_no_exit.html"))
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    findings = await probe.run(page)
    assert findings == []


# --------------------------------------------------------------------
# Safety / robustness.
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_never_raises_on_quirky_page(page) -> None:  # type: ignore[no-untyped-def]
    """A pathological page (no focusables, weird markup) must not crash."""
    await page.set_content("<!doctype html><html><body><p>Just some text.</p></body></html>")
    probe = KeyboardProbe(max_focusable=20, stuck_threshold=4)
    # Must return an empty list, not raise.
    findings = await probe.run(page)
    assert findings == []
