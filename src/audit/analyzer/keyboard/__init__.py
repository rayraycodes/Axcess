"""Dynamic keyboard probe (WCAG SC 2.1.2 traps + SC 2.1.1 modal checks).

Runs *inside* a live Playwright page after axe-core has finished, before
the page closes. The probe first checks modal dialogs that are open at
page load (initial focus, focus containment, keyboard dismissability —
SC 2.1.1 review leads), then presses Tab in a loop while watching
``document.activeElement``; if focus gets stuck on a single element in
both directions, the probe records an SC 2.1.2 review lead.

This is the GenA11y-paper-acknowledged "dynamic analysis" criterion:
static DOM/CSS reading can't tell you whether a keydown handler will
swallow the next Tab. Playwright is the only way.
"""

from audit.analyzer.keyboard.base import KeyboardFinding, KeyboardTrap, ModalKeyboardFinding
from audit.analyzer.keyboard.probe import KeyboardProbe

__all__ = ["KeyboardFinding", "KeyboardProbe", "KeyboardTrap", "ModalKeyboardFinding"]
