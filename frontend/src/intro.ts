/**
 * The intro: a short boot sequence over the orb when the page opens.
 *
 * Longer when the wake word opened this window (`?summoned=1`): the rings
 * spin up, the name assembles, and it ends on "listening" — then Auvren asks
 * what you want. Built with createElement/textContent only, like the rest of
 * the UI, and cut to a plain fade for anyone who has asked for reduced motion.
 */

const NAME = "AUVREN";

function el(tag: string, cls: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function playIntro(summoned: boolean): Promise<void> {
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  const total = reduced ? 600 : summoned ? 2400 : 1500;

  const root = el("div", `intro${summoned ? " is-summoned" : ""}${reduced ? " is-reduced" : ""}`);
  root.setAttribute("aria-hidden", "true");
  root.style.setProperty("--intro-ms", `${total}ms`);

  const rings = el("div", "intro-rings");
  for (const r of ["r1", "r2", "r3"]) rings.appendChild(el("span", `intro-ring ${r}`));
  rings.appendChild(el("span", "intro-core"));
  root.appendChild(rings);

  const word = el("div", "intro-word");
  NAME.split("").forEach((ch, i) => {
    const letter = el("span", "intro-letter", ch);
    letter.style.animationDelay = `${(reduced ? 0 : 250) + i * (summoned ? 90 : 60)}ms`;
    word.appendChild(letter);
  });
  root.appendChild(word);

  root.appendChild(el("div", "intro-line"));
  const status = el("div", "intro-status", summoned ? "waking" : "initialising");
  root.appendChild(status);
  document.body.appendChild(root);

  const steps = summoned
    ? [[700, "voice online"], [1400, "systems online"], [2000, "listening"]]
    : [[800, "online"]];
  if (!reduced) {
    for (const [at, text] of steps) {
      window.setTimeout(() => { status.textContent = String(text); }, Number(at));
    }
  } else if (summoned) {
    status.textContent = "listening";
  }

  return new Promise((resolve) => {
    window.setTimeout(() => {
      root.classList.add("is-done");
      resolve();
      window.setTimeout(() => root.remove(), 600);
    }, total);
  });
}
