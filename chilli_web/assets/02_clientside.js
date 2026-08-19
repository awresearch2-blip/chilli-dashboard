/* Client-side callbacks: small, purely cosmetic DOM updates that don't
   justify a Python round-trip. Dash auto-loads every assets/*.js file and
   exposes functions registered here under window.dash_clientside.<namespace>. */

window.dash_clientside = window.dash_clientside || {};

window.dash_clientside.chilli = {
  /**
   * Highlight the sidebar link matching the current URL. Runs purely in the
   * browser: matching an href against window.location is not worth a Flask
   * round trip, and it avoids needing a per-link pattern-matching component
   * id in the DOM (see layout_shell.py's _nav_links for why that was
   * dropped).
   */
  highlightNav: function (pathname) {
    var links = document.querySelectorAll(".nav-link");
    links.forEach(function (el) {
      if (el.getAttribute("href") === pathname) {
        el.classList.add("active");
      } else {
        el.classList.remove("active");
      }
    });
    return pathname;
  },
};
