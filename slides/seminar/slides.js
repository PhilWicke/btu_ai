// slides.js — snap-aware scroll-reveal, keyboard navigation, progress dots
// vanilla, no dependencies; respects prefers-reduced-motion

(function () {
  'use strict';

  const slides = Array.from(document.querySelectorAll('.slide'));
  if (!slides.length) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // -----------------------------------------------------------------------
  // 1. Scroll-reveal — flag each slide as visible when >= 40% in viewport
  // -----------------------------------------------------------------------
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.4, rootMargin: '0px' });
    slides.forEach(function (s) { io.observe(s); });
  } else {
    // Fallback — show everything
    slides.forEach(function (s) { s.classList.add('is-visible'); });
  }

  // -----------------------------------------------------------------------
  // 2. Current slide tracking
  // -----------------------------------------------------------------------
  function currentSlideIndex() {
    const probe = window.innerHeight * 0.45;
    for (let i = 0; i < slides.length; i++) {
      const r = slides[i].getBoundingClientRect();
      if (r.top <= probe && r.bottom > probe) return i;
    }
    return 0;
  }

  function goToSlide(i) {
    if (i < 0 || i >= slides.length) return;
    slides[i].scrollIntoView({
      behavior: reducedMotion ? 'auto' : 'smooth',
      block: 'start',
    });
  }

  // -----------------------------------------------------------------------
  // 3. Keyboard navigation — Arrows / Space / PgUp / PgDn / Home / End
  // -----------------------------------------------------------------------
  document.addEventListener('keydown', function (e) {
    // Don't hijack typing in inputs / contenteditable
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;

    const i = currentSlideIndex();
    let next = null;

    switch (e.key) {
      case 'ArrowDown':
      case 'PageDown':
      case ' ':
      case 'j':
        next = i + 1; break;
      case 'ArrowUp':
      case 'PageUp':
      case 'k':
        next = i - 1; break;
      case 'Home':
        next = 0; break;
      case 'End':
        next = slides.length - 1; break;
      default:
        return;
    }
    e.preventDefault();
    goToSlide(next);
  });

  // -----------------------------------------------------------------------
  // 4. Progress dots — right-edge column, click-to-jump
  // -----------------------------------------------------------------------
  const nav = document.createElement('nav');
  nav.className = 'slide-nav';
  nav.setAttribute('aria-label', 'Slide navigation');
  slides.forEach(function (s, i) {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'slide-nav-dot';
    dot.setAttribute('aria-label', 'Go to slide ' + (i + 1));
    dot.addEventListener('click', function () { goToSlide(i); });
    nav.appendChild(dot);
  });
  document.body.appendChild(nav);

  const dots = nav.querySelectorAll('.slide-nav-dot');
  let lastActive = -1;
  function updateNav() {
    const i = currentSlideIndex();
    if (i === lastActive) return;
    if (lastActive >= 0) dots[lastActive].classList.remove('is-active');
    dots[i].classList.add('is-active');
    lastActive = i;
  }

  // -----------------------------------------------------------------------
  // 5. Slide counter — bottom-left readout (slide N / total)
  // -----------------------------------------------------------------------
  const counter = document.createElement('div');
  counter.className = 'slide-counter';
  counter.setAttribute('aria-hidden', 'true');
  document.body.appendChild(counter);
  function updateCounter() {
    counter.textContent = (currentSlideIndex() + 1) + ' / ' + slides.length;
  }

  // -----------------------------------------------------------------------
  // 6. Lightbox — click a figure to expand it; click backdrop / ESC to close
  // -----------------------------------------------------------------------
  const backdrop = document.createElement('div');
  backdrop.className = 'lightbox-backdrop';
  backdrop.setAttribute('role', 'dialog');
  backdrop.setAttribute('aria-modal', 'true');
  backdrop.setAttribute('aria-hidden', 'true');
  backdrop.innerHTML =
    '<div class="lightbox-figure"><img alt=""></div>' +
    '<button class="lightbox-close" type="button" aria-label="Close">×</button>' +
    '<div class="lightbox-hint">click outside or press ESC to close</div>';
  document.body.appendChild(backdrop);

  const lbImg = backdrop.querySelector('.lightbox-figure img');
  const lbClose = backdrop.querySelector('.lightbox-close');

  function openLightbox(src, alt) {
    lbImg.src = src;
    lbImg.alt = alt || '';
    backdrop.classList.add('is-open');
    backdrop.setAttribute('aria-hidden', 'false');
    document.documentElement.style.overflow = 'hidden';
  }
  function closeLightbox() {
    backdrop.classList.remove('is-open');
    backdrop.setAttribute('aria-hidden', 'true');
    document.documentElement.style.overflow = '';
  }

  document.querySelectorAll('.figure img').forEach(function (img) {
    img.addEventListener('click', function (e) {
      e.stopPropagation();
      openLightbox(img.src, img.alt);
    });
  });

  backdrop.addEventListener('click', function (e) {
    // Clicks on the figure container itself shouldn't close; only backdrop or close button
    if (e.target === backdrop || e.target === lbClose) closeLightbox();
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && backdrop.classList.contains('is-open')) {
      e.preventDefault();
      e.stopPropagation();
      closeLightbox();
    }
  }, true);

  // -----------------------------------------------------------------------
  // 7. Scroll listener (throttled via rAF) for nav + counter
  // -----------------------------------------------------------------------
  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      updateNav();
      updateCounter();
      ticking = false;
    });
  }
  document.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  updateNav();
  updateCounter();
})();
