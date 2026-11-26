"use strict";

const chapterLinks = Array.from(document.querySelectorAll('.chapters a[href^="#"]'));
const chapters = chapterLinks.map((link) => document.getElementById(link.hash.slice(1)));
let chapterUpdatePending = false;
function updateChapter() {
  const threshold = Math.min(window.innerHeight * 0.3, 150);
  let current = 0;
  for (let index = 1; index < chapters.length; index += 1) {
    if (chapters[index].getBoundingClientRect().top > threshold) break;
    current = index;
  }
  chapterLinks.forEach((link, index) => {
    if (index === current) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
  chapterUpdatePending = false;
}
function queueChapterUpdate() {
  if (chapterUpdatePending) return;
  chapterUpdatePending = true;
  requestAnimationFrame(updateChapter);
}
window.addEventListener("scroll", queueChapterUpdate, { passive: true });
window.addEventListener("resize", queueChapterUpdate);
window.addEventListener("pageshow", updateChapter);
updateChapter();
