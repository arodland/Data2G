// Render the card to PNG and the 8-up sheet to PDF with Playwright's Chromium.
//   node docs/id-card/render.js
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const dir = __dirname;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1012, height: 638 }, deviceScaleFactor: 2 });
  const svg = fs.readFileSync(path.join(dir, 'data2g-id-card.svg'), 'utf8')
    .replace(/<\?xml[^>]*>/, '')
    .replace('width="85.6mm" height="54mm"', 'width="1012" height="638"');
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
  await page.screenshot({ path: path.join(dir, 'data2g-id-card.png'), omitBackground: true,
                          clip: { x: 0, y: 0, width: 1012, height: 638 } });
  await page.goto('file://' + path.join(dir, 'sheet.html'));
  await page.pdf({ path: path.join(dir, 'data2g-id-cards-letter.pdf'), width: '8.5in', height: '11in',
                   printBackground: true });
  await browser.close();
  console.log('wrote data2g-id-card.png and data2g-id-cards-letter.pdf');
})();
