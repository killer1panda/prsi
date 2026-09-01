import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('http://localhost:3000')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='/Users/ajay/.gemini/antigravity/brain/86194ca3-759c-4a6e-8090-95f0f4ce7e2e/scratch/dashboard.png', full_page=True)
        await browser.close()

asyncio.run(main())
