"""Capture screenshots of every page of the running Streamlit app (python scripts/capture_screenshots.py)."""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
PAGES = ["Dashboard", "Dataset Upload and Profiling", "Missingness Simulation", "Imputation Comparison",
         "Model Evaluation", "Global SHAP Analysis", "Class-Wise SHAP Analysis", "Results and Export"]


def wait_idle(page, extra=1500):
    page.wait_for_timeout(800)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=600_000)
    page.wait_for_timeout(extra)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1500}, device_scale_factor=1)
        page.goto("http://localhost:8501", wait_until="networkidle")
        wait_idle(page, 4000)
        for i, name in enumerate(PAGES, start=1):
            page.get_by_text(name, exact=True).first.click()
            wait_idle(page)
            if name == "Imputation Comparison":
                page.get_by_role("button", name="Run lightweight experiment").click()
                page.get_by_text("Reconstruction error", exact=False).first.wait_for(timeout=900_000)
                wait_idle(page, 3000)
            page.screenshot(path=str(OUT / f"{i:02d}_{name.lower().replace(' ', '_').replace('-', '_')}.png"), full_page=True)
            print("captured", name)
        browser.close()


if __name__ == "__main__":
    main()

