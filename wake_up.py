from playwright.sync_api import sync_playwright
import time

URL = "https://pdf-fact-checker.streamlit.app/"

def wake_up_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        print(f"Checking URL: {URL}")
        page = browser.new_page()
        try:
            page.goto(URL, timeout=60000)
            print("Page loaded, waiting 30s for Streamlit...")
            time.sleep(30)
            title = page.title()
            print(f"Success loading {title}")
        except Exception as e:
            print(f"Error visiting {URL}: {e}")
        finally:
            page.close()
        
        browser.close()

if __name__ == "__main__":
    wake_up_app()