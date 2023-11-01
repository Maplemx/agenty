import json
import asyncio
import threading
import aiohttp
import importlib.metadata

async def check_version_async():
    current_version = importlib.metadata.version("Agently")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://agently.tech/meta/api/latest?version={ current_version }",
            headers = { "Content-Type": "application/json" },
        ) as response:
            try:
                async for chunk in response.content.iter_chunks():
                    result = chunk[0].decode("utf-8")
                    result = json.loads(result)
                    stable_version = result["data"]["version"] if "version" in result["data"] else None
                    latest_version = result["data"]["latest"] if "latest" in result["data"] else None
                    tips_type = result["data"]["type"]
                    tips_content = result["data"]["tips"]
                    print("[Agently Version Check]")
                    print("This check only work 1 time each day in debug model.")
                    print(f"Current Installed Version: { current_version }")
                    if stable_version:
                        print(f"Stable Version: { stable_version }")
                    if latest_version:
                        print(f"Latest Version: { latest_version }")
                    if tips_type != "" and tips_content != "":
                        print(f"[{ tips_type.upper() }]{ tips_content }")
            except Exception as e:
                print(e)

def run_check_version():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(check_version_async())
    loop.close()

def check_version(global_storage, today):
    thread = threading.Thread(target = run_check_version)
    thread.start()
    #thread.join()
    global_storage.set("agently", "check_version_record", today)