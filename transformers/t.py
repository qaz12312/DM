import asyncio
from async_google_trans_new import AsyncTranslator


def ensure_future(func):
    def wrapper(*argv, **kargv):
        return asyncio.ensure_future(func(*argv, **kargv))
    return wrapper


@ensure_future
async def translate(text: str):
    g = AsyncTranslator()
    return await g.translate(text, "en")


def run(tasks):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    return [t.result() for t in tasks]

print(run([translate('蘋果'),translate('香蕉')]))