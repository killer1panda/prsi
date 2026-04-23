import asyncio
from api import analyze_text, TextAnalysisRequest

async def test():
    request = TextAnalysisRequest(text="This is a test text about cancellation and backlash")
    result = await analyze_text(request)
    print("Prediction result:", result)

if __name__ == "__main__":
    asyncio.run(test())