from fastapi import FastAPI

from model import TranslationRequest, TranslationResponse
from translate import packup, run

app = FastAPI()

@app.post('/translate')
def en_translate_to_zh(source:TranslationRequest) -> TranslationResponse:
    result = run(source)
    return packup(result)
