from functools import reduce
from typing import Dict, List

# from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

from BertBilstmCRF import BertBilstmCrf
from typing import Optional
from utils import process_token, token2word, calculate_word_label, pipeline
from fastapi import FastAPI
from pydantic import BaseModel


# 用于测试

class Document(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    content: str


app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.tokenizer = AutoTokenizer.from_pretrained('checkpoint')
app.model = BertBilstmCrf.from_pretrained('checkpoint')



@app.post('/find')
async def find(document: Document):
    res = pipeline(document.content, app.model, app.tokenizer)
    return res
