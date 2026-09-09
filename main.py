import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 1. 超軽量なEmbeddingモデルをロード
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(title="Embedding API", description="テキストをリアルタイムでベクトル化するAPI")

# ユーザーから受け取るデータの形（型）を定義
class TextInput(BaseModel):
    text: str

# データを送る（POST）ためのエンドポイントを作成
@app.post("/embedding")
def get_embedding(data: TextInput):
    # テキストが空の場合は空のリストを返す
    if not data.text.strip():
        return {"error": "テキストが空です", "embedding": []}
    
    # ベクトル化を実行
    embedding = model.encode(data.text)
    
    # 扱いやすいように通常のPythonリスト（floatの配列）に変換して返す
    return {
        "text": data.text,
        "dimension": len(embedding),
        "embedding": embedding.tolist()
    }
