from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Query, SemanticMemory
from backend.db.session import get_db
from backend.schemas.query import QueryRequest, QueryResponse, QueryResult
from backend.services.retrieval import RetrievalProvider, get_retrieval_provider

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    provider: RetrievalProvider = Depends(get_retrieval_provider),
):
    # Check for empty or whitespace query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Resolve semantic memory aliases if present
    filters = dict(request.filters or {})
    if "camera_id" not in filters:
        try:
            memories = db.scalars(select(SemanticMemory)).all()
            q_lower = request.query.lower()
            for mem in memories:
                if mem.key and mem.key.lower() in q_lower and mem.camera_id:
                    filters["camera_id"] = mem.camera_id
                    break
        except Exception:
            pass

    # 2. Execute retrieval
    try:
        results = provider.search(
            query=request.query,
            filters=filters if filters else None,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Record query in DB
    try:
        db_query = Query(query_text=request.query, status="completed")
        db.add(db_query)
        db.commit()
    except Exception:
        db.rollback()

    # 4. Return formatted response
    return QueryResponse(
        query=request.query,
        status="ok",
        results=[
            QueryResult(**result)
            for result in results
        ],
    )